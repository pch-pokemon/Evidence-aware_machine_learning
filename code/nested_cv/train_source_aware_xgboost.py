# -*- coding: utf-8 -*-
"""
Nested source-aware 5-fold CV for oxidation-kinetics XGBoost.

This script estimates out-of-fold performance with nested source-aware validation.
The workflow does not overwrite the original source-aware XGBoost models or
recompute MED/PED evidence states.

Inputs (unchanged):
    splits/source_kfold/fold1_train.csv ... fold5_train.csv
    splits/source_kfold/fold1_test.csv  ... fold5_test.csv

Main output:
    models/xgb_source_nested_kfold/nested_cv_results.xlsx

Other outputs: models, Optuna trials, fold/pooled OOF CSVs, config, JSON.

Usage from your project root (same folder as train_xgb_source_kfold_optuna.py):
    python train_source_aware_xgboost.py

Optional quick smoke test:
    python train_source_aware_xgboost.py --trials 2 --inner-folds 3
    [Use a separate output directory for tests, e.g. --out-dir models/nested_smoke]

Requirements:
    pip install numpy pandas scikit-learn xgboost optuna joblib openpyxl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

# =============================================================================
# 1. Identical feature order and Optuna search space to original source-aware run
# =============================================================================
HT_COLS = [
    "solu_temp", "solu_time", "middle_temp", "middle_time",
    "aging_temp", "aging_time",
]
COMP_COLS = [
    "x_Co", "x_Al", "x_W", "x_Ni", "x_Cr", "x_Mo", "x_Fe", "x_Nb",
    "x_C", "x_Hf", "x_Si", "x_Ta", "x_Ti", "x_Y", "x_V", "x_B",
    "x_Zr", "x_Mn", "x_Sc", "x_La", "x_Re",
]
FEATURE_COLS = HT_COLS + COMP_COLS + [
    "Temperature", "E_Time", "has_solu", "has_middle", "has_aging",
]
TARGET_COL = "lgkp"
SOURCE_COL = "source_key"
FOLDS = range(1, 6)
SEED = 42

SEARCH_SPACE = {
    "max_depth": "int[4,50]",
    "learning_rate": "log_uniform[0.0001,0.01]",
    "n_estimators": "int[100,1000]",
    "min_child_weight": "uniform[2.0,10.0]",
    "subsample": "uniform[0.1,1.0]",
    "colsample_bytree": "uniform[0.1,1.0]",
    "gamma": "uniform[0.1,1.0]",
    "reg_alpha": "uniform[0.1,1.0]",
    "reg_lambda": "uniform[0.1,1.0]",
}


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost nested source-aware CV")
    parser.add_argument("--data-dir", default="splits/source_kfold",
                        help="Existing five-fold CSV directory")
    parser.add_argument("--out-dir", default="models/xgb_source_nested_kfold",
                        help="Separate output directory (original models remain untouched)")
    parser.add_argument("--original-model-dir", default="models/xgb_source_kfold",
                        help="Optional: original five-fold joblib models, for direct comparison")
    parser.add_argument("--trials", type=int, default=100,
                        help="Optuna trials PER OUTER FOLD (default 100)")
    parser.add_argument("--inner-folds", type=int, default=4,
                        help="GroupKFold splits inside each outer training set (default 4)")
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="XGBoost threads per fit; 8 as in original objective")
    parser.add_argument("--report-every", type=int, default=10,
                        help="Print progress every N completed trials")
    args = parser.parse_args()
    if args.trials < 1 or args.inner_folds < 2 or args.n_jobs < 1:
        parser.error("trials >= 1, inner-folds >= 2, n-jobs >= 1 required")
    return args


def as_path(path_string: str) -> Path:
    """Use project working directory, matching the original scripts."""
    return Path(path_string).expanduser().resolve()


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def metrics(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(yt, yp)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(math.sqrt(mean_squared_error(yt, yp))),
    }


def proposed_params(trial, n_jobs):
    # Deliberately identical ranges, sampling scales and model constants to
    # train_xgb_source_kfold_optuna.py (100-trial global Optuna experiment).
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": n_jobs,
        "max_depth": trial.suggest_int("max_depth", 4, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "gamma": trial.suggest_float("gamma", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1.0),
    }


def final_model_params(best_params, n_jobs):
    p = dict(
        objective="reg:squarederror", eval_metric="rmse", booster="gbtree",
        tree_method="hist", random_state=SEED, n_jobs=n_jobs,
    )
    p.update(best_params)
    return p


def load_outer_data(data_dir: Path, fold: int):
    tr_path = data_dir / f"fold{fold}_train.csv"
    te_path = data_dir / f"fold{fold}_test.csv"
    for path in (tr_path, te_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run your existing split_dataset.py first.")
    train = pd.read_csv(tr_path, low_memory=False)
    test = pd.read_csv(te_path, low_memory=False)
    required = FEATURE_COLS + [TARGET_COL, SOURCE_COL]
    for name, df in (("train", train), ("test", test)):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Fold {fold} {name} missing columns: {missing}")
        if df[SOURCE_COL].isna().any():
            raise ValueError(f"Fold {fold} {name} has null source_key")
        if df[TARGET_COL].isna().any():
            raise ValueError(f"Fold {fold} {name} has null lgkp")
    train_sources = set(train[SOURCE_COL].astype(str))
    test_sources = set(test[SOURCE_COL].astype(str))
    overlap = train_sources & test_sources
    if overlap:
        raise ValueError(f"Fold {fold}: outer source leakage: {list(overlap)[:5]}")
    if len(train_sources) < 2:
        raise ValueError(f"Fold {fold}: not enough distinct sources in training set")
    for name, df in (("train", train), ("test", test)):
        X = df[FEATURE_COLS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        y = pd.to_numeric(df[TARGET_COL], errors="raise").to_numpy(dtype=float)
        if np.isinf(X).any() or not np.isfinite(y).all():
            raise ValueError(f"Fold {fold} {name} contains infinite features or invalid target")
    return train, test


def original_prediction_if_available(original_dir, fold, test_df):
    """Optional reference predictions. Never participates in Nested-CV tuning."""
    model_path = original_dir / f"xgb_fold{fold}.joblib"
    meta_path = original_dir / f"xgb_fold{fold}_meta.json"
    if not model_path.exists():
        return None
    try:
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("feature_cols") != FEATURE_COLS:
                print(f"  [Baseline skipped] Fold {fold} original feature list differs.", flush=True)
                return None
        old_model = joblib.load(model_path)
        pred = old_model.predict(test_df[FEATURE_COLS].to_numpy(dtype=float))
        return np.asarray(pred, dtype=float)
    except Exception as exc:
        print(f"  [Baseline skipped] Could not load original model {fold}: {exc}", flush=True)
        return None


def summarize_results(fold_df: pd.DataFrame, all_oof: pd.DataFrame, prefix: str):
    m = metrics(all_oof["y_true"], all_oof[prefix])
    summary = {
        "method": "Nested source-aware CV" if prefix == "y_pred_nested" else "Original global-tuned CV",
        "mean_fold_R2": float(fold_df["R2"].mean()),
        "sd_fold_R2": float(fold_df["R2"].std(ddof=1)),
        "mean_fold_MAE": float(fold_df["MAE"].mean()),
        "sd_fold_MAE": float(fold_df["MAE"].std(ddof=1)),
        "mean_fold_RMSE": float(fold_df["RMSE"].mean()),
        "sd_fold_RMSE": float(fold_df["RMSE"].std(ddof=1)),
        "pooled_OOF_R2": m["R2"],
        "pooled_OOF_MAE": m["MAE"],
        "pooled_OOF_RMSE": m["RMSE"],
        "n_oof_records": int(len(all_oof)),
    }
    return summary


def export_workbook(path: Path, summary_df, folds_df, params_df, oof_df, trials_df,
                    comparison_df, config):
    # Excel output: requires openpyxl. CSVs are also saved independently.
    import openpyxl  # noqa: F401
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    sheets = [
        ("Summary", summary_df),
        ("Fold_Metrics", folds_df),
        ("Best_Params", params_df),
        ("OOF_Predictions", oof_df),
        ("Optuna_Trials", trials_df),
        ("Original_Comparison", comparison_df),
        ("Run_Config", pd.DataFrame([
            {"setting": k, "value": json.dumps(v, ensure_ascii=False)
             if isinstance(v, (list, dict)) else str(v)}
            for k, v in config.items()
        ])),
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets:
            if frame is None or frame.empty:
                frame = pd.DataFrame({"information": ["No records available"]})
            frame.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="17365D")
                cell.alignment = Alignment(vertical="center")
            ws.row_dimensions[1].height = 24
            for j, col in enumerate(frame.columns, start=1):
                # Avoid inspecting all rows to keep 613-row / 500-trial export responsive.
                values = [str(v) for v in frame[col].head(30) if pd.notna(v)]
                width = min(39, max(12, len(str(col)) + 2,
                                     *(min(39, len(v) + 1) for v in values)))
                ws.column_dimensions[get_column_letter(j)].width = width
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, float):
                        c.number_format = "0.000000"
    print(f"Excel saved: {path}", flush=True)


def main():
    args = parse_args()
    data_dir = as_path(args.data_dir)
    out_dir = as_path(args.out_dir)
    orig_dir = as_path(args.original_model_dir)
    if out_dir == orig_dir:
        raise ValueError("Output directory must differ from original model directory!")
    out_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    config = dict(
        method="Nested 5x4 GroupKFold (source_key), Optuna inside outer training folds only",
        outer_folds=5,
        inner_folds=args.inner_folds,
        requested_optuna_trials_per_outer_fold=args.trials,
        optuna_objective="mean of inner-fold R2 (unweighted)",
        outer_metrics="both mean+sample-SD across 5 folds and pooled 613-record OOF",
        random_state_model=SEED,
        seed_optuna_per_fold="42 + outer_fold",
        n_jobs=args.n_jobs,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        source_col=SOURCE_COL,
        search_space=SEARCH_SPACE,
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        original_model_dir=str(orig_dir),
        note="Original global-tuned CV has model-selection bias; comparison is diagnostic, not an independent holdout for original tuning.",
    )
    old_config_path = out_dir / "run_config.json"
    if old_config_path.is_file():
        prev = json.loads(old_config_path.read_text(encoding="utf-8"))
        for key in ["inner_folds", "n_jobs", "feature_cols", "data_dir", "search_space"]:
            if prev.get(key) != config[key]:
                raise RuntimeError(
                    f"Existing output directory has different {key}. "
                    "Use a NEW --out-dir for another experiment."
                )
    save_json(config, old_config_path)

    print("=" * 86, flush=True)
    print("NESTED 5-FOLD SOURCE-AWARE XGBOOST / OPTUNA", flush=True)
    print(f"Data: {data_dir}\nOutput: {out_dir}", flush=True)
    print(f"Outer folds=5, inner folds={args.inner_folds}, trials/outer={args.trials}", flush=True)
    print(f"Features={len(FEATURE_COLS)}, source column={SOURCE_COL}", flush=True)
    print("Original models and MED/PED pipeline will not be modified.", flush=True)
    print("=" * 86, flush=True)

    fold_metrics, best_params_rows, oof_parts, trial_parts, baseline_fold_rows = [], [], [], [], []
    all_test_source_sets = []

    for outer in FOLDS:
        print(f"\n{'=' * 25} OUTER FOLD {outer}/5 {'=' * 25}", flush=True)
        train_df, test_df = load_outer_data(data_dir, outer)
        groups = train_df[SOURCE_COL].astype(str).to_numpy()
        num_sources = len(set(groups))
        if num_sources < args.inner_folds:
            raise ValueError(f"Fold {outer}: {num_sources} sources < {args.inner_folds} inner folds")
        test_sources = set(test_df[SOURCE_COL].astype(str))
        all_test_source_sets.append(test_sources)
        print(f"Outer train: {len(train_df)} rows / {num_sources} sources", flush=True)
        print(f"Outer test : {len(test_df)} rows / {len(test_sources)} sources", flush=True)
        print("Source overlap between outer train/test: 0", flush=True)

        X_train = train_df[FEATURE_COLS].to_numpy(dtype=float)
        y_train = train_df[TARGET_COL].to_numpy(dtype=float)
        X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
        y_test = test_df[TARGET_COL].to_numpy(dtype=float)
        inner_splits = list(GroupKFold(n_splits=args.inner_folds).split(
            X_train, y_train, groups=groups
        ))
        for i, (idx_tr, idx_va) in enumerate(inner_splits, start=1):
            overlap = set(groups[idx_tr]) & set(groups[idx_va])
            if overlap:
                raise RuntimeError(f"Inner group leakage in outer {outer}, inner {i}")

        db_path = (out_dir / f"outer_fold{outer}_optuna.db").resolve().as_posix()
        # Persistent trial storage: running again resumes unfinished trials.
        sampler = optuna.samplers.TPESampler(seed=SEED + outer)
        study = optuna.create_study(
            direction="maximize", study_name=f"nested_source_outer_{outer}",
            sampler=sampler, storage=f"sqlite:///{db_path}", load_if_exists=True,
        )

        def objective(trial):
            p = proposed_params(trial, args.n_jobs)
            inner_r2 = []
            for tr_idx, va_idx in inner_splits:
                model = xgb.XGBRegressor(**p)
                model.fit(X_train[tr_idx], y_train[tr_idx])
                pred = model.predict(X_train[va_idx])
                inner_r2.append(r2_score(y_train[va_idx], pred))
            for j, score in enumerate(inner_r2, start=1):
                trial.set_user_attr(f"inner{j}_R2", float(score))
            return float(np.mean(inner_r2))

        complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        remaining = max(0, args.trials - complete)
        print(f"Optuna completed: {complete}; running: {remaining} new trials", flush=True)
        if remaining:
            def report_progress(st, trial):
                done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in st.trials)
                if done == 1 or done % args.report_every == 0 or done == args.trials:
                    print(f"  Outer {outer}: {done}/{args.trials} trials; "
                          f"best inner mean R2={st.best_value:.5f}", flush=True)
            study.optimize(objective, n_trials=remaining, callbacks=[report_progress])

        best_params = study.best_params
        print(f"Best inner mean R2: {study.best_value:.6f}", flush=True)
        print("Selected parameters:", best_params, flush=True)

        # Only now fit on ALL OUTER TRAINING ROWS, then evaluate untouched outer test.
        outer_model = xgb.XGBRegressor(**final_model_params(best_params, args.n_jobs))
        outer_model.fit(X_train, y_train)
        y_pred = np.asarray(outer_model.predict(X_test), dtype=float)
        sc = metrics(y_test, y_pred)
        print("OUTER HELD-OUT | R2={R2:.5f} | MAE={MAE:.5f} | RMSE={RMSE:.5f}".format(**sc), flush=True)

        model_path = out_dir / f"nested_outer_fold{outer}.joblib"
        joblib.dump(outer_model, model_path)
        save_json({
            "fold": outer,
            "n_train": len(train_df), "n_test": len(test_df),
            "n_train_sources": num_sources, "n_test_sources": len(test_sources),
            "inner_cv_best_mean_R2": study.best_value,
            "best_params": best_params, "fixed_params": final_model_params({}, args.n_jobs),
            "outer_test_metrics": sc, "model_path": str(model_path),
            "feature_cols": FEATURE_COLS,
        }, out_dir / f"nested_outer_fold{outer}_meta.json")

        fold_metrics.append({
            "fold": outer, "n_train": len(train_df), "n_test": len(test_df),
            "n_train_sources": num_sources, "n_test_sources": len(test_sources),
            "inner_best_mean_R2": study.best_value, **sc,
        })
        best_params_rows.append({"fold": outer, "inner_best_mean_R2": study.best_value, **best_params})

        identity_cols = [c for c in [
            "sample_id", SOURCE_COL, "Ref_source_num", "Year", "Temperature", "E_Time",
            "x_Co", "x_Ni", "x_Cr", "x_Al",
        ] if c in test_df.columns]
        oof = test_df[identity_cols].copy().reset_index(drop=True)
        oof.insert(0, "fold", outer)
        oof.insert(1, "fold_test_idx", np.arange(len(oof)))
        oof["y_true"] = y_test
        oof["y_pred_nested"] = y_pred
        oof["error_nested"] = y_pred - y_test
        oof["abs_error_nested"] = np.abs(y_pred - y_test)

        baseline_pred = original_prediction_if_available(orig_dir, outer, test_df)
        if baseline_pred is not None:
            bs = metrics(y_test, baseline_pred)
            baseline_fold_rows.append({"fold": outer, **bs})
            oof["y_pred_original"] = baseline_pred
            oof["abs_error_original"] = np.abs(baseline_pred - y_test)
            print("ORIGINAL HELD-OUT | R2={R2:.5f} | MAE={MAE:.5f} | RMSE={RMSE:.5f}".format(**bs), flush=True)
        oof_parts.append(oof)
        oof.to_csv(out_dir / f"nested_outer_fold{outer}_oof.csv", index=False, encoding="utf-8-sig")

        trials = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
        trials.insert(0, "fold", outer)
        trials.rename(columns={"value": "inner_mean_R2"}, inplace=True)
        trials.to_csv(out_dir / f"nested_outer_fold{outer}_trials.csv", index=False, encoding="utf-8-sig")
        trial_parts.append(trials)

        # Save intermediate CSV checkpoint in case later folds are interrupted.
        pd.DataFrame(fold_metrics).to_csv(out_dir / "fold_metrics_checkpoint.csv", index=False)
        pd.concat(oof_parts, ignore_index=True).to_csv(out_dir / "oof_checkpoint.csv", index=False)

    # Source groups cannot appear in the held-out test subsets of different outer folds.
    for i in range(len(all_test_source_sets)):
        for j in range(i + 1, len(all_test_source_sets)):
            overlap = all_test_source_sets[i] & all_test_source_sets[j]
            if overlap:
                raise RuntimeError(f"Outer test source overlap folds {i+1} / {j+1}")

    fold_df = pd.DataFrame(fold_metrics)
    params_df = pd.DataFrame(best_params_rows)
    oof_df = pd.concat(oof_parts, ignore_index=True)
    trials_df = pd.concat(trial_parts, ignore_index=True)
    if "sample_id" in oof_df and oof_df["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id in outer OOF predictions")
    if len(oof_df) != sum(fold_df["n_test"]):
        raise RuntimeError("OOF size mismatch")

    nested_summary = summarize_results(fold_df, oof_df, "y_pred_nested")
    summary_records = [nested_summary]
    comparison_df = pd.DataFrame({"information": [
        "Original fold models not found/loaded for every fold; only Nested CV was evaluated."
    ]})
    if len(baseline_fold_rows) == len(FOLDS):
        baseline_df = pd.DataFrame(baseline_fold_rows)
        original_summary = summarize_results(baseline_df, oof_df, "y_pred_original")
        summary_records.append(original_summary)
        delta = {key: nested_summary[key] - original_summary[key]
                 for key in nested_summary if key not in ["method", "n_oof_records"]}
        delta["method"] = "Delta: Nested minus Original"
        delta["n_oof_records"] = len(oof_df)
        summary_records.append(delta)
        comparison_df = pd.DataFrame([original_summary, nested_summary, delta])
        baseline_df.to_csv(out_dir / "original_fold_metrics.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_records)
    fold_df.to_csv(out_dir / "nested_fold_metrics.csv", index=False, encoding="utf-8-sig")
    params_df.to_csv(out_dir / "nested_best_params.csv", index=False, encoding="utf-8-sig")
    oof_df.to_csv(out_dir / "nested_oof_predictions.csv", index=False, encoding="utf-8-sig")
    trials_df.to_csv(out_dir / "nested_optuna_trials.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "nested_summary.csv", index=False, encoding="utf-8-sig")
    save_json(summary_df.to_dict(orient="records"), out_dir / "nested_summary.json")

    print("\n" + "=" * 86, flush=True)
    print("FINAL RESULTS - USE POOLED OOF for manuscript's primary pooled estimate", flush=True)
    cols = ["method", "mean_fold_R2", "sd_fold_R2", "pooled_OOF_R2",
            "pooled_OOF_MAE", "pooled_OOF_RMSE"]
    print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
    print("\nPer outer fold:", flush=True)
    print(fold_df[["fold", "n_train", "n_test", "R2", "MAE", "RMSE"]].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"), flush=True)

    xlsx = out_dir / "nested_cv_results.xlsx"
    try:
        export_workbook(xlsx, summary_df, fold_df, params_df, oof_df, trials_df,
                        comparison_df, config)
    except (ImportError, PermissionError) as e:
        print(f"WARNING: Excel export failed ({e}). All CSVs remain available.", flush=True)
        print("Install openpyxl, or close Excel if the workbook is open, then rerun.", flush=True)
    print(f"\nAll results: {out_dir}", flush=True)
    print("Interpretation: Nested CV corrects hyperparameter-selection reuse in performance evaluation.")
    print("It does NOT replace or evaluate MED/PED state calibration.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
