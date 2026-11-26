# -*- coding: utf-8 -*-
"""Nested RANDOM 5x4 cross-validation for XGBoost oxidation-kinetics prediction.

Matched to train_source_aware_xgboost.py:
- Same 32 features, target lgkp, XGBoost search space, TPE/Optuna, 100 trials per outer fold.
- OUTER folds: the EXISTING splits/random_kfold/fold{1..5}_{train,test}.csv.
- INNER folds: shuffled record-level KFold(4, random_state=42+outer_fold).
- No grouping by DOI in the random validation arm, by design.
- Each held-out outer fold is excluded from hyperparameter tuning and model fitting.

Outputs (new directory, never overwrites original models):
    models/xgb_random_nested_kfold/nested_cv_results.xlsx
    models/xgb_random_nested_kfold/nested_outer_fold1.joblib ... fold5.joblib
    models/xgb_random_nested_kfold/nested_outer_fold{f}_meta.json
    models/xgb_random_nested_kfold/nested_oof_predictions.csv
    models/xgb_random_nested_kfold/nested_summary.csv
    models/xgb_random_nested_kfold/nested_fold_metrics.csv
    models/xgb_random_nested_kfold/nested_best_params.csv
    models/xgb_random_nested_kfold/nested_optuna_trials.csv
    models/xgb_random_nested_kfold/nested_vs_source_comparison.csv (if available)
    models/xgb_random_nested_kfold/outer_fold{f}_optuna.db (resume trials)

Run from the PROJECT ROOT (same directory containing splits/ and models/):
    !python train_random_xgboost.py
or:
    %run train_random_xgboost.py

Smoke test (use a SEPARATE output directory):
    !python train_random_xgboost.py --trials 2 --out-dir models/random_nested_smoke

Dependencies: numpy, pandas, scikit-learn, xgboost, optuna, joblib, openpyxl.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

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
SOURCE_COL = "source_key"  # for auditing ONLY; random CV never groups by this field
FOLDS = (1, 2, 3, 4, 5)
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
    p = argparse.ArgumentParser(description="Nested random five-fold XGBoost Optuna")
    p.add_argument("--data-dir", default="splits/random_kfold")
    p.add_argument("--out-dir", default="models/xgb_random_nested_kfold")
    p.add_argument("--original-model-dir", default="models/xgb_random_kfold")
    p.add_argument("--nested-source-dir", default="models/xgb_source_nested_kfold")
    p.add_argument("--trials", type=int, default=100, help="Optuna trials PER outer fold")
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--report-every", type=int, default=10)
    args = p.parse_args()
    if args.trials < 1 or args.inner_folds < 2 or args.n_jobs < 1 or args.report_every < 1:
        p.error("--trials>=1, --inner-folds>=2, --n-jobs>=1, --report-every>=1 required")
    return args


def as_path(value):
    return Path(value).expanduser().resolve()


def save_json(data, path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def metrics(y, pred):
    return {
        "R2": float(r2_score(y, pred)),
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
    }


def model_params(best, n_jobs):
    params = {
        "objective": "reg:squarederror", "eval_metric": "rmse", "booster": "gbtree",
        "tree_method": "hist", "random_state": SEED, "n_jobs": n_jobs,
    }
    params.update(best)
    return params


def propose(trial, n_jobs):
    # Identical to train_source_aware_xgboost.py.
    p = model_params({}, n_jobs)
    p.update({
        "max_depth": trial.suggest_int("max_depth", 4, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "gamma": trial.suggest_float("gamma", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1.0),
    })
    return p


def load_outer(data_dir, fold):
    tr = data_dir / f"fold{fold}_train.csv"
    te = data_dir / f"fold{fold}_test.csv"
    if not tr.is_file() or not te.is_file():
        raise FileNotFoundError(f"Missing random split fold {fold}: {tr}, {te}")
    train = pd.read_csv(tr, low_memory=False)
    test = pd.read_csv(te, low_memory=False)
    required = FEATURE_COLS + [TARGET_COL, SOURCE_COL, "sample_id"]
    for label, df in (("train", train), ("test", test)):
        missing = sorted(set(required) - set(df.columns))
        if missing:
            raise ValueError(f"Fold {fold} {label} missing columns: {missing}")
        if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
            raise ValueError(f"Fold {fold} {label} has null/duplicate sample_id")
        if df[SOURCE_COL].isna().any():
            raise ValueError(f"Fold {fold} {label} missing source_key")
        vals = df[FEATURE_COLS + [TARGET_COL]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError(f"Fold {fold} {label} contains non-finite features or target")
    train_ids = set(train.sample_id.astype(str))
    test_ids = set(test.sample_id.astype(str))
    if train_ids & test_ids:
        raise RuntimeError(f"Fold {fold} record overlap in outer random split")
    return train, test


def optional_original_predict(model_dir, fold, test):
    p = model_dir / f"xgb_fold{fold}.joblib"
    meta_path = model_dir / f"xgb_fold{fold}_meta.json"
    if not p.is_file():
        return None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if list(meta.get("feature_cols", [])) != FEATURE_COLS:
            print(f"  Original random fold {fold}: feature order differs; comparison skipped", flush=True)
            return None
    old = joblib.load(p)
    return np.asarray(old.predict(test[FEATURE_COLS].to_numpy(dtype=float)), dtype=float)


def summarize(folds_df, pooled_df, column, label):
    m = metrics(pooled_df["y_true"], pooled_df[column])
    return {
        "method": label,
        "mean_fold_R2": float(folds_df.R2.mean()),
        "sd_fold_R2": float(folds_df.R2.std(ddof=1)),
        "mean_fold_MAE": float(folds_df.MAE.mean()),
        "sd_fold_MAE": float(folds_df.MAE.std(ddof=1)),
        "mean_fold_RMSE": float(folds_df.RMSE.mean()),
        "sd_fold_RMSE": float(folds_df.RMSE.std(ddof=1)),
        "pooled_OOF_R2": m["R2"],
        "pooled_OOF_MAE": m["MAE"],
        "pooled_OOF_RMSE": m["RMSE"],
        "n_oof_records": len(pooled_df),
    }


def load_source_nested_summary(source_dir):
    path = source_dir / "nested_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df.loc[df.method == "Nested source-aware CV"]
    if len(row) != 1:
        print("WARNING: Nested source-aware summary not uniquely located; skip cross-split comparison")
        return None
    required = ["mean_fold_R2", "mean_fold_MAE", "mean_fold_RMSE",
                "pooled_OOF_R2", "pooled_OOF_MAE", "pooled_OOF_RMSE"]
    if not all(c in row.columns for c in required):
        print("WARNING: Nested source-aware summary columns incomplete; skip comparison")
        return None
    return row.iloc[0].to_dict()


def write_workbook(path, sheets):
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for title, frame in sheets:
            if frame is None or frame.empty:
                frame = pd.DataFrame({"information": ["Not available"]})
            frame.to_excel(writer, sheet_name=title, index=False)
            ws = writer.sheets[title]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(color="FFFFFF", bold=True)
                cell.fill = PatternFill(fill_type="solid", fgColor="17365D")
                cell.alignment = Alignment(vertical="center")
            for j, col in enumerate(frame.columns, start=1):
                values = [str(x) for x in frame[col].head(30) if pd.notna(x)]
                longest = max([len(str(col)) + 2] + [len(x) + 2 for x in values])
                ws.column_dimensions[get_column_letter(j)].width = min(43, max(12, longest))
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000000"


def main():
    args = parse_args()
    data_dir = as_path(args.data_dir)
    out_dir = as_path(args.out_dir)
    original_dir = as_path(args.original_model_dir)
    source_dir = as_path(args.nested_source_dir)
    if out_dir in {original_dir, source_dir}:
        raise ValueError("New nested random output directory must differ from existing model directories")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    config = {
        "method": "Nested random 5 outer KFold x 4 inner shuffled KFold; record-level splits",
        "outer_splits": "reuse existing splits/random_kfold/fold{f}_{train,test}.csv",
        "inner_splits": "KFold(n_splits=inner_folds, shuffle=True, random_state=42+outer_fold)",
        "outer_folds": 5,
        "inner_folds": args.inner_folds,
        "trials_per_outer_fold": args.trials,
        "optuna_objective": "mean inner-fold R2 (unweighted)",
        "optimization": "TPE sampler, seeds 42 + outer_fold",
        "model_seed": SEED,
        "n_jobs": args.n_jobs,
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "source_column": SOURCE_COL,
        "group_by_source": False,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "original_model_dir": str(original_dir),
        "nested_source_dir": str(source_dir),
        "search_space": SEARCH_SPACE,
        "disclaimer": "Original random CV used outer folds for global tuning; its comparison is diagnostic. "
                      "Random vs source-aware nested comparison isolates the split strategy with the "
                      "same outer/inner optimization structure (not the same training partitions).",
    }
    config_path = out_dir / "run_config.json"
    if config_path.exists():
        prev = json.loads(config_path.read_text(encoding="utf-8"))
        keys = ("data_dir", "inner_folds", "search_space", "n_jobs", "inner_splits", "features")
        for key in keys:
            if prev.get(key) != config[key]:
                raise RuntimeError(f"Existing output directory has incompatible {key}; use new --out-dir")
    save_json(config, config_path)

    print("=" * 94, flush=True)
    print("NESTED 5-FOLD RANDOM XGBOOST / OPTUNA (MATCHED TO SOURCE-AWARE NESTED)", flush=True)
    print(f"Data: {data_dir}\nOutput: {out_dir}", flush=True)
    print(f"Outer=5 RANDOM KFold saved splits; inner={args.inner_folds} shuffled RANDOM KFold; "
          f"trials/outer={args.trials}", flush=True)
    print(f"Features={len(FEATURE_COLS)}, objective=mean inner R2, n_jobs={args.n_jobs}", flush=True)
    print("source_key is used for auditing ONLY; no source grouping in random CV", flush=True)
    print("No original models or evidence files will be modified", flush=True)
    print("=" * 94, flush=True)

    fold_metrics, original_fold_metrics, best_rows, oof_parts, trial_parts, split_rows = [], [], [], [], [], []
    all_outer_test_ids = []
    expected_all_ids = None

    for outer in FOLDS:
        print(f"\n{'=' * 25} OUTER RANDOM FOLD {outer}/5 {'=' * 25}", flush=True)
        train, test = load_outer(data_dir, outer)
        train_ids = set(train.sample_id.astype(str))
        test_ids = set(test.sample_id.astype(str))
        universe = train_ids | test_ids
        if expected_all_ids is None:
            expected_all_ids = universe
        elif universe != expected_all_ids:
            raise RuntimeError(f"Fold {outer} has a different complete sample universe")
        all_outer_test_ids.extend(test.sample_id.astype(str).tolist())
        train_src = set(train[SOURCE_COL].astype(str))
        test_src = set(test[SOURCE_COL].astype(str))
        n_src_overlap = len(train_src & test_src)
        print(f"Outer train={len(train)} / {len(train_src)} sources; "
              f"test={len(test)} / {len(test_src)} sources; "
              f"overlapping sources={n_src_overlap}", flush=True)
        split_rows.append({"fold": outer, "n_train": len(train), "n_test": len(test),
                           "n_train_sources": len(train_src), "n_test_sources": len(test_src),
                           "n_overlapping_sources": n_src_overlap,
                           "n_unique_train_samples": len(train_ids), "n_unique_test_samples": len(test_ids)})

        X_train = train[FEATURE_COLS].to_numpy(dtype=float)
        y_train = train[TARGET_COL].to_numpy(dtype=float)
        X_test = test[FEATURE_COLS].to_numpy(dtype=float)
        y_test = test[TARGET_COL].to_numpy(dtype=float)
        inner = list(KFold(n_splits=args.inner_folds, shuffle=True,
                           random_state=SEED + outer).split(X_train))
        for ii, (it, iv) in enumerate(inner, start=1):
            if np.intersect1d(it, iv).size:
                raise RuntimeError(f"Outer {outer} inner {ii} indices overlap")

        db_path = (out_dir / f"outer_fold{outer}_optuna.db").resolve().as_posix()
        sampler = optuna.samplers.TPESampler(seed=SEED + outer)
        study = optuna.create_study(
            direction="maximize", study_name=f"nested_random_outer_{outer}", sampler=sampler,
            storage=f"sqlite:///{db_path}", load_if_exists=True,
        )

        def objective(trial):
            p = propose(trial, args.n_jobs)
            scores = []
            for it, iv in inner:
                m = xgb.XGBRegressor(**p)
                m.fit(X_train[it], y_train[it])
                pred = m.predict(X_train[iv])
                scores.append(float(r2_score(y_train[iv], pred)))
            for ii, score in enumerate(scores, start=1):
                trial.set_user_attr(f"inner{ii}_R2", score)
            return float(np.mean(scores))

        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        remaining = max(0, args.trials - completed)
        print(f"Optuna completed={completed}; new trials={remaining}", flush=True)
        if remaining:
            def report_progress(st, trial):
                done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in st.trials)
                if done == 1 or done % args.report_every == 0 or done >= args.trials:
                    print(f"  Fold {outer}: {done}/{args.trials}; best inner mean R2={st.best_value:.5f}",
                          flush=True)
            study.optimize(objective, n_trials=remaining, callbacks=[report_progress])

        best = study.best_params
        print(f"Best inner mean R2={study.best_value:.6f}", flush=True)
        print("Selected parameters:", best, flush=True)
        outer_model = xgb.XGBRegressor(**model_params(best, args.n_jobs))
        outer_model.fit(X_train, y_train)
        nested_pred = np.asarray(outer_model.predict(X_test), dtype=float)
        m_nested = metrics(y_test, nested_pred)
        print("NESTED HELD-OUT | R2={R2:.5f} | MAE={MAE:.5f} | RMSE={RMSE:.5f}".format(**m_nested), flush=True)

        model_path = out_dir / f"nested_outer_fold{outer}.joblib"
        joblib.dump(outer_model, model_path)
        save_json({
            "fold": outer,
            "n_train": len(train), "n_test": len(test),
            "n_train_sources": len(train_src), "n_test_sources": len(test_src),
            "n_overlapping_sources": n_src_overlap,
            "inner_cv_best_mean_R2": float(study.best_value),
            "best_params": best,
            "fixed_params": model_params({}, args.n_jobs),
            "outer_test_metrics": m_nested,
            "model_path": str(model_path),
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
        }, out_dir / f"nested_outer_fold{outer}_meta.json")
        fold_metrics.append({"fold": outer, "n_train": len(train), "n_test": len(test),
                             "n_train_sources": len(train_src), "n_test_sources": len(test_src),
                             "n_overlap_sources": n_src_overlap,
                             "inner_best_mean_R2": float(study.best_value), **m_nested})
        best_rows.append({"fold": outer, "inner_best_mean_R2": float(study.best_value), **best})

        cols = [c for c in ("sample_id", SOURCE_COL, "Ref_source_num", "Year", "Temperature",
                            "E_Time", "x_Co", "x_Ni", "x_Cr", "x_Al") if c in test]
        oof = test[cols].copy().reset_index(drop=True)
        oof.insert(0, "fold", outer)
        oof.insert(1, "fold_test_idx", np.arange(len(oof)))
        oof["y_true"] = y_test
        oof["y_pred_nested"] = nested_pred
        oof["error_nested"] = nested_pred - y_test
        oof["abs_error_nested"] = np.abs(oof["error_nested"])

        old_pred = optional_original_predict(original_dir, outer, test)
        if old_pred is not None:
            old_metric = metrics(y_test, old_pred)
            original_fold_metrics.append({"fold": outer, **old_metric})
            oof["y_pred_original"] = old_pred
            oof["abs_error_original"] = np.abs(old_pred - y_test)
            print("ORIGINAL RANDOM   | R2={R2:.5f} | MAE={MAE:.5f} | RMSE={RMSE:.5f}".format(
                **old_metric), flush=True)
        oof_parts.append(oof)
        oof.to_csv(out_dir / f"nested_outer_fold{outer}_oof.csv", index=False, encoding="utf-8-sig")
        trials = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
        trials.insert(0, "fold", outer)
        trials.rename(columns={"value": "inner_mean_R2"}, inplace=True)
        trials.to_csv(out_dir / f"nested_outer_fold{outer}_trials.csv", index=False, encoding="utf-8-sig")
        trial_parts.append(trials)
        pd.DataFrame(fold_metrics).to_csv(out_dir / "fold_metrics_checkpoint.csv", index=False)
        pd.concat(oof_parts, ignore_index=True).to_csv(out_dir / "oof_checkpoint.csv", index=False)

    if len(all_outer_test_ids) != len(set(all_outer_test_ids)):
        raise RuntimeError("Duplicate held-out sample_id across outer folds")
    if expected_all_ids is None or set(all_outer_test_ids) != expected_all_ids:
        raise RuntimeError("Some records not represented exactly once across outer tests")
    pooled = pd.concat(oof_parts, ignore_index=True)
    fold_df = pd.DataFrame(fold_metrics)
    best_df = pd.DataFrame(best_rows)
    trials_df = pd.concat(trial_parts, ignore_index=True)
    split_df = pd.DataFrame(split_rows)
    random_nested = summarize(fold_df, pooled, "y_pred_nested", "Nested random CV")
    summary_rows = [random_nested]
    if len(original_fold_metrics) == len(FOLDS):
        old_fold = pd.DataFrame(original_fold_metrics)
        old_summ = summarize(old_fold, pooled, "y_pred_original", "Original random global-tuned CV")
        summary_rows.append(old_summ)
        diff = {key: random_nested[key] - old_summ[key]
                for key in random_nested if key not in ("method", "n_oof_records")}
        diff.update({"method": "Delta: Nested random minus Original random",
                     "n_oof_records": len(pooled)})
        summary_rows.append(diff)
        old_fold.to_csv(out_dir / "original_fold_metrics.csv", index=False, encoding="utf-8-sig")

    source_summ = load_source_nested_summary(source_dir)
    nested_comparison = None
    if source_summ is not None:
        comparison_cols = ["mean_fold_R2", "mean_fold_MAE", "mean_fold_RMSE",
                           "pooled_OOF_R2", "pooled_OOF_MAE", "pooled_OOF_RMSE"]
        nested_comparison = pd.DataFrame([
            {"method": "Nested random CV", **{k: random_nested[k] for k in comparison_cols}},
            {"method": "Nested source-aware CV", **{k: float(source_summ[k]) for k in comparison_cols}},
            {"method": "Delta: source-aware minus random",
             **{k: float(source_summ[k]) - random_nested[k] for k in comparison_cols}},
        ])
        nested_comparison.to_csv(out_dir / "nested_vs_source_comparison.csv", index=False, encoding="utf-8-sig")
        print("\nNESTED RANDOM vs NESTED SOURCE-AWARE (both independent inner tuning):", flush=True)
        print(nested_comparison.to_string(index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
    else:
        print("\nNOTE: matched Nested source-aware summary not found; comparison skipped.", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "nested_summary.csv", index=False, encoding="utf-8-sig")
    save_json(summary.to_dict(orient="records"), out_dir / "nested_summary.json")
    fold_df.to_csv(out_dir / "nested_fold_metrics.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(out_dir / "nested_best_params.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(out_dir / "nested_oof_predictions.csv", index=False, encoding="utf-8-sig")
    trials_df.to_csv(out_dir / "nested_optuna_trials.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(out_dir / "outer_split_audit.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 94, flush=True)
    print("FINAL RANDOM NESTED CV RESULTS", flush=True)
    print(summary[["method", "mean_fold_R2", "sd_fold_R2", "mean_fold_MAE", "mean_fold_RMSE",
                   "pooled_OOF_R2", "pooled_OOF_MAE", "pooled_OOF_RMSE"]].to_string(
                       index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
    print("\nPer fold:", flush=True)
    print(fold_df[["fold", "n_train", "n_test", "n_overlap_sources", "R2", "MAE", "RMSE"]].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"), flush=True)
    print(f"OOF distinct records={len(set(all_outer_test_ids))} / total={len(pooled)}", flush=True)

    settings_df = pd.DataFrame([
        {"setting": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)}
        for k, v in config.items()
    ])
    xlsx = out_dir / "nested_cv_results.xlsx"
    try:
        write_workbook(xlsx, [
            ("Summary", summary), ("Fold_Metrics", fold_df), ("Best_Params", best_df),
            ("OOF_Predictions", pooled), ("Optuna_Trials", trials_df),
            ("Random_vs_SourceNested", nested_comparison), ("Outer_Split_Audit", split_df),
            ("Run_Config", settings_df),
        ])
        print(f"Excel saved: {xlsx}", flush=True)
    except (ImportError, PermissionError) as exc:
        print(f"WARNING: Excel export failed: {exc}; CSVs were saved successfully", flush=True)
    print(f"\nAll results: {out_dir}", flush=True)
    print("Random and source-aware Nested CV can now be compared under matched optimization levels.", flush=True)
    print("This experiment does NOT assess MED/PED calibration.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
