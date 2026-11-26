# -*- coding: utf-8 -*-
"""fit_historical_fold_models.py

Historical <=2022 source-aware models paired with the FROZEN nested outer folds.

This script DOES NOT optimize hyperparameters and DOES NOT recompute MED/PED/Q.
For each outer fold i:
  * Read the SAME source-aware train/test files as the nested current model.
  * Read theta_i* from the frozen nested_outer_fold{i}_meta.json.
  * Restrict TRAIN rows to Year <= year_cutoff (TEST rows are unchanged).
  * Fit a new XGBRegressor(theta_i*) on that historical training subset.
  * Evaluate it on the same held-out test records as the current nested model.
  * Validate that current-model predictions reproduce BOTH stored nested OOF
    and formal evidence OOF to catch mismatched data/copies of the project.

The hyperparameters were selected with the full contemporary outer training
pool (possibly including post-2022 records). This is a retrospective
fixed-configuration comparison, NOT a 2022-time-causal evaluation.

Run from the project directory:
    python fit_historical_fold_models.py
    !python fit_historical_fold_models.py

Main outputs (new directory, original models untouched):
  models/xgb_nested_temporal_2022/
    nested_temporal_fold{i}.joblib
    nested_temporal_fold{i}_meta.json
    fold{i}_train_2022.csv
    fold{i}_test_predictions_2022.csv
    temporal_fold_metrics.csv
    temporal_oof_predictions.csv
    temporal_pooled_summary.csv
    temporal_best_params.csv
    run_config.json
    nested_temporal_results.xlsx

Requirements: numpy pandas scikit-learn xgboost joblib openpyxl (Excel optional).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FOLDS = (1, 2, 3, 4, 5)
YEAR_COL = "Year"
SOURCE_COL = "source_key"
SAMPLE_COL = "sample_id"
TARGET_COL = "lgkp"
MODEL_FIXED_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "booster": "gbtree",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Historical XGBoost using fold-specific nested CV parameters")
    p.add_argument("--project-root", default=None, help="Default: directory holding this script")
    p.add_argument("--year-cutoff", type=int, default=2022)
    p.add_argument("--data-dir", default="splits/source_kfold")
    p.add_argument("--nested-model-dir", default="models/xgb_source_nested_kfold")
    p.add_argument("--current-evidence-csv", default="models/nested_evidence_final/pooled_source_oof_evidence.csv")
    p.add_argument("--output-dir", default=None,
                   help="Default: models/xgb_nested_temporal_<year-cutoff>")
    p.add_argument("--expected-records", type=int, default=613,
                   help="Expected number of distinct held-out records; 0 disables count check")
    p.add_argument("--n-jobs", type=int, default=None,
                   help="Override runtime XGBoost threads only; default saved fixed_params n_jobs")
    p.add_argument("--skip-frozen-check", action="store_true",
                   help="For isolated smoke tests only: skip saved OOF and formal evidence matching")
    p.add_argument("--no-excel", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow overwriting files from a prior historical run in output-dir")
    args = p.parse_args()
    if args.expected_records < 0 or args.year_cutoff < 1900:
        p.error("expected-records must be >=0 and year-cutoff must be >=1900")
    if args.n_jobs is not None and args.n_jobs < 1:
        p.error("--n-jobs must be positive")
    return args


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def metrics(y_true, y_pred) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape != yp.shape or not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Invalid shapes/non-finite target or prediction")
    return {
        "R2": float(r2_score(yt, yp)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(math.sqrt(mean_squared_error(yt, yp))),
    }


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def require_ids(df: pd.DataFrame, context: str) -> None:
    require_columns(df, [SAMPLE_COL, SOURCE_COL], context)
    if df[SAMPLE_COL].isna().any() or df[SAMPLE_COL].astype(str).duplicated().any():
        raise ValueError(f"{context}: missing/duplicate sample_id")
    if df[SOURCE_COL].isna().any():
        raise ValueError(f"{context}: missing source_key")


def as_model_input(model, frame: pd.DataFrame, features: list[str]):
    names = model.get_booster().feature_names
    if names is not None:
        if list(names) != features:
            raise RuntimeError("Frozen current model feature names differ from meta feature_cols")
        return frame[features].astype(float)
    return frame[features].to_numpy(dtype=float)


def check_numerics(frame: pd.DataFrame, features: list[str], context: str) -> None:
    require_columns(frame, features + [TARGET_COL], context)
    X = frame[features].to_numpy(dtype=float)
    y = frame[TARGET_COL].to_numpy(dtype=float)
    if np.isinf(X).any() or not np.isfinite(y).all():
        raise RuntimeError(f"{context}: invalid numeric input")


def validate_params(fold: int, meta: dict, current_model, n_jobs: int | None):
    if int(meta.get("fold", -1)) != fold:
        raise RuntimeError(f"Fold {fold}: nested meta fold identifier differs")
    best = meta.get("best_params")
    if not isinstance(best, dict) or not best:
        raise RuntimeError(f"Fold {fold}: nested meta lacks best_params")
    fixed = meta.get("fixed_params", MODEL_FIXED_PARAMS)
    if not isinstance(fixed, dict):
        raise RuntimeError(f"Fold {fold}: nested meta fixed_params invalid")
    params = dict(fixed)
    params.update(best)
    params.setdefault("random_state", 42)
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("eval_metric", "rmse")
    params.setdefault("booster", "gbtree")
    params.setdefault("tree_method", "hist")
    # Verify exact hyperparameter provenance against the saved CURRENT model.
    current = current_model.get_params()
    for key, value in params.items():
        if key == "n_jobs":
            continue
        present = current.get(key)
        if present is None and value is not None:
            raise RuntimeError(f"Fold {fold}: current model lacks parameter {key}")
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            equal = (present is not None and np.isclose(float(present), float(value), rtol=1e-10, atol=1e-12))
        else:
            equal = present == value
        if not equal:
            raise RuntimeError(f"Fold {fold}: parameter {key} mismatch between meta and frozen current model: {value} != {present}")
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    return params, dict(best), dict(fixed)


def assert_against_frozen(data: pd.DataFrame, frozen: pd.DataFrame, fold: int,
                          prediction_col: str, context: str, tol: float = 2e-6) -> dict:
    required = [SAMPLE_COL, "fold", "y_true", prediction_col]
    require_columns(frozen, required, context)
    require_ids(frozen, context)
    check = data[[SAMPLE_COL, "fold", "y_true", "y_pred_current"]].merge(
        frozen[required], on=SAMPLE_COL, how="outer", validate="one_to_one",
        indicator=True, suffixes=("_computed", "_saved"))
    if len(check) != len(data) or not check["_merge"].eq("both").all():
        raise RuntimeError(f"Fold {fold}: {context} sample coverage mismatch")
    if not check.fold_computed.eq(check.fold_saved).all():
        raise RuntimeError(f"Fold {fold}: {context} fold identifier mismatch")
    for left, right in (("y_true_computed", "y_true_saved"), ("y_pred_current", prediction_col)):
        a = check[left].to_numpy(dtype=float)
        b = check[right].to_numpy(dtype=float)
        d = float(np.max(np.abs(a - b)))
        if not np.isfinite(d) or d > tol:
            raise RuntimeError(f"Fold {fold}: {context} {left} vs {right}: max diff {d:.8g} > {tol}")
    return {"fold": fold, "source": context, "n": len(data), "max_abs_prediction_diff":
            float(np.max(np.abs(check.y_pred_current.to_numpy(dtype=float) -
                                    check[prediction_col].to_numpy(dtype=float))))}


def excel_export(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    import openpyxl
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(color="FFFFFF", bold=True)
                cell.fill = PatternFill("solid", fgColor="17365D")
            for i, col in enumerate(frame.columns, start=1):
                vals = [str(x) for x in frame[col].head(20).dropna()]
                ws.column_dimensions[get_column_letter(i)].width = min(37, max(13, len(str(col))+2, *(len(v)+1 for v in vals)))
    print(f"Excel saved: {path}", flush=True)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd()
    data_dir = resolve_path(root, args.data_dir)
    nested_dir = resolve_path(root, args.nested_model_dir)
    formal_path = resolve_path(root, args.current_evidence_csv)
    output_dir = resolve_path(root, args.output_dir or f"models/xgb_nested_temporal_{args.year_cutoff}")
    nested_oof_path = nested_dir / "nested_oof_predictions.csv"

    protected = (root, data_dir, nested_dir, formal_path.parent)
    # Output may be under project root; must not be the root or any protected input directory.
    if output_dir == root or any(output_dir == p or p in output_dir.parents or output_dir in p.parents
                                 for p in protected[1:]):
        raise ValueError(f"Unsafe output directory: {output_dir}")
    required = [data_dir / f"fold{f}_{side}.csv" for f in FOLDS for side in ("train", "test")]
    for f in FOLDS:
        required.extend([nested_dir / f"nested_outer_fold{f}.joblib",
                         nested_dir / f"nested_outer_fold{f}_meta.json"])
    if not args.skip_frozen_check:
        required += [nested_oof_path, formal_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output folder not empty: {output_dir}. Use --overwrite to rerun deliberately.")

    # Freeze the input row sets and validate all five folds BEFORE beginning model fits.
    payloads = []
    all_test_ids = []
    test_source_sets = []
    for fold in FOLDS:
        full_train = pd.read_csv(data_dir / f"fold{fold}_train.csv", low_memory=False)
        test = pd.read_csv(data_dir / f"fold{fold}_test.csv", low_memory=False)
        meta = json.loads((nested_dir / f"nested_outer_fold{fold}_meta.json").read_text(encoding="utf-8"))
        features = list(meta.get("feature_cols", []))
        if len(features) != 32 or len(set(features)) != 32:
            raise RuntimeError(f"Fold {fold}: expected 32 unique features in nested meta")
        for label, df in (("full train", full_train), ("fixed test", test)):
            require_ids(df, f"Fold {fold} {label}")
            check_numerics(df, features, f"Fold {fold} {label}")
        require_columns(full_train, [YEAR_COL], f"Fold {fold} full train")
        if set(full_train[SAMPLE_COL].astype(str)) & set(test[SAMPLE_COL].astype(str)):
            raise RuntimeError(f"Fold {fold}: train/test sample_id overlap")
        tr_sources = set(full_train[SOURCE_COL].astype(str))
        te_sources = set(test[SOURCE_COL].astype(str))
        if tr_sources & te_sources:
            raise RuntimeError(f"Fold {fold}: train/test source leakage")
        year = pd.to_numeric(full_train[YEAR_COL], errors="coerce")
        hist_train = full_train.loc[year <= args.year_cutoff].copy()
        if len(hist_train) <= 10:
            raise RuntimeError(f"Fold {fold}: historical train has too few records: {len(hist_train)}")
        if year.loc[hist_train.index].isna().any() or (year.loc[hist_train.index] > args.year_cutoff).any():
            raise RuntimeError(f"Fold {fold}: invalid historical year filter")
        check_numerics(hist_train, features, f"Fold {fold} historical train")
        if set(hist_train[SOURCE_COL].astype(str)) & te_sources:
            raise RuntimeError(f"Fold {fold}: historical train/test source overlap")
        if meta.get("n_train") is not None and int(meta["n_train"]) != len(full_train):
            raise RuntimeError(f"Fold {fold}: train count inconsistent with saved nested metadata")
        if meta.get("n_test") is not None and int(meta["n_test"]) != len(test):
            raise RuntimeError(f"Fold {fold}: test count inconsistent with saved nested metadata")
        all_test_ids.extend(test[SAMPLE_COL].astype(str).tolist())
        test_source_sets.append(te_sources)
        payloads.append((fold, full_train, test, hist_train, meta, features, int(year.isna().sum())))
    if len(all_test_ids) != len(set(all_test_ids)):
        raise RuntimeError("Across folds: duplicate held-out sample_id")
    if args.expected_records and len(all_test_ids) != args.expected_records:
        raise RuntimeError(f"Expected {args.expected_records} held-out records, got {len(all_test_ids)}")
    for i in range(5):
        for j in range(i+1, 5):
            if test_source_sets[i] & test_source_sets[j]:
                raise RuntimeError(f"Across folds {i+1}/{j+1}: repeated held-out literature source")

    frozen = pd.read_csv(nested_oof_path, low_memory=False) if not args.skip_frozen_check else None
    formal = pd.read_csv(formal_path, low_memory=False) if not args.skip_frozen_check else None
    if frozen is not None and (len(frozen) != len(all_test_ids) or len(formal) != len(all_test_ids)):
        raise RuntimeError("Saved current OOF does not have same number of records as fixed test folds")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "purpose": "Nested source-aware retrospective <= cutoff model comparison, NOT temporal forecast",
        "year_cutoff": args.year_cutoff, "outer_folds": 5,
        "params_source": "Each nested_outer_fold{i}_meta.json best_params + fixed_params",
        "hyperparameter_rule": "theta_i is frozen within historical/current pair but can differ across outer folds",
        "training_rule": "Only historical training rows satisfy Year <= cutoff; test set unchanged",
        "caveat": "theta_i was optimized on the corresponding current outer training pool, potentially including post-cutoff records; not a strict as-of-cutoff simulation",
        "data_dir": str(data_dir), "current_model_dir": str(nested_dir),
        "current_frozen_oof": None if args.skip_frozen_check else str(nested_oof_path),
        "current_formal_evidence": None if args.skip_frozen_check else str(formal_path),
        "output_dir": str(output_dir), "n_jobs_override": args.n_jobs,
        "no_med_ped_or_quadrants_computed": True,
    }
    write_json(output_dir / "run_config.json", config)
    print("=" * 93, flush=True)
    print("NESTED SOURCE-AWARE HISTORICAL TRAINING: FOLD-SPECIFIC FROZEN THETA", flush=True)
    print(f"Project: {root}", flush=True)
    print(f"Nested current models: {nested_dir}", flush=True)
    print(f"Source-aware folds: {data_dir}", flush=True)
    print(f"Cutoff: Year <= {args.year_cutoff}; fixed TEST rows kept at all publication years", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Fixed current OOF strict verification: {not args.skip_frozen_check}", flush=True)
    print("No Optuna. No MED/PED. No changes to original model/evidence files.", flush=True)
    print("=" * 93, flush=True)

    fold_rows, oof_parts, best_rows, checks = [], [], [], []
    for fold, full_train, test, hist_train, meta, features, n_missing_year in payloads:
        print(f"\n{'='*22} HISTORICAL OUTER FOLD {fold}/5 {'='*22}", flush=True)
        current_model = joblib.load(nested_dir / f"nested_outer_fold{fold}.joblib")
        params, best, fixed = validate_params(fold, meta, current_model, args.n_jobs)
        current_pred = np.asarray(current_model.predict(as_model_input(current_model, test, features)), dtype=float)
        current_metrics = metrics(test[TARGET_COL].to_numpy(dtype=float), current_pred)

        hist_model = xgb.XGBRegressor(**params)
        X_hist = hist_train[features].to_numpy(dtype=float)
        y_hist = hist_train[TARGET_COL].to_numpy(dtype=float)
        hist_model.fit(X_hist, y_hist)
        hist_pred = np.asarray(hist_model.predict(test[features].to_numpy(dtype=float)), dtype=float)
        y_test = test[TARGET_COL].to_numpy(dtype=float)
        hist_metrics = metrics(y_test, hist_pred)

        ids = [c for c in (SAMPLE_COL, SOURCE_COL, "Ref_source_num", YEAR_COL,
                           "Temperature", "E_Time", "x_Co", "x_Ni", "x_Cr", "x_Al") if c in test]
        oof = test[ids].copy().reset_index(drop=True)
        oof.insert(0, "fold", fold)
        oof.insert(1, "fold_test_idx", np.arange(len(test)))
        oof["y_true"] = y_test
        oof["y_pred_2022"] = hist_pred  # kept for next evidence script compatibility (year cutoff default 2022)
        oof["y_pred_current"] = current_pred
        oof["error_2022"] = hist_pred - y_test
        oof["error_current"] = current_pred - y_test
        oof["abs_error_2022"] = np.abs(oof["error_2022"])
        oof["abs_error_current"] = np.abs(oof["error_current"])
        oof["delta_pred_current_minus_2022"] = current_pred - hist_pred
        oof["delta_abs_error_current_minus_2022"] = oof["abs_error_current"] - oof["abs_error_2022"]
        if args.year_cutoff != 2022:
            # The default downstream column names explicitly denote 2022.
            oof.rename(columns={c: c.replace("2022", str(args.year_cutoff))
                                for c in oof.columns if "2022" in c}, inplace=True)

        if not args.skip_frozen_check:
            checks.append(assert_against_frozen(oof, frozen.loc[frozen.fold.eq(fold)], fold,
                                                "y_pred_nested", "Saved source-nested OOF"))
            checks.append(assert_against_frozen(oof, formal.loc[formal.fold.eq(fold)], fold,
                                                "y_pred", "Formal nested evidence OOF"))
            print("  Saved/current nested OOF & formal evidence: STRICT PASS", flush=True)

        model_path = output_dir / f"nested_temporal_fold{fold}.joblib"
        joblib.dump(hist_model, model_path)
        train_path = output_dir / f"fold{fold}_train_{args.year_cutoff}.csv"
        write_csv(train_path, hist_train)
        write_csv(output_dir / f"fold{fold}_test_predictions_{args.year_cutoff}.csv", oof)
        hist_sources = int(hist_train[SOURCE_COL].nunique())
        fold_meta = {
            "fold": fold, "year_cutoff": args.year_cutoff,
            "current_nested_model_path": str(nested_dir / f"nested_outer_fold{fold}.joblib"),
            "current_nested_metadata_path": str(nested_dir / f"nested_outer_fold{fold}_meta.json"),
            "feature_cols": features, "target_col": TARGET_COL, "source_col": SOURCE_COL,
            "best_params": best, "fixed_params": fixed, "model_params": params,
            "hyperparameter_match_current": True,
            "n_train_full": len(full_train), "n_train_historical": len(hist_train),
            "n_test_fixed": len(test), "n_train_sources_full": int(full_train[SOURCE_COL].nunique()),
            "n_train_sources_historical": hist_sources,
            "n_test_sources": int(test[SOURCE_COL].nunique()), "source_overlap": 0,
            "n_missing_year_in_full_train": n_missing_year,
            "train_year_min": float(pd.to_numeric(hist_train[YEAR_COL]).min()),
            "train_year_max": float(pd.to_numeric(hist_train[YEAR_COL]).max()),
            "metrics_historical": hist_metrics, "metrics_current": current_metrics,
            "model_path": str(model_path), "historical_train_path": str(train_path),
            "fixed_test_path": str(data_dir / f"fold{fold}_test.csv"),
        }
        write_json(output_dir / f"nested_temporal_fold{fold}_meta.json", fold_meta)
        fold_rows.append({
            "fold": fold, "n_train_full": len(full_train),
            "n_train_historical": len(hist_train), "n_train_sources_historical": hist_sources,
            "n_test": len(test), "n_test_sources": int(test[SOURCE_COL].nunique()),
            "n_full_train_missing_year": n_missing_year,
            "historical_R2": hist_metrics["R2"], "historical_MAE": hist_metrics["MAE"],
            "historical_RMSE": hist_metrics["RMSE"],
            "current_R2": current_metrics["R2"], "current_MAE": current_metrics["MAE"],
            "current_RMSE": current_metrics["RMSE"],
            "delta_R2_current_minus_historical": current_metrics["R2"] - hist_metrics["R2"],
            "delta_MAE_current_minus_historical": current_metrics["MAE"] - hist_metrics["MAE"],
        })
        best_rows.append({"fold": fold, "n_train_historical": len(hist_train), **best})
        oof_parts.append(oof)
        print(f"  Historical train={len(hist_train)} ({hist_sources} sources); full train={len(full_train)}, fixed test={len(test)}", flush=True)
        print("  HISTORICAL | R2={R2:.5f} | MAE={MAE:.5f} | RMSE={RMSE:.5f}".format(**hist_metrics), flush=True)
        print("  CURRENT    | R2={R2:.5f} | MAE={MAE:.5f} | RMSE={RMSE:.5f}".format(**current_metrics), flush=True)
        print(f"  Year missing in full train (excluded from historical): {n_missing_year}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_parts, ignore_index=True)
    if len(oof_df) != len(all_test_ids) or oof_df[SAMPLE_COL].astype(str).duplicated().any():
        raise RuntimeError("Temporal pooled OOF sample coverage error")
    hist_col = "y_pred_2022" if args.year_cutoff == 2022 else f"y_pred_{args.year_cutoff}"
    hist_metrics = metrics(oof_df.y_true, oof_df[hist_col])
    curr_metrics = metrics(oof_df.y_true, oof_df.y_pred_current)
    pooled_df = pd.DataFrame([
        {"method": f"Historical <= {args.year_cutoff}, nested theta_i fixed", "n": len(oof_df), **hist_metrics},
        {"method": "Current, frozen nested outer models", "n": len(oof_df), **curr_metrics},
        {"method": "Delta: Current minus Historical", "n": len(oof_df),
         **{k: curr_metrics[k] - hist_metrics[k] for k in hist_metrics}},
    ])
    assert len(fold_df) == 5
    write_csv(output_dir / "temporal_fold_metrics.csv", fold_df)
    write_csv(output_dir / "temporal_oof_predictions.csv", oof_df)
    write_csv(output_dir / "temporal_pooled_summary.csv", pooled_df)
    write_csv(output_dir / "temporal_best_params.csv", pd.DataFrame(best_rows))
    write_json(output_dir / "current_prediction_reproduction.json", checks)
    if not args.no_excel:
        try:
            excel_export(output_dir / "nested_temporal_results.xlsx", {
                "Summary": pooled_df, "Fold_Metrics": fold_df,
                "Best_Params": pd.DataFrame(best_rows), "OOF_Predictions": oof_df,
                "Frozen_Consistency": pd.DataFrame(checks) if checks else pd.DataFrame({"note": ["Skipped"]}),
            })
        except ImportError:
            print("openpyxl not installed; CSVs were saved; Excel skipped.", flush=True)

    print("\n" + "="*93, flush=True)
    print("FINAL NESTED TEMPORAL HISTORICAL vs CURRENT SUMMARY", flush=True)
    print("="*93, flush=True)
    print(pooled_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    print("\nFOLD METRICS", flush=True)
    print(fold_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
    print(f"\nMatched held-out sample_ids: {len(oof_df)} (all distinct); historical/current held-out sets IDENTICAL", flush=True)
    print("Frozen current predictions reproduction: " + ("SKIPPED" if args.skip_frozen_check else "PASS"), flush=True)
    print("Important: hyperparameters are fixed WITHIN each fold historical/current pair, not across folds.", flush=True)
    print("Important: not a genuine as-of-2022 model because theta_i was selected on contemporary training data.", flush=True)
    print("NO MED/PED/Q computed. Historical-current evidence scaling will be defined separately.", flush=True)
    print(f"ALL CHECKS PASSED. Output: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
