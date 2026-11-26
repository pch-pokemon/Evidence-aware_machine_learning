# -*- coding: utf-8 -*-
"""compute_oof_evidence.py

Formal nested source-aware 5-fold pooled OOF MED/PED evidence rebuild.

This is the FINAL-FORM counterpart of build_pooled_oof_evidence_v2.py, using
already fitted nested OUTER models from train_source_aware_xgboost.py.
No XGBoost training or Optuna search takes place here.

Protocol for each outer fold i:
  1. Load the frozen nested outer model M_i, outer train_i, outer test_i.
  2. Use feature gain of THAT SAME M_i to set seven protocol PED weights W_i.
  3. Fit MED normalization on M_i's training-reference LOO distances and
     calculate held-out MED using precisely that normalization.
  4. Use the SAME training reference to standardize PED and estimate its
     training LOO reference distribution, with fixed physical main priors.
  5. Compute P40/45/50/55/60 MED and log1p(PED) thresholds using ONLY train_i.
  6. Assign Q1-Q4 for held-out samples using that fold's thresholds.
  7. Reconcile EVERY sample against the prior independent nested diagnostic.

Important limitations:
  - Training-reference LOO MED excludes a sample from neighbors/occupancy but
    not from the trees fitted on that training sample. It is NOT cross-fitted.
  - P50 means median of the TRAINING REFERENCE, not 50% coverage on new sources.
  - This code does NOT implement sensitivity studies on MED weights
    or protocol lambda; it freezes the established primary specification.
  - Full-data deployment and temporal analysis are SEPARATE operations.

Run from the project root (where evidence_core.py is located):
    python compute_oof_evidence.py
    # Jupyter: !python compute_oof_evidence.py

Options:
    --project-root PATH
    --model-dir models/xgb_source_nested_kfold
    --data-dir splits/source_kfold
    --main-weights-json models/fixed_evidence_weights/final_fixed_evidence_weights.json
    --diagnostic-csv models/nested_evidence_diagnostic/nested_vs_original_613.csv
    --output-dir models/nested_evidence_final
    --expected-records 613              (set 0 for artificial/smoke projects)
    --skip-diagnostic-check             (explicitly opt out; NOT recommended)
    --no-excel

Requires: numpy pandas scikit-learn xgboost joblib openpyxl(optional),
          evidence_core.py from the same oxidation-kinetics project.
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

from evidence_core import (
    EvidenceMetricConfig,
    MAIN_PHYSICAL_FEATURES,
    PROTOCOL_NUM_FEATURES,
    HT_FLAG_FEATURES,
    assign_quadrants,
    build_med_from_training_reference,
    calc_regression_metrics,
    compute_ped_external,
    compute_ped_training_loo,
    derive_training_thresholds,
    ensure_2d_leaf_matrix,
    load_main_weights,
    local_evidence_statistics,
    physical_standardization_to_dict,
    quadrant_error_statistics,
    transform_ped_for_threshold,
)

FOLDS = (1, 2, 3, 4, 5)
PERCENTILES = (40, 45, 50, 55, 60)
PRIMARY_PERCENTILE = 50
STATES = ("Q1", "Q2", "Q3", "Q4")
TARGET_COL = "lgkp"
SOURCE_COL = "source_key"

CFG = EvidenceMetricConfig(
    topk=10,
    lambda_protocol=0.15,
    ht_route_mismatch_penalty=1.0,
    ped_threshold_scale="log1p",
    clip_training_reference_normalization=True,
)


def parse_args():
    p = argparse.ArgumentParser(description="Formal Nested source-aware pooled OOF evidence")
    p.add_argument("--project-root", default=None,
                   help="Folder containing this script by default")
    p.add_argument("--model-dir", default="models/xgb_source_nested_kfold")
    p.add_argument("--data-dir", default="splits/source_kfold")
    p.add_argument("--main-weights-json",
                   default="models/fixed_evidence_weights/final_fixed_evidence_weights.json")
    p.add_argument("--diagnostic-csv",
                   default="models/nested_evidence_diagnostic/nested_vs_original_613.csv")
    p.add_argument("--output-dir", default="models/nested_evidence_final")
    p.add_argument("--expected-records", type=int, default=613,
                   help="Validate this many unique held-out rows; 0 disables expected count")
    p.add_argument("--skip-diagnostic-check", action="store_true",
                   help="Proceed without STRICT previous nested diagnostic comparison")
    p.add_argument("--no-excel", action="store_true")
    return p.parse_args()


def path_from(root: Path, s: str) -> Path:
    p = Path(s).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def check_safe_output(output: Path, inputs: list[Path]) -> None:
    for p in inputs:
        if output == p or p in output.parents or output in p.parents:
            raise RuntimeError("Output path overlaps input/protected directory: " + str(p))
    if output == output.parent:
        raise RuntimeError("Invalid output directory")


def write_csv(df: pd.DataFrame, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, encoding="utf-8-sig")


def write_json(obj: Any, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def model_inputs(model, frame: pd.DataFrame, feature_cols: list[str]):
    names = model.get_booster().feature_names
    if names is not None:
        if list(names) != feature_cols:
            raise RuntimeError("Fitted XGBoost feature names differ from saved model metadata")
        return frame.loc[:, feature_cols].astype(float)
    return frame.loc[:, feature_cols].to_numpy(dtype=float)


def protocol_gain(model, feature_cols: list[str]) -> dict[str, float]:
    """Match diagnose_nested_vs_original_evidence.py gain normalization exactly."""
    raw = model.get_booster().get_score(importance_type="gain")
    mapped = pd.Series(0.0, index=feature_cols, dtype=float)
    for key, val in raw.items():
        name = str(key)
        if name not in mapped.index and name.startswith("f"):
            try:
                idx = int(name[1:])
            except ValueError:
                continue
            if 0 <= idx < len(feature_cols):
                name = feature_cols[idx]
        if name in mapped.index:
            mapped.loc[name] = float(val)
    if not np.isfinite(mapped.to_numpy()).all() or (mapped < 0).any() or float(mapped.sum()) <= 0:
        raise RuntimeError("Model gain cannot be normalized")
    mapped = mapped / float(mapped.sum())
    weights = mapped.reindex(PROTOCOL_NUM_FEATURES, fill_value=0.0).copy()
    if float(weights.sum()) <= 0:
        weights[:] = 1.0 / len(weights)
    else:
        weights /= float(weights.sum())
    return {name: float(weights[name]) for name in PROTOCOL_NUM_FEATURES}


def identity_columns(frame: pd.DataFrame) -> list[str]:
    prefer = (
        "sample_id", "source_key", "Ref_source_num", "Year", "Temperature", "E_Time",
        "x_Co", "x_Ni", "x_Cr", "x_Al", "x_Ti", "x_W",
    )
    return [c for c in prefer if c in frame.columns]


def valid_frames(fold: int, train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    required = set(features) | set(MAIN_PHYSICAL_FEATURES) | set(PROTOCOL_NUM_FEATURES) | set(HT_FLAG_FEATURES)
    required |= {SOURCE_COL, TARGET_COL, "sample_id"}
    for label, frame in (("train", train), ("test", test)):
        if len(frame) == 0:
            raise RuntimeError(f"Fold {fold}: empty {label}")
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Fold {fold}: {label} missing {sorted(missing)}")
        if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
            raise RuntimeError(f"Fold {fold}: {label} has bad sample_id")
        if frame[SOURCE_COL].isna().any():
            raise RuntimeError(f"Fold {fold}: {label} has missing literature sources")
        if not np.isfinite(frame[TARGET_COL].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Fold {fold}: {label} target not finite")
        X = frame.loc[:, features].to_numpy(dtype=float)
        if np.isinf(X).any():
            raise RuntimeError(f"Fold {fold}: {label} features contain inf")
    if set(train.sample_id.astype(str)) & set(test.sample_id.astype(str)):
        raise RuntimeError(f"Fold {fold}: train/test sample_id overlap")
    overlap = set(train[SOURCE_COL].astype(str)) & set(test[SOURCE_COL].astype(str))
    if overlap:
        raise RuntimeError(f"Fold {fold}: {len(overlap)} overlapping literature sources")
    if len(train) <= CFG.topk:
        raise RuntimeError("Not enough train records for training LOO evidence")


def compute_one_fold(fold: int, model, meta: dict, train: pd.DataFrame,
                     test: pd.DataFrame, main_w: dict, w_protocol: dict):
    features = list(meta["feature_cols"])
    valid_frames(fold, train, test, features)
    train_x = model_inputs(model, train, features)
    test_x = model_inputs(model, test, features)
    y_true = test[TARGET_COL].to_numpy(dtype=float)
    pred = np.asarray(model.predict(test_x), dtype=float)
    if pred.shape != y_true.shape or not np.isfinite(pred).all():
        raise RuntimeError(f"Fold {fold}: invalid model predictions")
    met = calc_regression_metrics(y_true, pred)

    tr_leaf = ensure_2d_leaf_matrix(model.apply(train_x))
    te_leaf = ensure_2d_leaf_matrix(model.apply(test_x))
    tr_med, te_med, _, _, norm = build_med_from_training_reference(tr_leaf, te_leaf, CFG)
    tr_ped_raw = compute_ped_training_loo(train, main_w, w_protocol, CFG)
    te_ped_raw, neighbor_idx, _, phys = compute_ped_external(train, test, main_w, w_protocol, CFG)
    tr_ped_t = transform_ped_for_threshold(tr_ped_raw, CFG)
    te_ped_t = transform_ped_for_threshold(te_ped_raw, CFG)
    neighbors = local_evidence_statistics(train, neighbor_idx, TARGET_COL)
    if not (np.isfinite(tr_med.MED).all() and np.isfinite(te_med.MED).all()
            and np.isfinite(tr_ped_t).all() and np.isfinite(te_ped_t).all()):
        raise RuntimeError(f"Fold {fold}: nonfinite evidence values")

    thr = derive_training_thresholds(tr_med.MED.to_numpy(dtype=float), tr_ped_t, PERCENTILES)
    thr.insert(0, "fold", fold)
    tr_ref = train.loc[:, [c for c in ("sample_id", SOURCE_COL, "Ref_source_num", "Year",
                                        "Temperature", "E_Time") if c in train.columns]].copy().reset_index(drop=True)
    tr_ref.insert(0, "fold", fold)
    tr_ref.insert(1, "training_ref_idx", np.arange(len(tr_ref)))
    tr_ref = pd.concat([tr_ref, tr_med.reset_index(drop=True)], axis=1)
    tr_ref["PED_raw"] = tr_ped_raw
    tr_ref["PED_log1p"] = np.log1p(tr_ped_raw)
    tr_ref["PED_for_threshold"] = tr_ped_t

    oof = test.loc[:, identity_columns(test)].copy().reset_index(drop=True)
    oof.insert(0, "fold", fold)
    oof.insert(1, "fold_test_idx", np.arange(len(oof)))
    oof["y_true"] = y_true
    oof["y_pred"] = pred
    oof["error"] = pred - y_true
    oof["abs_error"] = np.abs(oof["error"])
    oof = pd.concat([oof, te_med.reset_index(drop=True)], axis=1)
    oof["PED_raw"] = te_ped_raw
    oof["PED_log1p"] = np.log1p(te_ped_raw)
    oof["PED_for_threshold"] = te_ped_t
    oof = pd.concat([oof, neighbors.reset_index(drop=True)], axis=1)
    for row in thr.itertuples(index=False):
        p = int(row.percentile)
        oof[f"quadrant_P{p}"] = assign_quadrants(
            oof.MED.to_numpy(dtype=float), te_ped_t,
            float(row.MED_threshold), float(row.PED_threshold)
        )
    oof["quadrant"] = oof["quadrant_P50"]
    p50 = thr.loc[thr.percentile.eq(PRIMARY_PERCENTILE)].iloc[0]
    med_p50 = float(p50.MED_threshold)
    ped_p50 = float(p50.PED_threshold)
    oof["MED_threshold_P50_fold"] = med_p50
    oof["PED_threshold_P50_fold"] = ped_p50
    oof["MED_minus_threshold_P50"] = oof.MED - med_p50
    oof["PED_minus_threshold_P50"] = oof.PED_for_threshold - ped_p50

    calibration = {
        "fold": fold,
        "model": "nested_source_aware_outer_fold",
        "model_best_params": meta.get("best_params"),
        "n_train": len(train),
        "n_test": len(test),
        "n_train_sources": int(train[SOURCE_COL].nunique()),
        "n_test_sources": int(test[SOURCE_COL].nunique()),
        "source_overlap": 0,
        "main_weights": main_w,
        "protocol_weights": w_protocol,
        "med_normalization": norm,
        "physical_standardization": physical_standardization_to_dict(phys),
        "thresholds_by_percentile": {
            str(int(r.percentile)): {
                "MED_threshold": float(r.MED_threshold),
                "PED_threshold": float(r.PED_threshold),
            } for r in thr.itertuples(index=False)
        },
        "calibration_semantics": (
            "Model-fitted training-reference LOO MED / training-reference LOO PED. "
            "Not source-cross-fitted and not probabilistic uncertainty calibration. "
            "MED P50 denotes training-reference median, not external 50% coverage."
        ),
        "config": CFG.to_dict(),
    }
    return oof, tr_ref, thr, met, calibration


def finite_compare(left: pd.Series, right: pd.Series, field: str,
                   tol: float = 1e-8) -> float:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise RuntimeError(f"Diagnostic {field}: incompatible/nonfinite numeric fields")
    max_delta = float(np.max(np.abs(a - b))) if a.size else 0.0
    if max_delta > tol:
        raise RuntimeError(f"Diagnostic {field} NOT reproduced: max abs difference={max_delta:.9g}, tol={tol}")
    return max_delta


def reconcile_diagnostic(pooled: pd.DataFrame, diag_path: Path) -> dict:
    diag = pd.read_csv(diag_path)
    needed = {"sample_id", "fold", "y_true", "y_pred_nested", "MED_nested",
              "PED_log1p_nested", "Q_nested", "MED_P50_nested", "PED_P50_nested"}
    needed |= {f"Q{p}_nested" for p in PERCENTILES}
    missing = needed - set(diag.columns)
    if missing:
        raise RuntimeError(f"Diagnostic missing {sorted(missing)}")
    if diag.sample_id.isna().any() or diag.sample_id.duplicated().any():
        raise RuntimeError("Diagnostic sample identifiers missing/duplicate")
    key = ["sample_id"] + [col for col in (
        "fold", "y_true", "y_pred_nested", "MED_nested", "PED_log1p_nested",
        "D_LEAF_nested", "D_SUPPORT_nested", "MED_P50_nested", "PED_P50_nested",
        "MED_margin_nested", "PED_margin_nested", "Q_nested",
        *(f"Q{p}_nested" for p in PERCENTILES)
    ) if col in diag.columns]
    check = pooled.merge(diag[key], on="sample_id", how="outer", validate="one_to_one", indicator=True)
    if len(check) != len(pooled) or not check._merge.eq("both").all():
        raise RuntimeError("New formal OOF sample_id coverage differs from previous nested diagnostic")
    report = {"n_compared": int(len(check))}
    # Diagnostic and formal records are aligned by sample_id, never CSV row order.
    report["n_fold_mismatches"] = int((check["fold_x"] != check["fold_y"]).sum())
    if report["n_fold_mismatches"]:
        raise RuntimeError("Fold identities differ between formal data and diagnostic")
    checks = {
        "y_true": ("y_true_x", "y_true_y", 1e-8),
        "prediction": ("y_pred", "y_pred_nested", 2e-6),
        "MED": ("MED", "MED_nested", 1e-8),
        "PED_log1p": ("PED_log1p", "PED_log1p_nested", 1e-8),
        "MED_P50": ("MED_threshold_P50_fold", "MED_P50_nested", 1e-8),
        "PED_P50": ("PED_threshold_P50_fold", "PED_P50_nested", 1e-8),
    }
    optional = {
        "D_LEAF": ("D_LEAF", "D_LEAF_nested", 1e-8),
        "D_SUPPORT": ("D_SUPPORT", "D_SUPPORT_nested", 1e-8),
        "MED_margin": ("MED_minus_threshold_P50", "MED_margin_nested", 1e-8),
        "PED_margin": ("PED_minus_threshold_P50", "PED_margin_nested", 1e-8),
    }
    for k, (a, b, tol) in {**checks, **optional}.items():
        if b in check.columns:
            report["max_abs_diff_" + k] = finite_compare(check[a], check[b], k, tol)
    q_bad = int((check.quadrant.astype(str) != check.Q_nested.astype(str)).sum())
    report["mismatched_primary_states"] = q_bad
    if q_bad:
        raise RuntimeError("P50 Q states differ from prior nested diagnostic")
    for p in PERCENTILES:
        col = f"Q{p}_nested"
        mismatched = int((check[f"quadrant_P{p}"].astype(str) != check[col].astype(str)).sum())
        report[f"mismatched_P{p}_states"] = mismatched
        if mismatched:
            raise RuntimeError(f"P{p} Q states differ from prior nested diagnostic")
    return report


def verify_saved_predictions(pooled: pd.DataFrame, pred_path: Path) -> dict:
    if not pred_path.is_file():
        raise FileNotFoundError("Missing original frozen nested model OOF predictions: " + str(pred_path))
    pred = pd.read_csv(pred_path)
    need = {"sample_id", "fold", "y_true", "y_pred_nested"}
    if need - set(pred):
        raise RuntimeError("Nested model prediction CSV missing " + str(sorted(need - set(pred))))
    if pred.sample_id.isna().any() or pred.sample_id.duplicated().any():
        raise RuntimeError("Saved model predictions contain bad sample_id")
    merged = pooled.merge(pred.loc[:, ["sample_id", "fold", "y_true", "y_pred_nested"]],
                          on="sample_id", how="outer", indicator=True, validate="one_to_one",
                          suffixes=("_formal", "_frozen"))
    if len(merged) != len(pooled) or not merged._merge.eq("both").all():
        raise RuntimeError("Saved nested OOF does not cover the same sample IDs")
    if not (merged.fold_formal == merged.fold_frozen).all():
        raise RuntimeError("Saved nested OOF fold IDs differ")
    return {
        "n_compared": len(merged),
        "max_abs_y_true_diff": finite_compare(merged.y_true_formal, merged.y_true_frozen, "nested_model_y_true"),
        "max_abs_prediction_diff": finite_compare(merged.y_pred, merged.y_pred_nested,
                                                  "nested_model_prediction", tol=2e-6),
    }


def save_excel(dst: Path, sheets: dict[str, pd.DataFrame]) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("Excel skipped (openpyxl unavailable); CSVs written.")
        return
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(dst, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            if df.empty:
                df = pd.DataFrame({"information": ["No records"]})
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            ws = writer.sheets[sheet_name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="17365D")
                cell.font = Font(bold=True, color="FFFFFF")
                cell.alignment = Alignment(vertical="center")
            for i, col in enumerate(df.columns, start=1):
                values = [len(str(x)) for x in df[col].head(30).dropna()]
                ws.column_dimensions[get_column_letter(i)].width = min(
                    40, max(12, len(str(col)) + 2, max(values, default=0) + 1))
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    if isinstance(c.value, float):
                        c.number_format = "0.000000"
    print("Excel saved:", dst, flush=True)


def main() -> None:
    args = parse_args()
    if args.expected_records < 0:
        raise ValueError("--expected-records must be >= 0")
    CFG.validate()
    root = Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd()
    data_dir = path_from(root, args.data_dir)
    model_dir = path_from(root, args.model_dir)
    prior_path = path_from(root, args.main_weights_json)
    diag_path = path_from(root, args.diagnostic_csv)
    out_dir = path_from(root, args.output_dir)

    if not prior_path.is_file():
        raise FileNotFoundError("Missing fixed prior weight JSON: " + str(prior_path))
    if not args.skip_diagnostic_check and not diag_path.is_file():
        raise FileNotFoundError("Missing diagnostic comparison CSV: " + str(diag_path) +
                                "\nTo opt out explicitly: --skip-diagnostic-check")
    check_safe_output(out_dir, [data_dir, model_dir, prior_path.parent, diag_path.parent])
    main_w = load_main_weights(prior_path)
    if set(main_w) != set(MAIN_PHYSICAL_FEATURES):
        raise RuntimeError("Physical priors do not cover main evidence features")

    print("=" * 97)
    print("FORMAL NESTED SOURCE-AWARE POOLED OOF EVIDENCE (FROZEN XGBOOST, NO TRAINING)")
    print("Project:", root)
    print("Outer models:", model_dir)
    print("Data splits:", data_dir)
    print("Physical priors:", prior_path)
    print("Prior nested diagnostic:", diag_path, "(strict)" if not args.skip_diagnostic_check else "(SKIPPED)")
    print("Output:", out_dir)
    print("Evidence config:", CFG.to_dict())
    print("Thresholds: fold-specific training-reference LOO P40-P60; P50 is primary")
    print("MED: fitted-tree training reference LOO (NOT cross-fitted); no 50% new-source coverage claim")
    print("=" * 97, flush=True)

    all_oof, all_thr, all_weights, all_metric, refs, fold_cfg = [], [], [], [], {}, {}
    all_heldout_sources: dict[int, set[str]] = {}
    seen_sample_ids: set[str] = set()
    common_features = None
    for i in FOLDS:
        train_path = data_dir / f"fold{i}_train.csv"
        test_path = data_dir / f"fold{i}_test.csv"
        model_path = model_dir / f"nested_outer_fold{i}.joblib"
        meta_path = model_dir / f"nested_outer_fold{i}_meta.json"
        for p in (train_path, test_path, model_path, meta_path):
            if not p.is_file():
                raise FileNotFoundError("Missing required fold input: " + str(p))
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("fold", i)) != i:
            raise RuntimeError(f"Fold {i} metadata ID mismatch")
        feats = list(meta["feature_cols"])
        if common_features is None:
            common_features = feats
        if common_features != feats:
            raise RuntimeError("Different model feature order across outer folds")
        if meta.get("n_train") is not None and int(meta["n_train"]) != len(train):
            raise RuntimeError(f"Fold {i}: metadata n_train mismatch")
        if meta.get("n_test") is not None and int(meta["n_test"]) != len(test):
            raise RuntimeError(f"Fold {i}: metadata n_test mismatch")
        valid_frames(i, train, test, feats)
        test_ids = set(test.sample_id.astype(str))
        if seen_sample_ids & test_ids:
            raise RuntimeError("Some held-out samples appear in multiple folds")
        seen_sample_ids.update(test_ids)
        all_heldout_sources[i] = set(test[SOURCE_COL].astype(str))
        for j, source_set in all_heldout_sources.items():
            if i != j and all_heldout_sources[i] & source_set:
                raise RuntimeError(f"Overlapping heldout literature sources in folds {i}/{j}")
        model = joblib.load(model_path)
        weights = protocol_gain(model, feats)
        for feat, value in weights.items():
            all_weights.append({"fold": i, "feature": feat, "weight": value,
                                "source": "nested_outer_model_gain"})
        print(f"\nComputing nested outer Fold {i}/5...", flush=True)
        oof, ref, thr, met, cal = compute_one_fold(i, model, meta, train, test, main_w, weights)
        all_oof.append(oof)
        refs[i] = ref
        all_thr.append(thr)
        all_metric.append({"fold": i, "n_train": len(train), "n_test": len(test),
                           "n_train_sources": train[SOURCE_COL].nunique(),
                           "n_test_sources": test[SOURCE_COL].nunique(), **met})
        cal["model_path"] = str(model_path)
        cal["meta_path"] = str(meta_path)
        fold_cfg[i] = cal
        q50 = oof.quadrant.value_counts().reindex(STATES, fill_value=0)
        print(f"  train={len(train)}, test={len(test)}; "
              f"R2={met['R2']:.5f}, MAE={met['MAE']:.5f}, RMSE={met['RMSE']:.5f}")
        print(f"  train P50 MED={thr.loc[thr.percentile.eq(50), 'MED_threshold'].iloc[0]:.6f} "
              f"PED={thr.loc[thr.percentile.eq(50), 'PED_threshold'].iloc[0]:.6f}; "
              f"Q counts={q50.to_dict()}", flush=True)

    pooled = pd.concat(all_oof, ignore_index=True)
    thresholds = pd.concat(all_thr, ignore_index=True)
    fold_metrics = pd.DataFrame(all_metric)
    weights_df = pd.DataFrame(all_weights)
    if args.expected_records and len(pooled) != args.expected_records:
        raise RuntimeError(f"Expected {args.expected_records} OOF records, received {len(pooled)}")
    if pooled.sample_id.isna().any() or pooled.sample_id.duplicated().any():
        raise RuntimeError("Final OOF contains duplicate/missing sample_id")
    if pooled[SOURCE_COL].nunique() != len(set().union(*all_heldout_sources.values())):
        raise RuntimeError("Pooled OOF source set inconsistent")
    if len(pooled) != int(fold_metrics.n_test.sum()):
        raise RuntimeError("Fold and pooled OOF row counts disagree")

    # Mandatory frozen-prediction check, independent of prior evidence diagnosis.
    saved_pred_check = verify_saved_predictions(pooled, model_dir / "nested_oof_predictions.csv")
    diag_check = None
    if not args.skip_diagnostic_check:
        diag_check = reconcile_diagnostic(pooled, diag_path)

    pooled_met = calc_regression_metrics(pooled.y_true, pooled.y_pred)
    main_states = quadrant_error_statistics(pooled, "quadrant")
    sensitivity_parts = []
    sensitivity_summary = []
    for p in PERCENTILES:
        s = quadrant_error_statistics(pooled, f"quadrant_P{p}")
        s.insert(0, "percentile", p)
        sensitivity_parts.append(s)
        q1 = s.loc[s.quadrant.eq("Q1")].iloc[0]
        q4 = s.loc[s.quadrant.eq("Q4")].iloc[0]
        sensitivity_summary.append({
            "percentile": p,
            "Q1_count": int(q1["count"]), "Q1_fraction": float(q1["fraction"]),
            "Q1_MAE": float(q1.MAE), "Q1_RMSE": float(q1.RMSE),
            "Q4_count": int(q4["count"]), "Q4_fraction": float(q4["fraction"]),
            "Q4_MAE": float(q4.MAE), "Q4_RMSE": float(q4.RMSE),
            "Q4_minus_Q1_MAE": float(q4.MAE - q1.MAE),
            "threshold_scope": "fold_specific_training_reference",
        })
    sensitivity_stats = pd.concat(sensitivity_parts, ignore_index=True)
    sensitivity_summary = pd.DataFrame(sensitivity_summary)
    thr_summary = thresholds.groupby("percentile", as_index=False).agg(
        MED_median=("MED_threshold", "median"),
        MED_min=("MED_threshold", "min"),
        MED_max=("MED_threshold", "max"),
        PED_median=("PED_threshold", "median"),
        PED_min=("PED_threshold", "min"),
        PED_max=("PED_threshold", "max"),
    )
    thr_summary["note"] = "Summary only. Not used to assign held-out states."
    qcount = dict(zip(main_states.quadrant, main_states["count"]))
    mean_fold = {k: float(fold_metrics[k].mean()) for k in ("R2", "MAE", "RMSE")}
    sd_fold = {k: float(fold_metrics[k].std(ddof=1)) for k in ("R2", "MAE", "RMSE")}
    results = {
        "n_oof_records": len(pooled),
        "n_heldout_sources": int(pooled[SOURCE_COL].nunique()),
        "pooled_OOF": pooled_met,
        "mean_fold": mean_fold,
        "sd_fold": sd_fold,
        "Q_state_counts_P50": {q: int(qcount.get(q, 0)) for q in STATES},
        "nested_frozen_prediction_reproduction": saved_pred_check,
        "nested_evidence_diagnostic_reproduction": diag_check,
        "med_calibration_note": "fitted-model training-reference LOO, not cross-fitted or probabilistically calibrated",
    }
    print("\n" + "=" * 97)
    print("FORMAL NESTED POOLED OOF SUMMARY")
    print("=" * 97)
    print("Pooled OOF metrics:", {k: round(v, 5) for k, v in pooled_met.items()})
    print("Mean fold metrics:", {k: round(v, 5) for k, v in mean_fold.items()})
    print("SD fold metrics:", {k: round(v, 5) for k, v in sd_fold.items()})
    print("Training-derived fold P50 thresholds:")
    print(thresholds.loc[thresholds.percentile.eq(50)].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("Q1-Q4 P50 state populations and errors:")
    print(main_states[["quadrant", "count", "fraction", "MAE", "RMSE"]].to_string(
        index=False, float_format=lambda v: f"{v:.5f}"))
    print("P40-P60 Q1/Q4 sensitivity:")
    print(sensitivity_summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("Frozen model OOF check: PASS", saved_pred_check)
    if diag_check is not None:
        print("Strict previous diagnostic reproduction: PASS", diag_check)
    else:
        print("WARNING: previous evidence diagnostic comparison deliberately SKIPPED")

    # Save only AFTER all fold/source/prediction/diagnostic checks pass.
    out_dir.mkdir(parents=True, exist_ok=True)
    wdir = out_dir / "weights"
    wdir.mkdir(exist_ok=True)
    outputs = {
        "pooled_source_oof_evidence.csv": pooled,
        "fold_oof_metrics.csv": fold_metrics,
        "fold_training_reference_thresholds.csv": thresholds,
        "fold_threshold_summary_NOT_FOR_ASSIGNMENT.csv": thr_summary,
        "quadrant_stats_P50.csv": main_states,
        "threshold_sensitivity_quadrant_stats.csv": sensitivity_stats,
        "threshold_sensitivity_summary.csv": sensitivity_summary,
        "protocol_fold_weights.csv": weights_df,
    }
    for name, frame in outputs.items():
        dst = (wdir / name) if name == "protocol_fold_weights.csv" else (out_dir / name)
        write_csv(frame, dst)
    for fold in FOLDS:
        write_csv(pooled.loc[pooled.fold.eq(fold)].copy(), out_dir / f"fold{fold}_oof_evidence.csv")
        write_csv(refs[fold], out_dir / f"fold{fold}_training_reference_evidence.csv")
        write_json(fold_cfg[fold], wdir / f"fold{fold}_evidence_calibration.json")
        write_json({
            "fold": fold, "version": "nested_outer_source_aware",
            "model_path": fold_cfg[fold]["model_path"],
            "main_weights": main_w,
            "protocol_weights": fold_cfg[fold]["protocol_weights"],
            "weight_rule": "nested outer model gain normalized over 32 inputs, renormalized over 7 protocol inputs",
            "evidence_config": CFG.to_dict(),
        }, wdir / f"fold{fold}_weights.json")
    write_json(results, out_dir / "pooled_oof_metrics.json")
    write_json({
        "version": "nested_outer_source_aware_pooled_OOF_evidence_v1",
        "n_oof": len(pooled), "folds": list(FOLDS),
        "model_dir": str(model_dir), "data_dir": str(data_dir),
        "main_prior_weights_file": str(prior_path),
        "prior_nested_diagnostic": str(diag_path) if diag_check is not None else None,
        "output_dir": str(out_dir),
        "features": common_features,
        "main_physical_features": list(MAIN_PHYSICAL_FEATURES),
        "protocol_features": list(PROTOCOL_NUM_FEATURES),
        "route_flags": list(HT_FLAG_FEATURES),
        "percentiles": list(PERCENTILES),
        "primary_percentile": PRIMARY_PERCENTILE,
        "evidence_config": CFG.to_dict(),
        "threshold_rule": "Each held-out record uses ONLY training reference P50 of its own frozen nested outer model",
        "gain_rule": "Each PED protocol weight vector from the corresponding nested outer model only",
        "MED_limit": "Training LOO self-excludes neighbors/occupancy but training samples were used to fit trees",
        "temporal_limit": "Historical analysis requires within-fold shared hyperparameters; source-aware retrospective comparison is not as-of-2022 independent model selection",
        "full_data_limit": "No full-data model is fitted here; deployment calibration remains a separate task",
        "fold_calibration_json": {str(i): str(wdir / f"fold{i}_evidence_calibration.json") for i in FOLDS},
        "checks": {"nested_prediction": saved_pred_check, "nested_diagnostic": diag_check},
    }, out_dir / "run_config.json")
    if not args.no_excel:
        save_excel(out_dir / "nested_pooled_oof_evidence.xlsx", {
            "Summary": pd.DataFrame([{"metric": key, "value": str(val)} for key, val in results.items()]),
            "Fold_Metrics": fold_metrics,
            "P50_Q_Stats": main_states,
            "P40_P60_Summary": sensitivity_summary,
            "P40_P60_Stats": sensitivity_stats,
            "Fold_Thresholds": thresholds,
            "Protocol_Weights": weights_df,
            "Threshold_Overview": thr_summary,
            "OOF_613": pooled,
        })
    print("\nALL CHECKS PASSED. Formal Nested evidence files saved in:", out_dir)
    print("Primary OOF:", out_dir / "pooled_source_oof_evidence.csv", flush=True)


if __name__ == "__main__":
    main()
