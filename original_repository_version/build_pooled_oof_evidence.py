# -*- coding: utf-8 -*-
"""
build_pooled_oof_evidence.py
================================
Exposure-time-aware pooled five-fold source-aware OOF evidence analysis.

Prerequisites
-------------
1) python split_dataset.py
2) python train_xgb_source_kfold_optuna.py
3) python compute_fixed_evidence_weights.py

Final exposure-time-aware evidence definition
-----------------
- All MED/PED math is imported from evidence_core.py.
- PED protocol space explicitly includes E_Time (oxidation exposure duration).
- Training-derived MED/PED reference thresholds are calculated independently in
  each fold and summarized with one global cross-fold threshold per percentile.
- P50 is the main Q1-Q4 partition; P40/P45/P50/P55/P60 are exported for sensitivity.

Outputs
-------
models/xgb_source_kfold/pooled_oof_evidence/
    pooled_source_oof_evidence.csv
    pooled_oof_metrics.json
    fold_oof_metrics.csv
    fold_training_reference_thresholds.csv
    global_training_derived_thresholds.csv
    quadrant_stats_P50.csv
    threshold_sensitivity_quadrant_stats.csv
    threshold_sensitivity_summary.csv
    fold*_oof_evidence.csv
    fold*_training_reference_evidence.csv
    run_config.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from evidence_core import (
    DEFAULT_EVIDENCE_CONFIG,
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
    identify_source_column,
    load_fixed_evidence_weights,
    local_evidence_statistics,
    physical_standardization_to_dict,
    quadrant_error_statistics,
    transform_ped_for_threshold,
)


# =============================================================================
# 1. Configuration
# =============================================================================

MODEL_DIR = Path("models/xgb_source_kfold")
DATA_DIR = Path("splits/source_kfold")
WEIGHT_JSON = Path("models/fixed_evidence_weights/final_fixed_evidence_weights.json")
OUTPUT_DIR = MODEL_DIR / "pooled_oof_evidence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS: Sequence[int] = (1, 2, 3, 4, 5)
SENSITIVITY_PERCENTILES: Sequence[int] = (40, 45, 50, 55, 60)
MAIN_PERCENTILE = 50

# One canonical config, shared with later temporal/prospective scripts.
EVIDENCE_CONFIG = EvidenceMetricConfig(
    topk=10,
    lambda_protocol=0.15,
    ht_route_mismatch_penalty=1.0,
    ped_threshold_scale="log1p",
    clip_training_reference_normalization=True,
)


# =============================================================================
# 2. I/O
# =============================================================================

def load_fold(fold: int):
    model_path = MODEL_DIR / f"xgb_fold{fold}.joblib"
    meta_path = MODEL_DIR / f"xgb_fold{fold}_meta.json"
    train_path = DATA_DIR / f"fold{fold}_train.csv"
    test_path = DATA_DIR / f"fold{fold}_test.csv"

    required = [model_path, meta_path, train_path, test_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    model = joblib.load(model_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return model, meta, train_df, test_df


def preferred_identity_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "sample_id", "source_key", "Ref_source_num", "Year",
        "Temperature", "E_Time",
        "x_Co", "x_Ni", "x_Cr", "x_Al", "x_Ti", "x_W",
    ]
    return [c for c in preferred if c in df.columns]


# =============================================================================
# 3. Per-fold analysis
# =============================================================================

def analyze_fold(
    fold: int,
    main_weights: Dict[str, float],
    protocol_weights: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], dict]:
    print(f"\n{'=' * 90}\nFold {fold}\n{'=' * 90}")
    model, meta, train_df, test_df = load_fold(fold)

    feature_cols: List[str] = list(meta["feature_cols"])
    target_col = str(meta["target_col"])

    # Model/evidence schema must explicitly include E_Time in the final pipeline.
    required_evidence_features = (
        list(MAIN_PHYSICAL_FEATURES)
        + list(PROTOCOL_NUM_FEATURES)
        + list(HT_FLAG_FEATURES)
    )
    missing_model_features = [f for f in required_evidence_features if f not in feature_cols]
    if missing_model_features:
        raise ValueError(
            f"Fold {fold}: model metadata missing final evidence features: {missing_model_features}"
        )

    for name, df in (("train", train_df), ("test", test_df)):
        missing = [c for c in feature_cols + [target_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Fold {fold} {name} data missing columns: {missing}")

    # Strict source-leakage check.
    source_col = identify_source_column(train_df)
    if source_col and source_col in test_df.columns:
        overlap = set(train_df[source_col].dropna()) & set(test_df[source_col].dropna())
        if overlap:
            raise RuntimeError(
                f"Fold {fold}: source leakage detected ({len(overlap)} overlapping sources)"
            )
        print("Source overlap check: PASS (0 overlapping sources)")

    # -------------------------
    # OOF prediction
    # -------------------------
    x_train = train_df[feature_cols].to_numpy(dtype=float)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(x_test), dtype=float)
    metrics = calc_regression_metrics(y_test, y_pred)
    print(
        f"OOF prediction metrics: R2={metrics['R2']:.4f}, "
        f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}"
    )

    # -------------------------
    # MED
    # -------------------------
    train_leaf = ensure_2d_leaf_matrix(model.apply(x_train))
    test_leaf = ensure_2d_leaf_matrix(model.apply(x_test))
    med_ref, med_test, _, _, med_norm = build_med_from_training_reference(
        train_leaf, test_leaf, EVIDENCE_CONFIG
    )

    # -------------------------
    # PED incl. E_Time
    # -------------------------
    ped_ref_raw = compute_ped_training_loo(
        train_df,
        main_weights,
        protocol_weights,
        EVIDENCE_CONFIG,
    )
    ped_test_raw, ped_nn_idx, _, ped_arrays = compute_ped_external(
        train_df,
        test_df,
        main_weights,
        protocol_weights,
        EVIDENCE_CONFIG,
    )
    ped_ref_scaled = transform_ped_for_threshold(ped_ref_raw, EVIDENCE_CONFIG)
    ped_test_scaled = transform_ped_for_threshold(ped_test_raw, EVIDENCE_CONFIG)

    local_stats = local_evidence_statistics(train_df, ped_nn_idx, target_col)

    # -------------------------
    # Training-reference table
    # -------------------------
    ref_id_cols = [
        c for c in ("sample_id", "source_key", "Ref_source_num", "Year", "Temperature", "E_Time")
        if c in train_df.columns
    ]
    ref = train_df[ref_id_cols].copy().reset_index(drop=True)
    ref.insert(0, "fold", fold)
    ref.insert(1, "training_ref_idx", np.arange(len(ref), dtype=int))
    ref = pd.concat([ref, med_ref.reset_index(drop=True)], axis=1)
    ref["PED_raw"] = ped_ref_raw
    ref["PED_log1p"] = np.log1p(ped_ref_raw)
    ref["PED_for_threshold"] = ped_ref_scaled

    # -------------------------
    # Held-out OOF table
    # -------------------------
    oof = test_df[preferred_identity_columns(test_df)].copy().reset_index(drop=True)
    oof.insert(0, "fold", fold)
    oof.insert(1, "fold_test_idx", np.arange(len(oof), dtype=int))
    oof["y_true"] = y_test
    oof["y_pred"] = y_pred
    oof["error"] = y_pred - y_test
    oof["abs_error"] = np.abs(oof["error"])
    oof = pd.concat([oof, med_test.reset_index(drop=True)], axis=1)
    oof["PED_raw"] = ped_test_raw
    oof["PED_log1p"] = np.log1p(ped_test_raw)
    oof["PED_for_threshold"] = ped_test_scaled
    oof = pd.concat([oof, local_stats.reset_index(drop=True)], axis=1)

    # -------------------------
    # Fold-specific training-derived thresholds
    # -------------------------
    thresholds = derive_training_thresholds(
        ref["MED"].to_numpy(dtype=float),
        ref["PED_for_threshold"].to_numpy(dtype=float),
        SENSITIVITY_PERCENTILES,
    )
    thresholds.insert(0, "fold", fold)

    # Save per-fold artifacts.
    oof.to_csv(
        OUTPUT_DIR / f"fold{fold}_oof_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ref.to_csv(
        OUTPUT_DIR / f"fold{fold}_training_reference_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    diagnostics = {
        "fold": fold,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_train_sources": int(train_df[source_col].nunique()) if source_col else None,
        "n_test_sources": int(test_df[source_col].nunique()) if source_col else None,
        "med_normalization": med_norm,
        "physical_standardization": physical_standardization_to_dict(ped_arrays),
    }
    return oof, thresholds, metrics, diagnostics


# =============================================================================
# 4. Pooled analysis
# =============================================================================

def main() -> None:
    EVIDENCE_CONFIG.validate()
    main_weights, protocol_weights, weight_cfg = load_fixed_evidence_weights(WEIGHT_JSON)

    print(f"Loaded fixed evidence weights: {WEIGHT_JSON}")
    print(
        f"TOPK={EVIDENCE_CONFIG.topk}, "
        f"LAMBDA_PROTOCOL={EVIDENCE_CONFIG.lambda_protocol}, "
        f"HT mismatch penalty={EVIDENCE_CONFIG.ht_route_mismatch_penalty}"
    )
    print(f"Protocol features: {list(PROTOCOL_NUM_FEATURES)}")
    print(f"E_Time weight: {protocol_weights['E_Time']:.8f}")

    all_oof = []
    all_thresholds = []
    fold_metric_rows = []
    fold_diagnostics = []

    for fold in FOLDS:
        oof, thresholds, metrics, diagnostics = analyze_fold(
            fold,
            main_weights,
            protocol_weights,
        )
        all_oof.append(oof)
        all_thresholds.append(thresholds)
        fold_metric_rows.append({"fold": fold, "n_test": int(len(oof)), **metrics})
        fold_diagnostics.append(diagnostics)

    pooled = pd.concat(all_oof, ignore_index=True)
    fold_thresholds = pd.concat(all_thresholds, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    # -------------------------------------------------------------------------
    # Pooled sanity checks
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 90}\nPooled OOF sanity checks\n{'=' * 90}")
    print(f"Total OOF rows: {len(pooled)}")
    print("Fold test sizes:", pooled.groupby("fold").size().to_dict())

    if "sample_id" in pooled.columns:
        n_unique = int(pooled["sample_id"].nunique())
        print(f"Unique sample_id: {n_unique} / {len(pooled)}")
        if n_unique != len(pooled):
            duplicated = pooled.loc[pooled["sample_id"].duplicated(keep=False), "sample_id"]
            raise RuntimeError(
                "Pooled OOF contains duplicated sample_id values: "
                f"{duplicated.astype(str).unique()[:10].tolist()}"
            )

    if "source_key" in pooled.columns:
        source_fold_counts = pooled.groupby("source_key")["fold"].nunique()
        if (source_fold_counts > 1).any():
            bad = source_fold_counts[source_fold_counts > 1]
            raise RuntimeError(
                "GroupKFold violation: some source_key values occur in multiple held-out folds: "
                f"{bad.index.astype(str).tolist()[:10]}"
            )
        print("Held-out source assigned to one fold only: PASS")

    pooled_metrics = calc_regression_metrics(
        pooled["y_true"].to_numpy(dtype=float),
        pooled["y_pred"].to_numpy(dtype=float),
    )
    print(
        f"Pooled OOF: R2={pooled_metrics['R2']:.4f}, "
        f"MAE={pooled_metrics['MAE']:.4f}, RMSE={pooled_metrics['RMSE']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Global training-derived thresholds
    # Median of the five fold-specific percentile thresholds.
    # -------------------------------------------------------------------------
    global_thresholds = (
        fold_thresholds.groupby("percentile", as_index=False)
        .agg(
            MED_threshold=("MED_threshold", "median"),
            PED_threshold=("PED_threshold", "median"),
            MED_threshold_min=("MED_threshold", "min"),
            MED_threshold_max=("MED_threshold", "max"),
            PED_threshold_min=("PED_threshold", "min"),
            PED_threshold_max=("PED_threshold", "max"),
        )
        .sort_values("percentile")
        .reset_index(drop=True)
    )

    print("\nGlobal training-derived thresholds:")
    print(global_thresholds.to_string(index=False))

    # -------------------------------------------------------------------------
    # Q assignment and threshold sensitivity
    # -------------------------------------------------------------------------
    sensitivity_stats_frames = []
    sensitivity_summary_rows = []

    for _, thr in global_thresholds.iterrows():
        p = int(thr["percentile"])
        qcol = f"quadrant_P{p}"
        pooled[qcol] = assign_quadrants(
            pooled["MED"].to_numpy(dtype=float),
            pooled["PED_for_threshold"].to_numpy(dtype=float),
            float(thr["MED_threshold"]),
            float(thr["PED_threshold"]),
        )

        stats = quadrant_error_statistics(pooled, qcol)
        stats.insert(0, "percentile", p)
        stats["MED_threshold"] = float(thr["MED_threshold"])
        stats["PED_threshold"] = float(thr["PED_threshold"])
        sensitivity_stats_frames.append(stats)

        q1 = stats.loc[stats["quadrant"] == "Q1"].iloc[0]
        q4 = stats.loc[stats["quadrant"] == "Q4"].iloc[0]
        sensitivity_summary_rows.append({
            "percentile": p,
            "MED_threshold": float(thr["MED_threshold"]),
            "PED_threshold": float(thr["PED_threshold"]),
            "Q1_count": int(q1["count"]),
            "Q1_fraction": float(q1["fraction"]),
            "Q1_MAE": float(q1["MAE"]),
            "Q1_RMSE": float(q1["RMSE"]),
            "Q4_count": int(q4["count"]),
            "Q4_fraction": float(q4["fraction"]),
            "Q4_MAE": float(q4["MAE"]),
            "Q4_RMSE": float(q4["RMSE"]),
            "Q4_minus_Q1_MAE": float(q4["MAE"] - q1["MAE"]),
            "Q4_over_Q1_MAE": float(q4["MAE"] / q1["MAE"]) if q1["MAE"] > 0 else np.nan,
        })

    pooled["quadrant"] = pooled[f"quadrant_P{MAIN_PERCENTILE}"]
    sensitivity_stats = pd.concat(sensitivity_stats_frames, ignore_index=True)
    sensitivity_summary = pd.DataFrame(sensitivity_summary_rows)

    main_thr = global_thresholds.loc[
        global_thresholds["percentile"] == MAIN_PERCENTILE
    ].iloc[0]
    main_stats = quadrant_error_statistics(pooled, "quadrant")
    main_stats["MED_threshold"] = float(main_thr["MED_threshold"])
    main_stats["PED_threshold"] = float(main_thr["PED_threshold"])

    print(f"\nMain P{MAIN_PERCENTILE} quadrant statistics:")
    print(main_stats.to_string(index=False))

    print("\nP40-P60 Q1/Q4 sensitivity summary:")
    print(sensitivity_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # Save pooled artifacts
    # -------------------------------------------------------------------------
    pooled_path = OUTPUT_DIR / "pooled_source_oof_evidence.csv"
    pooled.to_csv(pooled_path, index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(OUTPUT_DIR / "fold_oof_metrics.csv", index=False, encoding="utf-8-sig")
    fold_thresholds.to_csv(
        OUTPUT_DIR / "fold_training_reference_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    global_thresholds.to_csv(
        OUTPUT_DIR / "global_training_derived_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_stats.to_csv(
        OUTPUT_DIR / f"quadrant_stats_P{MAIN_PERCENTILE}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_stats.to_csv(
        OUTPUT_DIR / "threshold_sensitivity_quadrant_stats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_summary.to_csv(
        OUTPUT_DIR / "threshold_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_json = {
        "n_oof": int(len(pooled)),
        "fold_test_sizes": {
            str(int(k)): int(v) for k, v in pooled.groupby("fold").size().to_dict().items()
        },
        "pooled_metrics": pooled_metrics,
        "mean_fold_metrics": {
            "R2_mean": float(fold_metrics["R2"].mean()),
            "R2_std": float(fold_metrics["R2"].std(ddof=1)),
            "MAE_mean": float(fold_metrics["MAE"].mean()),
            "MAE_std": float(fold_metrics["MAE"].std(ddof=1)),
            "RMSE_mean": float(fold_metrics["RMSE"].mean()),
            "RMSE_std": float(fold_metrics["RMSE"].std(ddof=1)),
        },
    }
    with (OUTPUT_DIR / "pooled_oof_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    run_config = {
        "version": "final_exposure_time_aware",
        "model_dir": str(MODEL_DIR),
        "data_dir": str(DATA_DIR),
        "weight_json": str(WEIGHT_JSON),
        "folds": list(FOLDS),
        "evidence_config": EVIDENCE_CONFIG.to_dict(),
        "sensitivity_percentiles": list(SENSITIVITY_PERCENTILES),
        "main_percentile": int(MAIN_PERCENTILE),
        "main_physical_features": list(MAIN_PHYSICAL_FEATURES),
        "protocol_numerical_features": list(PROTOCOL_NUM_FEATURES),
        "exposure_time_feature": "E_Time",
        "global_threshold_rule": (
            "For each percentile, take the median of the five fold-specific "
            "training-reference thresholds."
        ),
        "weight_config": weight_cfg,
        "fold_diagnostics": fold_diagnostics,
    }
    with (OUTPUT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print(f"\nSaved all outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
