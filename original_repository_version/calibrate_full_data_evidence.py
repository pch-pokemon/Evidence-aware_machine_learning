# -*- coding: utf-8 -*-
"""
calibrate_full_data_evidence.py
===============================
Calibrate the final full-data deployment model's MED/PED evidence reference.

Purpose
-------
The retrospective manuscript evaluation uses pooled source-aware OOF predictions
and cross-fold training-derived thresholds. Prospective screening, however, uses
a single XGBoost model retrained on all available records. Because MED is tied
to a particular tree ensemble and evidence pool, the deployment model receives
its own leave-one-out training-reference calibration.

This script:
1) builds full-data leave-one-out MED and PED reference distributions;
2) derives P40/P45/P50/P55/P60 deployment thresholds;
3) assigns descriptive Q1-Q4 states to the leave-one-out reference records;
4) exports MED normalization and PED physical-standardization parameters;
5) compares deployment thresholds with the retrospective pooled-OOF thresholds.

No model fitting occurs here.

Prerequisites
-------------
- python train_xgb_full_data.py
- models/fixed_evidence_weights/final_fixed_evidence_weights.json
- evidence_core.py

Outputs
-------
models/xgb_full_data/evidence_calibration/
    full_data_training_reference_evidence.csv
    deployment_thresholds.csv
    state_distribution_P50.csv
    threshold_comparison_with_pooled_oof.csv
    deployment_evidence_config.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
import pandas as pd

from evidence_core import (
    EvidenceMetricConfig,
    assign_quadrants,
    build_med_from_training_reference,
    build_physical_arrays,
    compute_ped_training_loo,
    derive_training_thresholds,
    ensure_2d_leaf_matrix,
    load_fixed_evidence_weights,
    physical_standardization_to_dict,
    transform_ped_for_threshold,
)


# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = Path("models/xgb_full_data")
MODEL_PATH = MODEL_DIR / "xgb_full_data.joblib"
META_PATH = MODEL_DIR / "xgb_full_data_meta.json"
FULL_DATA_PATH = MODEL_DIR / "full_dataset.csv"

WEIGHT_JSON = Path("models/fixed_evidence_weights/final_fixed_evidence_weights.json")

RETROSPECTIVE_THRESHOLD_PATH = (
    Path("models/xgb_source_kfold")
    / "pooled_oof_evidence"
    / "global_training_derived_thresholds.csv"
)

OUTPUT_DIR = MODEL_DIR / "evidence_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSITIVITY_PERCENTILES: Sequence[int] = (40, 45, 50, 55, 60)
MAIN_PERCENTILE = 50

EVIDENCE_CONFIG = EvidenceMetricConfig(
    topk=10,
    lambda_protocol=0.15,
    ht_route_mismatch_penalty=1.0,
    ped_threshold_scale="log1p",
    clip_training_reference_normalization=True,
)


# =============================================================================
# Utilities
# =============================================================================

def load_inputs():
    required = [MODEL_PATH, META_PATH, FULL_DATA_PATH, WEIGHT_JSON]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisites:\n" + "\n".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    full_df = pd.read_csv(FULL_DATA_PATH)
    main_weights, protocol_weights, weight_cfg = load_fixed_evidence_weights(
        WEIGHT_JSON
    )
    return model, meta, full_df, main_weights, protocol_weights, weight_cfg


def preferred_identity_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "source_oof_fold",
        "sample_id",
        "source_key",
        "Ref_source_num",
        "Year",
        "Temperature",
        "E_Time",
        "x_Co",
        "x_Ni",
        "x_Cr",
        "x_Al",
        "x_Ti",
        "x_W",
    ]
    return [c for c in preferred if c in df.columns]


def compare_with_retrospective(deployment: pd.DataFrame) -> pd.DataFrame:
    if not RETROSPECTIVE_THRESHOLD_PATH.exists():
        return pd.DataFrame()

    retrospective = pd.read_csv(RETROSPECTIVE_THRESHOLD_PATH)
    needed = {"percentile", "MED_threshold", "PED_threshold"}
    if not needed.issubset(retrospective.columns):
        return pd.DataFrame()

    left = deployment.rename(columns={
        "MED_threshold": "deployment_MED_threshold",
        "PED_threshold": "deployment_PED_threshold",
    })
    right = retrospective[
        ["percentile", "MED_threshold", "PED_threshold"]
    ].rename(columns={
        "MED_threshold": "pooled_oof_MED_threshold",
        "PED_threshold": "pooled_oof_PED_threshold",
    })

    out = left.merge(right, on="percentile", how="left", validate="one_to_one")
    out["delta_MED_deployment_minus_pooled"] = (
        out["deployment_MED_threshold"] - out["pooled_oof_MED_threshold"]
    )
    out["delta_PED_deployment_minus_pooled"] = (
        out["deployment_PED_threshold"] - out["pooled_oof_PED_threshold"]
    )
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    EVIDENCE_CONFIG.validate()

    (
        model,
        meta,
        full_df,
        main_weights,
        protocol_weights,
        weight_cfg,
    ) = load_inputs()

    feature_cols = list(meta["feature_cols"])
    target_col = str(meta["target_col"])

    required = feature_cols + [target_col]
    missing = [c for c in required if c not in full_df.columns]
    if missing:
        raise ValueError(f"Full evidence pool missing model columns: {missing}")

    if "sample_id" not in full_df.columns:
        raise ValueError("Full-data calibration requires sample_id")
    if int(full_df["sample_id"].nunique(dropna=False)) != len(full_df):
        raise RuntimeError("sample_id is not unique in the full evidence pool")

    print("=" * 90)
    print("Final full-data evidence calibration")
    print("=" * 90)
    print(f"Records: {len(full_df)}")
    print(
        f"TOPK={EVIDENCE_CONFIG.topk}, "
        f"LAMBDA_PROTOCOL={EVIDENCE_CONFIG.lambda_protocol}, "
        f"HT mismatch penalty={EVIDENCE_CONFIG.ht_route_mismatch_penalty}"
    )

    X = full_df[feature_cols].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # MED: full-data leave-one-out training reference
    # -------------------------------------------------------------------------
    full_leaf = ensure_2d_leaf_matrix(model.apply(X))

    # The function always constructs the training LOO reference correctly.
    # The query output is intentionally ignored here.
    med_ref, _, _, _, med_norm = build_med_from_training_reference(
        full_leaf,
        full_leaf,
        EVIDENCE_CONFIG,
    )

    # -------------------------------------------------------------------------
    # PED: full-data leave-one-out physical reference
    # -------------------------------------------------------------------------
    ped_ref_raw = compute_ped_training_loo(
        full_df,
        main_weights,
        protocol_weights,
        EVIDENCE_CONFIG,
    )
    ped_ref_scaled = transform_ped_for_threshold(
        ped_ref_raw,
        EVIDENCE_CONFIG,
    )

    # Build once only to export the exact deployment mean/std and normalized weights.
    physical_arrays = build_physical_arrays(
        full_df,
        full_df,
        main_weights,
        protocol_weights,
    )
    physical_standardization = physical_standardization_to_dict(physical_arrays)

    # -------------------------------------------------------------------------
    # Deployment thresholds
    # -------------------------------------------------------------------------
    thresholds = derive_training_thresholds(
        med_ref["MED"].to_numpy(dtype=float),
        ped_ref_scaled,
        SENSITIVITY_PERCENTILES,
    )

    print("\nDeployment thresholds derived from the 613-record LOO reference:")
    print(thresholds.to_string(index=False))

    main_row = thresholds.loc[
        thresholds["percentile"] == MAIN_PERCENTILE
    ]
    if len(main_row) != 1:
        raise RuntimeError(f"Could not resolve P{MAIN_PERCENTILE} deployment thresholds")
    main_row = main_row.iloc[0]

    med_thr = float(main_row["MED_threshold"])
    ped_thr = float(main_row["PED_threshold"])

    # -------------------------------------------------------------------------
    # Full-data reference table
    # -------------------------------------------------------------------------
    ref = full_df[preferred_identity_columns(full_df)].copy().reset_index(drop=True)
    ref.insert(0, "training_ref_idx", np.arange(len(ref), dtype=int))
    ref = pd.concat([ref, med_ref.reset_index(drop=True)], axis=1)
    ref["PED_raw"] = ped_ref_raw
    ref["PED_log1p"] = np.log1p(ped_ref_raw)
    ref["PED_for_threshold"] = ped_ref_scaled
    ref["quadrant_P50"] = assign_quadrants(
        ref["MED"].to_numpy(dtype=float),
        ref["PED_for_threshold"].to_numpy(dtype=float),
        med_thr,
        ped_thr,
    )

    state_counts = (
        ref["quadrant_P50"]
        .value_counts()
        .reindex(["Q1", "Q2", "Q3", "Q4"], fill_value=0)
        .rename_axis("quadrant")
        .reset_index(name="count")
    )
    state_counts["fraction"] = state_counts["count"] / len(ref)

    print(f"\nMain deployment P50 thresholds:")
    print(f"  MED = {med_thr:.6f}")
    print(f"  PED = {ped_thr:.6f} ({EVIDENCE_CONFIG.ped_threshold_scale})")
    print("\nLOO reference state distribution:")
    print(state_counts.to_string(index=False))

    # -------------------------------------------------------------------------
    # Compare with retrospective cross-fold thresholds.
    # This is diagnostic only; prospective screening should use deployment
    # thresholds because it uses the full-data deployment model.
    # -------------------------------------------------------------------------
    comparison = compare_with_retrospective(thresholds)
    if not comparison.empty:
        print("\nDeployment vs retrospective pooled-OOF thresholds:")
        print(comparison.to_string(index=False))

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    ref.to_csv(
        OUTPUT_DIR / "full_data_training_reference_evidence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    thresholds.to_csv(
        OUTPUT_DIR / "deployment_thresholds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    state_counts.to_csv(
        OUTPUT_DIR / "state_distribution_P50.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not comparison.empty:
        comparison.to_csv(
            OUTPUT_DIR / "threshold_comparison_with_pooled_oof.csv",
            index=False,
            encoding="utf-8-sig",
        )

    config_payload = {
        "purpose": (
            "deployment evidence calibration for the final XGBoost model trained "
            "on all available literature records"
        ),
        "n_reference_records": int(len(full_df)),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "evidence_config": EVIDENCE_CONFIG.to_dict(),
        "fixed_weight_json": str(WEIGHT_JSON),
        "fixed_weight_config": weight_cfg,
        "med_normalization": med_norm,
        "physical_standardization": physical_standardization,
        "threshold_percentiles": list(SENSITIVITY_PERCENTILES),
        "main_percentile": int(MAIN_PERCENTILE),
        "main_thresholds": {
            "MED_threshold": med_thr,
            "PED_threshold": ped_thr,
            "PED_scale": EVIDENCE_CONFIG.ped_threshold_scale,
        },
        "deployment_threshold_rule": (
            "Prospective candidates use P50 thresholds derived from the complete "
            "613-record leave-one-out reference distribution of the final full-data model."
        ),
        "retrospective_threshold_rule": (
            "Pooled source-aware OOF evaluation retains its separately derived "
            "cross-fold training-reference thresholds; these are not overwritten."
        ),
        "files": {
            "model": str(MODEL_PATH),
            "full_evidence_pool": str(FULL_DATA_PATH),
            "training_reference_evidence": str(
                OUTPUT_DIR / "full_data_training_reference_evidence.csv"
            ),
            "deployment_thresholds": str(
                OUTPUT_DIR / "deployment_thresholds.csv"
            ),
        },
    }

    with (OUTPUT_DIR / "deployment_evidence_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved calibration outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
