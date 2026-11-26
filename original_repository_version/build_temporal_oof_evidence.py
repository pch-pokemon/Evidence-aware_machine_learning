# -*- coding: utf-8 -*-
"""
build_temporal_oof_evidence.py
==============================
Pooled five-fold temporal evidence-state evolution analysis.

Comparison
----------
Historical state:
    each source-aware fold model trained only on records with Year <= YEAR_CUTOFF.

Current state:
    the already-frozen pooled source-aware OOF results produced by
    build_pooled_oof_evidence.py.

Key temporal rules
------------------
1) The held-out sample in each fold is identical between historical and current states.
2) The fixed evidence weights, TOPK, lambda_protocol, route penalty, and P50 boundaries
   are unchanged.
3) The historical MED is calibrated only from the historical training evidence pool.
4) For PED, the HISTORICAL evidence pool is used for neighbors, but numerical
   standardization is fixed to the corresponding FULL current fold-training pool.
   This keeps the physical distance scale directly comparable between historical
   and current states while ensuring that future records do not act as historical neighbors.
5) The same global current P50 MED/PED thresholds are used for both states.

Prerequisites
-------------
1) build_pooled_oof_evidence.py
2) train_xgb_temporal_source_kfold.py

Outputs
-------
models/xgb_source_kfold/temporal_evidence_2022_vs_current/
    historical_2022_pooled_oof.csv
    temporal_comparison.csv
    transition_matrix_counts.csv
    transition_matrix_row_fraction.csv
    transition_summary.csv
    fold_temporal_metrics.csv
    q4_to_q1_candidates.csv
    into_q1_candidates.csv
    into_q4_candidates.csv
    run_config.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
    ensure_2d_leaf_matrix,
    identify_source_column,
    load_fixed_evidence_weights,
    local_evidence_statistics,
    normalize_weight_mapping,
    physical_distance_components_for_query,
    safe_std,
    topk_indices,
    transform_ped_for_threshold,
)


# =============================================================================
# Configuration
# =============================================================================

YEAR_CUTOFF = 2022
FOLDS: Sequence[int] = (1, 2, 3, 4, 5)

DATA_DIR = Path("splits/source_kfold")
CURRENT_MODEL_DIR = Path("models/xgb_source_kfold")
HIST_MODEL_DIR = Path(f"models/xgb_source_kfold_temporal_{YEAR_CUTOFF}")

POOLED_DIR = CURRENT_MODEL_DIR / "pooled_oof_evidence"
CURRENT_POOLED_PATH = POOLED_DIR / "pooled_source_oof_evidence.csv"
THRESHOLD_PATH = POOLED_DIR / "global_training_derived_thresholds.csv"
WEIGHT_JSON = Path("models/fixed_evidence_weights/final_fixed_evidence_weights.json")

OUTPUT_DIR = CURRENT_MODEL_DIR / f"temporal_evidence_{YEAR_CUTOFF}_vs_current"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_PERCENTILE = 50

EVIDENCE_CONFIG = EvidenceMetricConfig(
    topk=10,
    lambda_protocol=0.15,
    ht_route_mismatch_penalty=1.0,
    ped_threshold_scale="log1p",
    clip_training_reference_normalization=True,
)


# =============================================================================
# Fixed-scale PED helpers
# =============================================================================

def build_physical_arrays_on_reference_scale(
    evidence_df: pd.DataFrame,
    query_df: pd.DataFrame,
    scale_reference_df: pd.DataFrame,
    main_weights: Mapping[str, float],
    protocol_weights: Mapping[str, float],
) -> dict:
    """
    Build evidence/query physical arrays using mean/std fitted from scale_reference_df.

    For temporal analysis:
      evidence_df        = historical Year<=cutoff training pool
      query_df           = fixed held-out fold
      scale_reference_df = full current fold training pool

    Therefore new literature records set the common measurement scale but do not enter
    the historical neighbor pool.
    """
    required = list(MAIN_PHYSICAL_FEATURES) + list(PROTOCOL_NUM_FEATURES) + list(HT_FLAG_FEATURES)
    for name, df in (
        ("evidence_df", evidence_df),
        ("query_df", query_df),
        ("scale_reference_df", scale_reference_df),
    ):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing required physical features: {missing}")

    main_ref = scale_reference_df.loc[:, MAIN_PHYSICAL_FEATURES].astype(float)
    main_mu = main_ref.mean()
    main_sigma = safe_std(main_ref)
    w_main = normalize_weight_mapping(
        main_weights, MAIN_PHYSICAL_FEATURES, name="main_weights"
    )
    sw_main = np.sqrt(w_main.to_numpy(dtype=float))

    z_main_train = (
        (evidence_df.loc[:, MAIN_PHYSICAL_FEATURES].astype(float) - main_mu) / main_sigma
    ).to_numpy() * sw_main
    z_main_query = (
        (query_df.loc[:, MAIN_PHYSICAL_FEATURES].astype(float) - main_mu) / main_sigma
    ).to_numpy() * sw_main

    protocol_ref = scale_reference_df.loc[:, PROTOCOL_NUM_FEATURES].astype(float)
    protocol_mu = protocol_ref.mean()
    protocol_sigma = safe_std(protocol_ref)
    w_protocol = normalize_weight_mapping(
        protocol_weights, PROTOCOL_NUM_FEATURES, name="protocol_weights"
    )
    sw_protocol = np.sqrt(w_protocol.to_numpy(dtype=float))

    z_protocol_train = (
        (evidence_df.loc[:, PROTOCOL_NUM_FEATURES].astype(float) - protocol_mu)
        / protocol_sigma
    ).to_numpy() * sw_protocol
    z_protocol_query = (
        (query_df.loc[:, PROTOCOL_NUM_FEATURES].astype(float) - protocol_mu)
        / protocol_sigma
    ).to_numpy() * sw_protocol

    return {
        "z_main_train": z_main_train,
        "z_main_query": z_main_query,
        "z_protocol_train": z_protocol_train,
        "z_protocol_query": z_protocol_query,
        "flags_train": evidence_df.loc[:, HT_FLAG_FEATURES].fillna(0).astype(int).to_numpy(),
        "flags_query": query_df.loc[:, HT_FLAG_FEATURES].fillna(0).astype(int).to_numpy(),
        "main_mu": main_mu,
        "main_sigma": main_sigma,
        "protocol_mu": protocol_mu,
        "protocol_sigma": protocol_sigma,
        "w_main": w_main,
        "w_protocol": w_protocol,
    }


def compute_ped_external_on_reference_scale(
    evidence_df: pd.DataFrame,
    query_df: pd.DataFrame,
    scale_reference_df: pd.DataFrame,
    main_weights: Mapping[str, float],
    protocol_weights: Mapping[str, float],
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    arrays = build_physical_arrays_on_reference_scale(
        evidence_df,
        query_df,
        scale_reference_df,
        main_weights,
        protocol_weights,
    )

    ped = np.empty(len(query_df), dtype=float)
    nn_idx: List[np.ndarray] = []
    nn_dist: List[np.ndarray] = []

    for i in range(len(query_df)):
        d, _, _ = physical_distance_components_for_query(arrays, i, EVIDENCE_CONFIG)
        idx = topk_indices(d, EVIDENCE_CONFIG.topk)
        selected = d[idx]
        ped[i] = float(np.mean(selected))
        nn_idx.append(idx)
        nn_dist.append(selected)

    return ped, nn_idx, nn_dist


# =============================================================================
# I/O and matching
# =============================================================================

def load_thresholds() -> Tuple[float, float]:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing {THRESHOLD_PATH}. Run build_pooled_oof_evidence.py first."
        )
    df = pd.read_csv(THRESHOLD_PATH)
    row = df.loc[df["percentile"] == MAIN_PERCENTILE]
    if len(row) != 1:
        raise ValueError(f"Could not resolve a unique P{MAIN_PERCENTILE} threshold row")
    row = row.iloc[0]
    return float(row["MED_threshold"]), float(row["PED_threshold"])


def preferred_identity_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        "sample_id", "source_key", "Ref_source_num", "Year",
        "Temperature", "E_Time",
        "x_Co", "x_Ni", "x_Cr", "x_Al", "x_Ti", "x_W",
    ]
    return [c for c in cols if c in df.columns]


# =============================================================================
# Per-fold historical analysis
# =============================================================================

def analyze_historical_fold(
    fold: int,
    main_weights: Dict[str, float],
    protocol_weights: Dict[str, float],
    med_threshold: float,
    ped_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, float], dict]:

    print(f"\n{'=' * 90}\nTemporal evidence | Fold {fold}\n{'=' * 90}")

    full_train_path = DATA_DIR / f"fold{fold}_train.csv"
    test_path = DATA_DIR / f"fold{fold}_test.csv"
    hist_train_path = HIST_MODEL_DIR / f"fold{fold}_train_{YEAR_CUTOFF}.csv"
    hist_model_path = HIST_MODEL_DIR / f"xgb_fold{fold}.joblib"
    current_meta_path = CURRENT_MODEL_DIR / f"xgb_fold{fold}_meta.json"

    required = [
        full_train_path, test_path, hist_train_path,
        hist_model_path, current_meta_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    full_train = pd.read_csv(full_train_path)
    test_df = pd.read_csv(test_path)
    hist_train = pd.read_csv(hist_train_path)
    model = joblib.load(hist_model_path)

    with current_meta_path.open("r", encoding="utf-8") as f:
        current_meta = json.load(f)
    feature_cols = list(current_meta["feature_cols"])
    target_col = str(current_meta["target_col"])

    source_col = identify_source_column(full_train)
    if source_col and source_col in test_df.columns:
        overlap = set(hist_train[source_col].dropna()) & set(test_df[source_col].dropna())
        if overlap:
            raise RuntimeError(f"Fold {fold}: historical source leakage detected")
        print("Source overlap check: PASS")

    X_hist = hist_train[feature_cols].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(X_test), dtype=float)
    metrics = calc_regression_metrics(y_test, y_pred)

    # MED: calibrated only on contemporaneous historical evidence.
    hist_leaf = ensure_2d_leaf_matrix(model.apply(X_hist))
    test_leaf = ensure_2d_leaf_matrix(model.apply(X_test))
    med_ref, med_test, _, _, med_norm = build_med_from_training_reference(
        hist_leaf, test_leaf, EVIDENCE_CONFIG
    )

    # PED: historical neighbors, current full-fold physical standardization.
    ped_raw, ped_nn_idx, _ = compute_ped_external_on_reference_scale(
        hist_train,
        test_df,
        full_train,
        main_weights,
        protocol_weights,
    )
    ped_scaled = transform_ped_for_threshold(ped_raw, EVIDENCE_CONFIG)
    state = assign_quadrants(
        med_test["MED"].to_numpy(dtype=float),
        ped_scaled,
        med_threshold,
        ped_threshold,
    )

    local_stats = local_evidence_statistics(hist_train, ped_nn_idx, target_col)

    out = test_df[preferred_identity_columns(test_df)].copy().reset_index(drop=True)
    out.insert(0, "fold", fold)
    out.insert(1, "fold_test_idx", np.arange(len(out), dtype=int))
    out["y_true"] = y_test
    out["y_pred_2022"] = y_pred
    out["error_2022"] = y_pred - y_test
    out["abs_error_2022"] = np.abs(out["error_2022"])
    out["D_LEAF_2022"] = med_test["D_LEAF"].to_numpy(dtype=float)
    out["D_SUPPORT_2022"] = med_test["D_SUPPORT"].to_numpy(dtype=float)
    out["MED_2022"] = med_test["MED"].to_numpy(dtype=float)
    out["PED_raw_2022"] = ped_raw
    out["PED_log1p_2022"] = np.log1p(ped_raw)
    out["PED_for_threshold_2022"] = ped_scaled
    out["quadrant_2022"] = state

    # Prefix historical local evidence statistics.
    local_stats = local_stats.add_suffix("_2022")
    out = pd.concat([out, local_stats.reset_index(drop=True)], axis=1)

    diagnostics = {
        "fold": int(fold),
        "n_train_current_full": int(len(full_train)),
        "n_train_historical": int(len(hist_train)),
        "n_test": int(len(test_df)),
        "historical_model_metrics": metrics,
        "historical_med_normalization": med_norm,
        "historical_PED_scale_rule": (
            "Neighbors from Year<=cutoff training pool; mean/std from the full current "
            "training fold to preserve a common physical scale."
        ),
    }

    print(
        f"Historical pooled component | R2={metrics['R2']:.4f}, "
        f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}"
    )

    return out, metrics, diagnostics


# =============================================================================
# Transition summaries
# =============================================================================

def build_transition_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    order = ["Q1", "Q2", "Q3", "Q4"]
    counts = pd.crosstab(
        pd.Categorical(df["quadrant_2022"], categories=order, ordered=True),
        pd.Categorical(df["quadrant_current"], categories=order, ordered=True),
        dropna=False,
    )
    counts.index.name = "state_2022"
    counts.columns.name = "state_current"

    row_frac = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    return counts, row_frac


def add_transition_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    old = out["quadrant_2022"]
    cur = out["quadrant_current"]

    out["state_changed"] = old != cur
    out["transition"] = old.astype(str) + "→" + cur.astype(str)
    out["transition_class"] = np.select(
        [
            old == cur,
            (old != "Q1") & (cur == "Q1"),
            (old == "Q1") & (cur != "Q1"),
            (old != "Q4") & (cur == "Q4"),
            (old == "Q4") & (cur != "Q4"),
        ],
        [
            "stable",
            "into_Q1",
            "out_of_Q1",
            "into_Q4",
            "out_of_Q4",
        ],
        default="other_transition",
    )

    out["delta_MED_current_minus_2022"] = out["MED_current"] - out["MED_2022"]
    out["delta_PED_log1p_current_minus_2022"] = (
        out["PED_log1p_current"] - out["PED_log1p_2022"]
    )
    out["delta_abs_error_current_minus_2022"] = (
        out["abs_error_current"] - out["abs_error_2022"]
    )
    out["delta_n_sources_current_minus_2022"] = (
        out["n_sources_current"] - out["n_sources_2022"]
    )
    out["delta_lgkp_std_current_minus_2022"] = (
        out["lgkp_std_current"] - out["lgkp_std_2022"]
    )

    # Positive evidence_gain_score = movement toward closer / more diverse / more coherent evidence.
    out["evidence_gain_score"] = (
        -out["delta_MED_current_minus_2022"]
        -out["delta_PED_log1p_current_minus_2022"]
        +0.05 * out["delta_n_sources_current_minus_2022"].fillna(0)
        -0.10 * out["delta_lgkp_std_current_minus_2022"].fillna(0)
    )
    return out


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    EVIDENCE_CONFIG.validate()

    required = [
        CURRENT_POOLED_PATH,
        THRESHOLD_PATH,
        WEIGHT_JSON,
        HIST_MODEL_DIR,
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))

    current = pd.read_csv(CURRENT_POOLED_PATH)
    main_weights, protocol_weights, weight_cfg = load_fixed_evidence_weights(WEIGHT_JSON)
    med_threshold, ped_threshold = load_thresholds()

    print(f"Using fixed current P{MAIN_PERCENTILE} thresholds:")
    print(f"  MED = {med_threshold:.6f}")
    print(f"  PED = {ped_threshold:.6f} ({EVIDENCE_CONFIG.ped_threshold_scale})")

    historical_frames = []
    fold_metrics = []
    fold_diagnostics = []

    for fold in FOLDS:
        hist, metrics, diagnostics = analyze_historical_fold(
            fold,
            main_weights,
            protocol_weights,
            med_threshold,
            ped_threshold,
        )
        historical_frames.append(hist)
        fold_metrics.append({"fold": fold, "n_test": len(hist), **metrics})
        fold_diagnostics.append(diagnostics)

    historical = pd.concat(historical_frames, ignore_index=True)

    # Current table: keep only columns needed for temporal matching.
    current_cols = [
        "fold", "sample_id", "y_pred", "error", "abs_error",
        "MED", "PED_raw", "PED_log1p", "PED_for_threshold", "quadrant",
        "n_sources", "lgkp_std", "neighbor_lgkp_mean",
        "neighbor_lgkp_min", "neighbor_lgkp_max",
        "neighbor_year_min", "neighbor_year_max",
    ]
    current_cols = [c for c in current_cols if c in current.columns]
    cur = current[current_cols].copy()
    cur = cur.rename(columns={
        "y_pred": "y_pred_current",
        "error": "error_current",
        "abs_error": "abs_error_current",
        "MED": "MED_current",
        "PED_raw": "PED_raw_current",
        "PED_log1p": "PED_log1p_current",
        "PED_for_threshold": "PED_for_threshold_current",
        "quadrant": "quadrant_current",
        "n_sources": "n_sources_current",
        "lgkp_std": "lgkp_std_current",
        "neighbor_lgkp_mean": "neighbor_lgkp_mean_current",
        "neighbor_lgkp_min": "neighbor_lgkp_min_current",
        "neighbor_lgkp_max": "neighbor_lgkp_max_current",
        "neighbor_year_min": "neighbor_year_min_current",
        "neighbor_year_max": "neighbor_year_max_current",
    })

    keys = ["fold", "sample_id"]
    if "sample_id" not in historical.columns or "sample_id" not in cur.columns:
        raise ValueError("Temporal pooled analysis requires unique sample_id")

    comparison = historical.merge(cur, on=keys, how="inner", validate="one_to_one")
    if len(comparison) != len(current):
        raise RuntimeError(
            f"Temporal merge produced {len(comparison)} rows but current OOF has {len(current)} rows"
        )

    comparison = add_transition_labels(comparison)

    # Pooled historical metrics.
    hist_metrics = calc_regression_metrics(
        comparison["y_true"].to_numpy(dtype=float),
        comparison["y_pred_2022"].to_numpy(dtype=float),
    )
    cur_metrics = calc_regression_metrics(
        comparison["y_true"].to_numpy(dtype=float),
        comparison["y_pred_current"].to_numpy(dtype=float),
    )

    counts, row_frac = build_transition_matrix(comparison)

    transition_summary = (
        comparison.groupby("transition_class", as_index=False)
        .agg(
            count=("sample_id", "size"),
            mean_delta_MED=("delta_MED_current_minus_2022", "mean"),
            mean_delta_PED_log1p=("delta_PED_log1p_current_minus_2022", "mean"),
            mean_delta_abs_error=("delta_abs_error_current_minus_2022", "mean"),
            mean_delta_n_sources=("delta_n_sources_current_minus_2022", "mean"),
            mean_delta_lgkp_std=("delta_lgkp_std_current_minus_2022", "mean"),
        )
    )
    transition_summary["fraction"] = transition_summary["count"] / len(comparison)

    # Candidate tables for later Fig. 4 case selection.
    q4_to_q1 = comparison[
        (comparison["quadrant_2022"] == "Q4")
        & (comparison["quadrant_current"] == "Q1")
    ].sort_values("evidence_gain_score", ascending=False)

    into_q1 = comparison[
        (comparison["quadrant_2022"] != "Q1")
        & (comparison["quadrant_current"] == "Q1")
    ].sort_values("evidence_gain_score", ascending=False)

    into_q4 = comparison[
        (comparison["quadrant_2022"] != "Q4")
        & (comparison["quadrant_current"] == "Q4")
    ].sort_values("evidence_gain_score", ascending=True)

    # Save.
    historical.to_csv(
        OUTPUT_DIR / f"historical_{YEAR_CUTOFF}_pooled_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        OUTPUT_DIR / "temporal_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    counts.to_csv(
        OUTPUT_DIR / "transition_matrix_counts.csv",
        encoding="utf-8-sig",
    )
    row_frac.to_csv(
        OUTPUT_DIR / "transition_matrix_row_fraction.csv",
        encoding="utf-8-sig",
    )
    transition_summary.to_csv(
        OUTPUT_DIR / "transition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_metrics).to_csv(
        OUTPUT_DIR / "fold_temporal_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    q4_to_q1.to_csv(
        OUTPUT_DIR / "q4_to_q1_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    into_q1.to_csv(
        OUTPUT_DIR / "into_q1_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    into_q4.to_csv(
        OUTPUT_DIR / "into_q4_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "year_cutoff": YEAR_CUTOFF,
        "main_percentile": MAIN_PERCENTILE,
        "fixed_thresholds": {
            "MED_threshold": med_threshold,
            "PED_threshold": ped_threshold,
            "PED_scale": EVIDENCE_CONFIG.ped_threshold_scale,
        },
        "historical_pooled_metrics": hist_metrics,
        "current_pooled_metrics": cur_metrics,
        "evidence_config": EVIDENCE_CONFIG.to_dict(),
        "weight_json": str(WEIGHT_JSON),
        "weight_config": weight_cfg,
        "temporal_rules": {
            "held_out_samples": "identical source-aware fold test sets",
            "historical_model": "retrained on fold training records with Year <= cutoff",
            "hyperparameters": "same globally optimized parameters as current source-aware XGBoost",
            "MED_calibration": "historical training leave-one-out reference only",
            "PED_neighbors": "historical training pool only",
            "PED_standardization": "fixed to corresponding full current fold-training pool",
            "Q_boundaries": "same global current P50 thresholds for historical and current states",
        },
        "fold_diagnostics": fold_diagnostics,
    }
    with (OUTPUT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("Temporal pooled summary")
    print("=" * 90)
    print(
        f"Historical {YEAR_CUTOFF}: "
        f"R2={hist_metrics['R2']:.4f}, MAE={hist_metrics['MAE']:.4f}, RMSE={hist_metrics['RMSE']:.4f}"
    )
    print(
        f"Current: "
        f"R2={cur_metrics['R2']:.4f}, MAE={cur_metrics['MAE']:.4f}, RMSE={cur_metrics['RMSE']:.4f}"
    )
    print("\nTransition counts:")
    print(counts.to_string())
    print("\nTransition summary:")
    print(transition_summary.to_string(index=False))
    print(f"\nQ4→Q1 candidate cases: {len(q4_to_q1)}")
    print(f"All →Q1 candidate cases: {len(into_q1)}")
    print(f"All →Q4 candidate cases: {len(into_q4)}")
    print(f"\nSaved all outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
