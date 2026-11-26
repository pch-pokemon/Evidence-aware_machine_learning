# -*- coding: utf-8 -*-
"""
prospective_evidence_screening.py
=================================
Final prospective evidence-aware screening pipeline.

This script is the deployment-stage counterpart of the retrospective pooled OOF
analysis. It uses:

1) the final XGBoost model trained on ALL available literature records;
2) the complete literature dataset as the evidence pool;
3) the fixed evidence weights from the source-aware five-fold pipeline;
4) the full-data leave-one-out deployment calibration;
5) the canonical MED/PED definitions in evidence_core.py.

The script does NOT use any single random/source-aware fold and does NOT hard-code
legacy MED/PED thresholds.

Default prospective design space
--------------------------------
The default grid reproduces the manuscript-style constrained Ni/Co-based design space:

    Co(balance)-30Ni-(5-15)Al-(5-15)Cr-2W-3Ta-3Ti-1Mo-1Nb  [at.%]

with Al and Cr independently varied in 0.5 at.% increments, giving 21 x 21 = 441
virtual candidates.

Fixed processing / oxidation scenario:
    solution treatment: 1200 degC, 12 h
    no intermediate treatment
    aging: 900 degC, 100 h
    oxidation temperature: 900 degC
    oxidation exposure duration E_Time: 100 h

IMPORTANT:
E_Time is an explicit screening-condition variable in the final model/PED pipeline.
If the intended prospective experiment uses another exposure duration, change
OXIDATION_EXPOSURE_TIME_HOURS below and rerun. This is a scenario definition,
not a fitted/tuned parameter.

Primary prospective question
----------------------------
Identify ALL Q1-Q4 candidate pairs with similar predicted lgkp, then select one
representative pair using a predeclared deterministic rule. This avoids selecting
a visually convenient pair first and only later constructing a justification.

Representative-pair rule
------------------------
1) primary state contrast: Q1 vs Q4;
2) |Delta predicted lgkp| <= MAX_DELTA_PREDICTED_LGKP;
3) both candidates must be at least MIN_NORMALIZED_BOUNDARY_MARGIN away from
   their nearest MED/PED decision boundary;
4) normalized composition distance <= MAX_NORMALIZED_COMPOSITION_DISTANCE;
5) eligible pairs are ranked deterministically by:
       a. smaller |Delta predicted lgkp|;
       b. smaller normalized composition distance;
       c. larger minimum boundary margin;
       d. candidate IDs as deterministic tie-breakers.

Outputs
-------
prospective_screening_results/
    all_screened_candidates.csv
    candidate_state_summary_P50.csv
    candidate_state_sensitivity.csv

    all_prediction_matched_pairs.csv
    representative_pair_pool.csv
    representative_pair.csv
    matched_pair_population_summary.csv
    matched_pair_tolerance_summary.csv
    pair_count_threshold_sensitivity.csv

    representative_candidates.csv
    representative_A_model_neighbors.csv
    representative_A_physical_neighbors.csv
    representative_B_model_neighbors.csv
    representative_B_physical_neighbors.csv

    run_config.json

Interpretation
--------------
This is a prospective/conceptual screening demonstration. The generated candidates
are experimentally testable hypotheses, not experimentally validated alloy designs.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from evidence_core import (
    COMPOSITION_FEATURES,
    HT_FLAG_FEATURES,
    MAIN_PHYSICAL_FEATURES,
    PROTOCOL_NUM_FEATURES,
    EvidenceMetricConfig,
    assign_quadrants,
    build_med_from_training_reference,
    compute_ped_external,
    ensure_2d_leaf_matrix,
    load_fixed_evidence_weights,
    local_evidence_statistics,
    transform_ped_for_threshold,
)


# =============================================================================
# 1. Final deployment artifacts
# =============================================================================

MODEL_DIR = Path("models/xgb_full_data")
MODEL_PATH = MODEL_DIR / "xgb_full_data.joblib"
META_PATH = MODEL_DIR / "xgb_full_data_meta.json"
EVIDENCE_POOL_PATH = MODEL_DIR / "full_dataset.csv"

CALIBRATION_DIR = MODEL_DIR / "evidence_calibration"
DEPLOYMENT_CONFIG_PATH = CALIBRATION_DIR / "deployment_evidence_config.json"
DEPLOYMENT_THRESHOLDS_PATH = CALIBRATION_DIR / "deployment_thresholds.csv"

FIXED_WEIGHT_PATH = Path(
    "models/fixed_evidence_weights/final_fixed_evidence_weights.json"
)

OUTPUT_DIR = Path("prospective_screening_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_PERCENTILE = 50


# =============================================================================
# 2. Prospective screening scenario
# =============================================================================

# Oxidation condition
OXIDATION_TEMPERATURE = 900.0
OXIDATION_EXPOSURE_TIME_HOURS = 100.0

# Fixed heat-treatment route
FIXED_PROCESSING: Dict[str, float] = {
    "solu_temp": 1200.0,
    "solu_time": 12.0,
    "middle_temp": 0.0,
    "middle_time": 0.0,
    "aging_temp": 900.0,
    "aging_time": 100.0,
    "has_solu": 1,
    "has_middle": 0,
    "has_aging": 1,
}

# Base alloy, at.%. x_Co is recalculated as the balance element.
BASE_COMPOSITION: Dict[str, float] = {
    "x_Co": 0.0,
    "x_Al": 9.0,
    "x_W": 7.0,
    "x_Ni": 30.0,
    "x_Cr": 10.0,
    "x_Mo": 0.0,
    "x_Fe": 0.0,
    "x_Nb": 0.0,
    "x_C": 0.0,
    "x_Hf": 0.0,
    "x_Si": 0.0,
    "x_Ta": 2.0,
    "x_Ti": 0.0,
    "x_Y": 0.0,
    "x_V": 0.0,
    "x_B": 0.0,
    "x_Zr": 0.0,
    "x_Mn": 0.0,
    "x_Sc": 0.0,
    "x_La": 0.0,
    "x_Re": 0.0,
}

# min, max, step in at.%.
# (0, 0, 0) means fixed at BASE_COMPOSITION.
ELEMENT_RANGES: Dict[str, Tuple[float, float, float]] = {
    "x_Co": (0.0, 0.0, 0.0),
    "x_Al": (5.0, 15.0, 0.5),
    "x_W": (0.0, 0.0, 0.0),
    "x_Ni": (0.0, 0.0, 0.0),
    "x_Cr": (5.0, 20.0, 0.5),
    "x_Mo": (0.0, 0.0, 0.0),
    "x_Fe": (0.0, 0.0, 0.0),
    "x_Nb": (0.0, 0.0, 0.0),
    "x_C": (0.0, 0.0, 0.0),
    "x_Hf": (0.0, 0.0, 0.0),
    "x_Si": (0.0, 0.0, 0.0),
    "x_Ta": (0.0, 0.0, 0.0),
    "x_Ti": (0.0, 0.0, 0.0),
    "x_Y": (0.0, 0.0, 0.0),
    "x_V": (0.0, 0.0, 0.0),
    "x_B": (0.0, 0.0, 0.0),
    "x_Zr": (0.0, 0.0, 0.0),
    "x_Mn": (0.0, 0.0, 0.0),
    "x_Sc": (0.0, 0.0, 0.0),
    "x_La": (0.0, 0.0, 0.0),
    "x_Re": (0.0, 0.0, 0.0),
}

BALANCE_ELEMENT = "x_Co"
TARGET_COMPOSITION_SUM = 100.0
COMPOSITION_SUM_TOLERANCE = 1e-8
MIN_BALANCE_CONTENT = 0.0
MAX_BALANCE_CONTENT = 100.0

# Optional simple feasibility limits after balancing.
# Example: {"x_Co": (25.0, 55.0)}
FEASIBILITY_LIMITS: Dict[str, Tuple[float, float]] = {}

MAX_CANDIDATES = 250_000


# =============================================================================
# 3. Matched-pair settings
# =============================================================================

# Main manuscript contrast.
STATE_CONTRASTS: Sequence[Tuple[str, str]] = (
    ("Q1", "Q4"),
)

# Similar kinetics criterion.
MAX_DELTA_PREDICTED_LGKP = 0.10

# Report how many matched pairs remain under several stricter tolerances.
PAIR_TOLERANCE_REPORT: Sequence[float] = (0.02, 0.05, 0.10)

# Representative-pair stability constraints.
MIN_NORMALIZED_BOUNDARY_MARGIN = 0.08
MAX_NORMALIZED_COMPOSITION_DISTANCE = 1.00

# Safety against accidental pair explosion if a much larger grid is used later.
MAX_EXPORTED_PAIRS = 2_000_000


# =============================================================================
# 4. General utilities
# =============================================================================

def inclusive_grid(vmin: float, vmax: float, step: float) -> np.ndarray:
    """Numerically stable inclusive grid."""
    if math.isclose(vmin, 0.0) and math.isclose(vmax, 0.0):
        raise ValueError("inclusive_grid must not be called for a fixed element")
    if vmax < vmin:
        raise ValueError(f"Invalid range: max={vmax} < min={vmin}")
    if math.isclose(vmax, vmin):
        return np.array([float(vmin)])
    if step <= 0:
        raise ValueError(f"Grid step must be positive for range ({vmin}, {vmax})")

    n = int(math.floor((vmax - vmin) / step + 1e-10))
    values = vmin + np.arange(n + 1, dtype=float) * step
    if values[-1] < vmax - 1e-9:
        values = np.append(values, vmax)
    values[-1] = min(float(values[-1]), float(vmax))
    return np.unique(np.round(values, 10))


def validate_base_configuration() -> List[str]:
    missing_base = [f for f in COMPOSITION_FEATURES if f not in BASE_COMPOSITION]
    missing_ranges = [f for f in COMPOSITION_FEATURES if f not in ELEMENT_RANGES]
    if missing_base:
        raise ValueError(f"BASE_COMPOSITION missing: {missing_base}")
    if missing_ranges:
        raise ValueError(f"ELEMENT_RANGES missing: {missing_ranges}")

    varied: List[str] = []
    for feature in COMPOSITION_FEATURES:
        vmin, vmax, step = ELEMENT_RANGES[feature]
        if math.isclose(vmin, 0.0) and math.isclose(vmax, 0.0):
            continue
        inclusive_grid(vmin, vmax, step)
        varied.append(feature)

    if not varied:
        raise ValueError("At least one composition feature must be varied")
    if BALANCE_ELEMENT not in COMPOSITION_FEATURES:
        raise ValueError(f"Invalid BALANCE_ELEMENT: {BALANCE_ELEMENT}")
    if BALANCE_ELEMENT in varied:
        raise ValueError("The balance element cannot also be varied")

    return varied


def estimated_candidate_count(varied_features: Sequence[str]) -> int:
    count = 1
    for f in varied_features:
        count *= len(inclusive_grid(*ELEMENT_RANGES[f]))
    return int(count)


def passes_feasibility(row: Mapping[str, float]) -> bool:
    for feature, (lower, upper) in FEASIBILITY_LIMITS.items():
        value = float(row[feature])
        if value < float(lower) or value > float(upper):
            return False
    return True


def normalized_boundary_margin(
    med: np.ndarray,
    ped_scaled: np.ndarray,
    med_threshold: float,
    ped_threshold: float,
) -> np.ndarray:
    med_margin = np.abs(med - med_threshold) / max(abs(med_threshold), 1e-12)
    ped_margin = np.abs(ped_scaled - ped_threshold) / max(abs(ped_threshold), 1e-12)
    return np.minimum(med_margin, ped_margin)


def check_close_dict(
    actual: Mapping[str, float],
    expected: Mapping[str, float],
    keys: Sequence[str],
    *,
    atol: float = 1e-10,
) -> None:
    for key in keys:
        if key not in actual or key not in expected:
            continue
        if not np.isclose(float(actual[key]), float(expected[key]), atol=atol, rtol=0.0):
            raise RuntimeError(
                f"Deployment calibration mismatch for {key}: "
                f"recomputed={actual[key]} saved={expected[key]}"
            )


# =============================================================================
# 5. Load final deployment artifacts
# =============================================================================

def load_deployment_artifacts():
    required = [
        MODEL_PATH,
        META_PATH,
        EVIDENCE_POOL_PATH,
        DEPLOYMENT_CONFIG_PATH,
        DEPLOYMENT_THRESHOLDS_PATH,
        FIXED_WEIGHT_PATH,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing final deployment artifacts:\n" + "\n".join(missing)
        )

    model = joblib.load(MODEL_PATH)

    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    with DEPLOYMENT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        deploy_cfg = json.load(f)

    evidence_pool = pd.read_csv(EVIDENCE_POOL_PATH)
    thresholds = pd.read_csv(DEPLOYMENT_THRESHOLDS_PATH)

    main_weights, protocol_weights, weight_cfg = load_fixed_evidence_weights(
        FIXED_WEIGHT_PATH
    )

    evidence_config = EvidenceMetricConfig(**deploy_cfg["evidence_config"])
    evidence_config.validate()

    return (
        model,
        meta,
        deploy_cfg,
        evidence_pool,
        thresholds,
        main_weights,
        protocol_weights,
        weight_cfg,
        evidence_config,
    )


def validate_deployment_schema(
    meta: dict,
    evidence_pool: pd.DataFrame,
) -> Tuple[List[str], str]:
    feature_cols = list(meta["feature_cols"])
    target_col = str(meta["target_col"])

    canonical_expected = (
        set(COMPOSITION_FEATURES)
        | set(PROTOCOL_NUM_FEATURES)
        | {"Temperature"}
        | set(HT_FLAG_FEATURES)
    )

    if set(feature_cols) != canonical_expected:
        extra = sorted(set(feature_cols) - canonical_expected)
        missing = sorted(canonical_expected - set(feature_cols))
        raise ValueError(
            "Final deployment model features do not match the canonical evidence/model "
            f"schema. Missing={missing}; extra={extra}"
        )

    required = feature_cols + [target_col]
    missing_cols = [c for c in required if c not in evidence_pool.columns]
    if missing_cols:
        raise ValueError(f"Full evidence pool missing columns: {missing_cols}")

    if "sample_id" not in evidence_pool.columns:
        raise ValueError("Full evidence pool must contain sample_id")
    if evidence_pool["sample_id"].nunique(dropna=False) != len(evidence_pool):
        raise RuntimeError("sample_id is not unique in the full evidence pool")

    return feature_cols, target_col


# =============================================================================
# 6. Candidate-space construction
# =============================================================================

def build_candidate_space(feature_cols: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    varied_features = validate_base_configuration()

    estimated = estimated_candidate_count(varied_features)
    print(f"Estimated raw candidate count: {estimated:,}")
    if estimated > MAX_CANDIDATES:
        raise ValueError(
            f"Candidate grid ({estimated:,}) exceeds MAX_CANDIDATES={MAX_CANDIDATES:,}"
        )

    grids = [inclusive_grid(*ELEMENT_RANGES[f]) for f in varied_features]
    records: List[dict] = []

    for values in itertools.product(*grids):
        composition = {
            f: float(BASE_COMPOSITION[f]) for f in COMPOSITION_FEATURES
        }

        for feature, value in zip(varied_features, values):
            composition[feature] = float(value)

        nonbalance_sum = sum(
            value for feature, value in composition.items()
            if feature != BALANCE_ELEMENT
        )
        balance_value = TARGET_COMPOSITION_SUM - nonbalance_sum

        if not (MIN_BALANCE_CONTENT <= balance_value <= MAX_BALANCE_CONTENT):
            continue

        composition[BALANCE_ELEMENT] = float(balance_value)

        if abs(sum(composition.values()) - TARGET_COMPOSITION_SUM) > COMPOSITION_SUM_TOLERANCE:
            continue

        if not passes_feasibility(composition):
            continue

        record = {feature: 0.0 for feature in feature_cols}
        record.update(composition)
        record.update(FIXED_PROCESSING)
        record["Temperature"] = float(OXIDATION_TEMPERATURE)
        record["E_Time"] = float(OXIDATION_EXPOSURE_TIME_HOURS)
        record["candidate_id"] = f"C{len(records):06d}"
        records.append(record)

    if not records:
        raise ValueError("No candidate survived balancing/feasibility constraints")

    candidates = pd.DataFrame(records)
    candidates.loc[:, feature_cols] = candidates.loc[:, feature_cols].astype(float)

    if candidates["candidate_id"].nunique() != len(candidates):
        raise RuntimeError("Candidate IDs are not unique")

    print(f"Feasible candidate count: {len(candidates):,}")
    print(f"Varied features: {varied_features}")

    return candidates, varied_features


# =============================================================================
# 7. Candidate prediction + canonical evidence tracing
# =============================================================================

def get_threshold_row(
    thresholds: pd.DataFrame,
    percentile: int,
) -> pd.Series:
    row = thresholds.loc[thresholds["percentile"] == int(percentile)]
    if len(row) != 1:
        raise ValueError(f"Could not resolve one threshold row for P{percentile}")
    return row.iloc[0]


def add_candidate_evidence(
    candidates: pd.DataFrame,
    model,
    evidence_pool: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    thresholds: pd.DataFrame,
    deploy_cfg: dict,
    main_weights: Mapping[str, float],
    protocol_weights: Mapping[str, float],
    evidence_config: EvidenceMetricConfig,
):
    X_train = evidence_pool.loc[:, feature_cols].to_numpy(dtype=float)
    X_query = candidates.loc[:, feature_cols].to_numpy(dtype=float)

    predicted = np.asarray(model.predict(X_query), dtype=float)

    # -------------------------
    # MED: same deployment training-reference scale
    # -------------------------
    train_leaf = ensure_2d_leaf_matrix(model.apply(X_train))
    query_leaf = ensure_2d_leaf_matrix(model.apply(X_query))

    _, med_query, med_nn_idx, med_nn_dist, med_norm = (
        build_med_from_training_reference(
            train_leaf,
            query_leaf,
            evidence_config,
        )
    )

    saved_med_norm = deploy_cfg.get("med_normalization", {})
    check_close_dict(
        med_norm,
        saved_med_norm,
        (
            "leaf_raw_min",
            "leaf_raw_max",
            "support_log1p_min",
            "support_log1p_max",
        ),
        atol=1e-10,
    )

    # -------------------------
    # PED: full 613-record physical evidence pool
    # -------------------------
    ped_raw, ped_nn_idx, ped_nn_dist, _ = compute_ped_external(
        evidence_pool,
        candidates,
        main_weights,
        protocol_weights,
        evidence_config,
    )
    ped_scaled = transform_ped_for_threshold(ped_raw, evidence_config)

    local_stats = local_evidence_statistics(
        evidence_pool,
        ped_nn_idx,
        target_col,
    )

    result = candidates.copy().reset_index(drop=True)
    result["predicted_lgkp"] = predicted
    result = pd.concat(
        [result, med_query.reset_index(drop=True)],
        axis=1,
    )
    result["PED_raw"] = ped_raw
    result["PED_log1p"] = np.log1p(ped_raw)
    result["PED_for_threshold"] = ped_scaled
    result = pd.concat(
        [result, local_stats.reset_index(drop=True)],
        axis=1,
    )

    # State sensitivity P40-P60 (or whatever exists in deployment thresholds).
    for _, thr in thresholds.sort_values("percentile").iterrows():
        p = int(thr["percentile"])
        result[f"state_P{p}"] = assign_quadrants(
            result["MED"].to_numpy(dtype=float),
            result["PED_for_threshold"].to_numpy(dtype=float),
            float(thr["MED_threshold"]),
            float(thr["PED_threshold"]),
        )

    main_thr = get_threshold_row(thresholds, MAIN_PERCENTILE)
    med_threshold = float(main_thr["MED_threshold"])
    ped_threshold = float(main_thr["PED_threshold"])

    result["state"] = result[f"state_P{MAIN_PERCENTILE}"]
    result["MED_threshold_P50"] = med_threshold
    result["PED_threshold_P50"] = ped_threshold

    result["MED_margin_abs"] = np.abs(result["MED"] - med_threshold)
    result["PED_margin_abs"] = np.abs(
        result["PED_for_threshold"] - ped_threshold
    )
    result["normalized_boundary_margin"] = normalized_boundary_margin(
        result["MED"].to_numpy(dtype=float),
        result["PED_for_threshold"].to_numpy(dtype=float),
        med_threshold,
        ped_threshold,
    )

    return (
        result,
        med_nn_idx,
        med_nn_dist,
        ped_nn_idx,
        ped_nn_dist,
        med_threshold,
        ped_threshold,
        med_norm,
    )


# =============================================================================
# 8. Candidate-state summaries
# =============================================================================

def candidate_state_summary(result: pd.DataFrame) -> pd.DataFrame:
    order = ["Q1", "Q2", "Q3", "Q4"]
    rows = []

    for state in order:
        sub = result[result["state"] == state]

        row = {
            "state": state,
            "count": int(len(sub)),
            "fraction": float(len(sub) / len(result)),
        }

        if len(sub) == 0:
            row.update({
                "predicted_lgkp_mean": np.nan,
                "predicted_lgkp_min": np.nan,
                "predicted_lgkp_max": np.nan,
                "MED_mean": np.nan,
                "PED_log1p_mean": np.nan,
                "n_sources_mean": np.nan,
                "lgkp_std_mean": np.nan,
            })
        else:
            row.update({
                "predicted_lgkp_mean": float(sub["predicted_lgkp"].mean()),
                "predicted_lgkp_min": float(sub["predicted_lgkp"].min()),
                "predicted_lgkp_max": float(sub["predicted_lgkp"].max()),
                "MED_mean": float(sub["MED"].mean()),
                "PED_log1p_mean": float(sub["PED_log1p"].mean()),
                "n_sources_mean": float(sub["n_sources"].mean()),
                "lgkp_std_mean": float(sub["lgkp_std"].mean()),
            })

        rows.append(row)

    return pd.DataFrame(rows)


def candidate_state_sensitivity(
    result: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, thr in thresholds.sort_values("percentile").iterrows():
        p = int(thr["percentile"])
        state_col = f"state_P{p}"

        counts = (
            result[state_col]
            .value_counts()
            .reindex(["Q1", "Q2", "Q3", "Q4"], fill_value=0)
        )

        for state in ("Q1", "Q2", "Q3", "Q4"):
            rows.append({
                "percentile": p,
                "MED_threshold": float(thr["MED_threshold"]),
                "PED_threshold": float(thr["PED_threshold"]),
                "state": state,
                "count": int(counts[state]),
                "fraction": float(counts[state] / len(result)),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 9. Prediction-matched population
# =============================================================================

def composition_range_scales(
    varied_features: Sequence[str],
) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for feature in varied_features:
        vmin, vmax, _ = ELEMENT_RANGES[feature]
        width = float(vmax - vmin)
        scales[feature] = width if width > 0 else 1.0
    return scales


def normalized_composition_distance(
    row_a: pd.Series,
    row_b: pd.Series,
    varied_features: Sequence[str],
    scales: Mapping[str, float],
) -> float:
    squared = 0.0
    for feature in varied_features:
        delta = (
            float(row_a[feature]) - float(row_b[feature])
        ) / float(scales[feature])
        squared += delta * delta
    return float(np.sqrt(squared))


def search_pairs_for_contrast(
    result: pd.DataFrame,
    state_a: str,
    state_b: str,
    varied_features: Sequence[str],
    max_delta_prediction: float,
) -> pd.DataFrame:
    """
    Enumerate all cross-state pairs within the specified prediction tolerance.

    Uses sorted-prediction windowing rather than an O(N^2) full Cartesian scan.
    """
    a = result[result["state"] == state_a].copy()
    b = result[result["state"] == state_b].copy()

    columns = [
        "contrast",
        "candidate_A", "state_A", "predicted_lgkp_A", "MED_A", "PED_A",
        "n_sources_A", "lgkp_std_A", "boundary_margin_A",
        "candidate_B", "state_B", "predicted_lgkp_B", "MED_B", "PED_B",
        "n_sources_B", "lgkp_std_B", "boundary_margin_B",
        "delta_predicted_lgkp",
        "normalized_composition_distance",
        "pair_boundary_margin",
        "normalized_evidence_separation",
    ]
    for feature in varied_features:
        columns.extend([f"{feature}_A", f"{feature}_B", f"delta_{feature}"])

    if len(a) == 0 or len(b) == 0:
        return pd.DataFrame(columns=columns)

    b = b.sort_values("predicted_lgkp").reset_index(drop=True)
    b_pred = b["predicted_lgkp"].to_numpy(dtype=float)

    scales = composition_range_scales(varied_features)

    # Thresholds are constant columns in result.
    med_thr = float(result["MED_threshold_P50"].iloc[0])
    ped_thr = float(result["PED_threshold_P50"].iloc[0])

    records: List[dict] = []

    for _, row_a in a.iterrows():
        pred_a = float(row_a["predicted_lgkp"])
        left = int(np.searchsorted(
            b_pred, pred_a - max_delta_prediction, side="left"
        ))
        right = int(np.searchsorted(
            b_pred, pred_a + max_delta_prediction, side="right"
        ))

        if right <= left:
            continue

        for j in range(left, right):
            row_b = b.iloc[j]
            delta_pred = abs(
                pred_a - float(row_b["predicted_lgkp"])
            )

            comp_dist = normalized_composition_distance(
                row_a, row_b, varied_features, scales
            )

            pair_margin = min(
                float(row_a["normalized_boundary_margin"]),
                float(row_b["normalized_boundary_margin"]),
            )

            med_sep = (
                float(row_a["MED"]) - float(row_b["MED"])
            ) / max(abs(med_thr), 1e-12)
            ped_sep = (
                float(row_a["PED_for_threshold"])
                - float(row_b["PED_for_threshold"])
            ) / max(abs(ped_thr), 1e-12)

            record = {
                "contrast": f"{state_a}-{state_b}",
                "candidate_A": row_a["candidate_id"],
                "state_A": state_a,
                "predicted_lgkp_A": float(row_a["predicted_lgkp"]),
                "MED_A": float(row_a["MED"]),
                "PED_A": float(row_a["PED_for_threshold"]),
                "n_sources_A": float(row_a["n_sources"]),
                "lgkp_std_A": float(row_a["lgkp_std"]),
                "boundary_margin_A": float(
                    row_a["normalized_boundary_margin"]
                ),

                "candidate_B": row_b["candidate_id"],
                "state_B": state_b,
                "predicted_lgkp_B": float(row_b["predicted_lgkp"]),
                "MED_B": float(row_b["MED"]),
                "PED_B": float(row_b["PED_for_threshold"]),
                "n_sources_B": float(row_b["n_sources"]),
                "lgkp_std_B": float(row_b["lgkp_std"]),
                "boundary_margin_B": float(
                    row_b["normalized_boundary_margin"]
                ),

                "delta_predicted_lgkp": float(delta_pred),
                "normalized_composition_distance": comp_dist,
                "pair_boundary_margin": pair_margin,
                "normalized_evidence_separation": float(
                    np.sqrt(med_sep ** 2 + ped_sep ** 2)
                ),
            }

            for feature in varied_features:
                va = float(row_a[feature])
                vb = float(row_b[feature])
                record[f"{feature}_A"] = va
                record[f"{feature}_B"] = vb
                record[f"delta_{feature}"] = float(vb - va)

            records.append(record)

            if len(records) > MAX_EXPORTED_PAIRS:
                raise RuntimeError(
                    f"Matched pair count exceeded MAX_EXPORTED_PAIRS="
                    f"{MAX_EXPORTED_PAIRS:,}. Narrow the search grid/tolerance."
                )

    pairs = pd.DataFrame(records)
    if pairs.empty:
        return pd.DataFrame(columns=columns)

    return pairs.sort_values(
        [
            "delta_predicted_lgkp",
            "normalized_composition_distance",
            "pair_boundary_margin",
            "candidate_A",
            "candidate_B",
        ],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)


def build_all_matched_pairs(
    result: pd.DataFrame,
    varied_features: Sequence[str],
) -> pd.DataFrame:
    frames = []

    for state_a, state_b in STATE_CONTRASTS:
        frame = search_pairs_for_contrast(
            result,
            state_a,
            state_b,
            varied_features,
            MAX_DELTA_PREDICTED_LGKP,
        )
        if len(frame) > 0:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_representative_pair_pool(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    if pairs.empty:
        return pairs.copy()

    eligible = pairs[
        (pairs["pair_boundary_margin"] >= MIN_NORMALIZED_BOUNDARY_MARGIN)
        & (
            pairs["normalized_composition_distance"]
            <= MAX_NORMALIZED_COMPOSITION_DISTANCE
        )
    ].copy()

    if eligible.empty:
        return eligible

    eligible = eligible.sort_values(
        [
            "delta_predicted_lgkp",
            "normalized_composition_distance",
            "pair_boundary_margin",
            "candidate_A",
            "candidate_B",
        ],
        ascending=[True, True, False, True, True],
    ).reset_index(drop=True)

    eligible.insert(
        0,
        "selection_rank",
        np.arange(1, len(eligible) + 1, dtype=int),
    )

    return eligible


def matched_pair_population_summary(
    pairs: pd.DataFrame,
    result: pd.DataFrame,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame([{
            "contrast": "none",
            "pair_count": 0,
            "unique_candidate_A": 0,
            "unique_candidate_B": 0,
            "unique_candidates_total": 0,
        }])

    rows = []

    for contrast, sub in pairs.groupby("contrast"):
        unique_candidates = set(sub["candidate_A"]) | set(sub["candidate_B"])

        rows.append({
            "contrast": contrast,
            "pair_count": int(len(sub)),
            "unique_candidate_A": int(sub["candidate_A"].nunique()),
            "unique_candidate_B": int(sub["candidate_B"].nunique()),
            "unique_candidates_total": int(len(unique_candidates)),

            "delta_lgkp_min": float(sub["delta_predicted_lgkp"].min()),
            "delta_lgkp_median": float(sub["delta_predicted_lgkp"].median()),
            "delta_lgkp_mean": float(sub["delta_predicted_lgkp"].mean()),
            "delta_lgkp_max": float(sub["delta_predicted_lgkp"].max()),

            "composition_distance_median": float(
                sub["normalized_composition_distance"].median()
            ),
            "composition_distance_mean": float(
                sub["normalized_composition_distance"].mean()
            ),

            "pair_boundary_margin_median": float(
                sub["pair_boundary_margin"].median()
            ),
            "evidence_separation_median": float(
                sub["normalized_evidence_separation"].median()
            ),
        })

    return pd.DataFrame(rows)


def matched_pair_tolerance_summary(
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for contrast_name in [
        f"{a}-{b}" for a, b in STATE_CONTRASTS
    ]:
        sub = pairs[pairs["contrast"] == contrast_name] if not pairs.empty else pairs

        for tol in PAIR_TOLERANCE_REPORT:
            within = (
                sub[sub["delta_predicted_lgkp"] <= float(tol)]
                if not sub.empty
                else sub
            )
            unique_candidates = (
                set(within["candidate_A"]) | set(within["candidate_B"])
                if not within.empty
                else set()
            )

            rows.append({
                "contrast": contrast_name,
                "delta_lgkp_tolerance": float(tol),
                "pair_count": int(len(within)),
                "unique_candidate_A": (
                    int(within["candidate_A"].nunique())
                    if not within.empty else 0
                ),
                "unique_candidate_B": (
                    int(within["candidate_B"].nunique())
                    if not within.empty else 0
                ),
                "unique_candidates_total": int(len(unique_candidates)),
            })

    return pd.DataFrame(rows)


# =============================================================================
# 10. Threshold sensitivity of matched-pair population
# =============================================================================

def count_prediction_matched_pairs_only(
    result: pd.DataFrame,
    state_col: str,
    state_a: str,
    state_b: str,
    tolerance: float,
) -> Tuple[int, int, int]:
    a = result[result[state_col] == state_a]
    b = result[result[state_col] == state_b]

    if len(a) == 0 or len(b) == 0:
        return 0, 0, 0

    b_pred = np.sort(b["predicted_lgkp"].to_numpy(dtype=float))

    count = 0
    involved_a = set()
    involved_b_ids = set()

    # Keep original B rows sorted so candidate IDs can be recovered.
    b_sorted = b.sort_values("predicted_lgkp").reset_index(drop=True)
    b_pred = b_sorted["predicted_lgkp"].to_numpy(dtype=float)

    for _, row_a in a.iterrows():
        pred_a = float(row_a["predicted_lgkp"])
        left = int(np.searchsorted(
            b_pred, pred_a - tolerance, side="left"
        ))
        right = int(np.searchsorted(
            b_pred, pred_a + tolerance, side="right"
        ))

        if right > left:
            count += right - left
            involved_a.add(str(row_a["candidate_id"]))
            involved_b_ids.update(
                b_sorted.iloc[left:right]["candidate_id"].astype(str).tolist()
            )

    return int(count), int(len(involved_a)), int(len(involved_b_ids))


def pair_count_threshold_sensitivity(
    result: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, thr in thresholds.sort_values("percentile").iterrows():
        p = int(thr["percentile"])
        state_col = f"state_P{p}"

        for state_a, state_b in STATE_CONTRASTS:
            n_pairs, n_a, n_b = count_prediction_matched_pairs_only(
                result,
                state_col,
                state_a,
                state_b,
                MAX_DELTA_PREDICTED_LGKP,
            )

            rows.append({
                "percentile": p,
                "MED_threshold": float(thr["MED_threshold"]),
                "PED_threshold": float(thr["PED_threshold"]),
                "contrast": f"{state_a}-{state_b}",
                "delta_lgkp_tolerance": float(
                    MAX_DELTA_PREDICTED_LGKP
                ),
                "pair_count": n_pairs,
                "unique_state_A_candidates": n_a,
                "unique_state_B_candidates": n_b,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 11. Representative-pair evidence export
# =============================================================================

def export_neighbor_table(
    evidence_pool: pd.DataFrame,
    indices: np.ndarray,
    distances: np.ndarray,
    distance_name: str,
    out_path: Path,
) -> None:
    table = evidence_pool.iloc[indices].copy().reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1, dtype=int))
    table.insert(1, distance_name, np.asarray(distances, dtype=float))
    table.to_csv(out_path, index=False, encoding="utf-8-sig")


def export_representative_pair_evidence(
    representative: pd.Series,
    result: pd.DataFrame,
    evidence_pool: pd.DataFrame,
    med_nn_idx: Sequence[np.ndarray],
    med_nn_dist: Sequence[np.ndarray],
    ped_nn_idx: Sequence[np.ndarray],
    ped_nn_dist: Sequence[np.ndarray],
) -> None:
    candidate_ids = [
        str(representative["candidate_A"]),
        str(representative["candidate_B"]),
    ]

    rep_candidates = result[
        result["candidate_id"].astype(str).isin(candidate_ids)
    ].copy()

    # Preserve A/B order.
    order_map = {
        str(representative["candidate_A"]): 0,
        str(representative["candidate_B"]): 1,
    }
    rep_candidates["_order"] = (
        rep_candidates["candidate_id"].astype(str).map(order_map)
    )
    rep_candidates = (
        rep_candidates.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    rep_candidates.to_csv(
        OUTPUT_DIR / "representative_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for label, candidate_id in zip(("A", "B"), candidate_ids):
        matches = np.where(
            result["candidate_id"].astype(str).to_numpy() == candidate_id
        )[0]

        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely resolve representative candidate {candidate_id}"
            )

        idx = int(matches[0])

        export_neighbor_table(
            evidence_pool,
            med_nn_idx[idx],
            med_nn_dist[idx],
            "leaf_distance",
            OUTPUT_DIR / f"representative_{label}_model_neighbors.csv",
        )
        export_neighbor_table(
            evidence_pool,
            ped_nn_idx[idx],
            ped_nn_dist[idx],
            "physical_distance",
            OUTPUT_DIR / f"representative_{label}_physical_neighbors.csv",
        )


# =============================================================================
# 12. Main
# =============================================================================

def main() -> None:
    (
        model,
        meta,
        deploy_cfg,
        evidence_pool,
        thresholds,
        main_weights,
        protocol_weights,
        weight_cfg,
        evidence_config,
    ) = load_deployment_artifacts()

    feature_cols, target_col = validate_deployment_schema(
        meta,
        evidence_pool,
    )

    main_thr = get_threshold_row(thresholds, MAIN_PERCENTILE)
    med_threshold = float(main_thr["MED_threshold"])
    ped_threshold = float(main_thr["PED_threshold"])

    # Cross-check the saved JSON main threshold against the CSV.
    saved_main = deploy_cfg.get("main_thresholds", {})
    if saved_main:
        if not np.isclose(
            med_threshold,
            float(saved_main["MED_threshold"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("Deployment MED threshold mismatch between CSV and JSON")
        if not np.isclose(
            ped_threshold,
            float(saved_main["PED_threshold"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError("Deployment PED threshold mismatch between CSV and JSON")

    print("=" * 90)
    print("Final prospective evidence-aware screening")
    print("=" * 90)
    print(f"Deployment model: {MODEL_PATH}")
    print(f"Evidence pool: {len(evidence_pool)} literature records")
    print(
        f"Evidence config: TOPK={evidence_config.topk}, "
        f"lambda_protocol={evidence_config.lambda_protocol}, "
        f"route penalty={evidence_config.ht_route_mismatch_penalty}"
    )
    print(
        f"P{MAIN_PERCENTILE} deployment thresholds: "
        f"MED={med_threshold:.6f}, "
        f"{evidence_config.ped_threshold_scale}(PED)={ped_threshold:.6f}"
    )
    print(
        f"Prospective condition: T={OXIDATION_TEMPERATURE:.0f} degC, "
        f"E_Time={OXIDATION_EXPOSURE_TIME_HOURS:.1f} h"
    )

    candidates, varied_features = build_candidate_space(feature_cols)

    (
        result,
        med_nn_idx,
        med_nn_dist,
        ped_nn_idx,
        ped_nn_dist,
        med_threshold,
        ped_threshold,
        med_norm,
    ) = add_candidate_evidence(
        candidates,
        model,
        evidence_pool,
        feature_cols,
        target_col,
        thresholds,
        deploy_cfg,
        main_weights,
        protocol_weights,
        evidence_config,
    )

    # -------------------------------------------------------------------------
    # Candidate population
    # -------------------------------------------------------------------------
    result.to_csv(
        OUTPUT_DIR / "all_screened_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    state_summary = candidate_state_summary(result)
    state_summary.to_csv(
        OUTPUT_DIR / "candidate_state_summary_P50.csv",
        index=False,
        encoding="utf-8-sig",
    )

    state_sensitivity = candidate_state_sensitivity(result, thresholds)
    state_sensitivity.to_csv(
        OUTPUT_DIR / "candidate_state_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # ALL matched-pair population first; representative pair second.
    # -------------------------------------------------------------------------
    all_pairs = build_all_matched_pairs(result, varied_features)
    all_pairs.to_csv(
        OUTPUT_DIR / "all_prediction_matched_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rep_pool = build_representative_pair_pool(all_pairs)
    rep_pool.to_csv(
        OUTPUT_DIR / "representative_pair_pool.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_summary = matched_pair_population_summary(all_pairs, result)
    pair_summary.to_csv(
        OUTPUT_DIR / "matched_pair_population_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tolerance_summary = matched_pair_tolerance_summary(all_pairs)
    tolerance_summary.to_csv(
        OUTPUT_DIR / "matched_pair_tolerance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_sensitivity = pair_count_threshold_sensitivity(
        result,
        thresholds,
    )
    pair_sensitivity.to_csv(
        OUTPUT_DIR / "pair_count_threshold_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Deterministic representative pair
    # -------------------------------------------------------------------------
    if rep_pool.empty:
        representative = pd.DataFrame()
        representative.to_csv(
            OUTPUT_DIR / "representative_pair.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        representative = rep_pool.iloc[[0]].copy()
        representative.to_csv(
            OUTPUT_DIR / "representative_pair.csv",
            index=False,
            encoding="utf-8-sig",
        )

        export_representative_pair_evidence(
            representative.iloc[0],
            result,
            evidence_pool,
            med_nn_idx,
            med_nn_dist,
            ped_nn_idx,
            ped_nn_dist,
        )

    # -------------------------------------------------------------------------
    # Run configuration
    # -------------------------------------------------------------------------
    run_config = {
        "interpretation": (
            "prospective/conceptual screening demonstration; candidates are "
            "experimentally testable hypotheses rather than validated alloy designs"
        ),
        "deployment_model": str(MODEL_PATH),
        "deployment_meta": str(META_PATH),
        "evidence_pool": str(EVIDENCE_POOL_PATH),
        "n_evidence_records": int(len(evidence_pool)),
        "fixed_weight_file": str(FIXED_WEIGHT_PATH),
        "deployment_calibration_file": str(DEPLOYMENT_CONFIG_PATH),
        "deployment_threshold_file": str(DEPLOYMENT_THRESHOLDS_PATH),

        "evidence_config": evidence_config.to_dict(),
        "main_percentile": int(MAIN_PERCENTILE),
        "main_thresholds": {
            "MED_threshold": med_threshold,
            "PED_threshold": ped_threshold,
            "PED_scale": evidence_config.ped_threshold_scale,
        },
        "recomputed_med_normalization": med_norm,

        "screening_scenario": {
            "oxidation_temperature_C": float(OXIDATION_TEMPERATURE),
            "oxidation_exposure_time_h": float(
                OXIDATION_EXPOSURE_TIME_HOURS
            ),
            "fixed_processing": FIXED_PROCESSING,
            "base_composition_at_percent": BASE_COMPOSITION,
            "element_ranges_at_percent": {
                key: list(value)
                for key, value in ELEMENT_RANGES.items()
            },
            "balance_element": BALANCE_ELEMENT,
            "target_composition_sum": TARGET_COMPOSITION_SUM,
            "feasibility_limits": {
                key: list(value)
                for key, value in FEASIBILITY_LIMITS.items()
            },
            "varied_features": list(varied_features),
            "n_candidates": int(len(result)),
        },

        "matched_pair_definition": {
            "state_contrasts": [list(v) for v in STATE_CONTRASTS],
            "max_delta_predicted_lgkp": float(
                MAX_DELTA_PREDICTED_LGKP
            ),
            "pair_tolerance_report": list(PAIR_TOLERANCE_REPORT),
            "all_matched_pairs_are_exported_before_representative_selection": True,
        },

        "representative_pair_rule": {
            "minimum_normalized_boundary_margin": float(
                MIN_NORMALIZED_BOUNDARY_MARGIN
            ),
            "maximum_normalized_composition_distance": float(
                MAX_NORMALIZED_COMPOSITION_DISTANCE
            ),
            "ranking": [
                "delta_predicted_lgkp ascending",
                "normalized_composition_distance ascending",
                "pair_boundary_margin descending",
                "candidate IDs ascending as deterministic tie-breakers",
            ],
        },

        "fixed_weight_config": weight_cfg,
    }

    with (OUTPUT_DIR / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("Prospective candidate population")
    print("=" * 90)
    print(f"Candidates: {len(result):,}")
    print(state_summary.to_string(index=False))

    print("\n" + "=" * 90)
    print("Prediction-matched pair population")
    print("=" * 90)

    if all_pairs.empty:
        print(
            "No candidate pair met the configured state-contrast and "
            "prediction-tolerance criteria."
        )
    else:
        print(pair_summary.to_string(index=False))
        print("\nTolerance summary:")
        print(tolerance_summary.to_string(index=False))

    print("\nRepresentative-pair eligible pool:")
    print(f"  all matched pairs: {len(all_pairs):,}")
    print(f"  eligible after boundary/composition constraints: {len(rep_pool):,}")

    if rep_pool.empty:
        print(
            "\nNo representative pair satisfied the stability constraints. "
            "Do NOT silently relax them. Review representative_pair_pool.csv "
            "requirements explicitly before changing any setting."
        )
    else:
        best = rep_pool.iloc[0]
        print("\nSelected representative pair:")
        fields = [
            "candidate_A", "state_A", "predicted_lgkp_A", "MED_A", "PED_A",
            "candidate_B", "state_B", "predicted_lgkp_B", "MED_B", "PED_B",
            "delta_predicted_lgkp",
            "normalized_composition_distance",
            "pair_boundary_margin",
        ]
        print(best[fields].to_string())

        print("\nVaried compositions:")
        for feature in varied_features:
            print(
                f"  {feature}: "
                f"A={best[f'{feature}_A']:.3f}, "
                f"B={best[f'{feature}_B']:.3f}"
            )

    print(f"\nSaved all prospective outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
