"""Published prospective composition grid and matched-pair definition."""

import itertools
import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from evidence_core import COMPOSITION_FEATURES, HT_FLAG_FEATURES

MAIN_PERCENTILE = 50


OXIDATION_TEMPERATURE = 900.0


OXIDATION_EXPOSURE_TIME_HOURS = 100.0


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


FEASIBILITY_LIMITS: Dict[str, Tuple[float, float]] = {}


MAX_CANDIDATES = 250_000


STATE_CONTRASTS: Sequence[Tuple[str, str]] = (
    ("Q1", "Q4"),
)


MAX_DELTA_PREDICTED_LGKP = 0.10


PAIR_TOLERANCE_REPORT: Sequence[float] = (0.02, 0.05, 0.10)


MIN_NORMALIZED_BOUNDARY_MARGIN = 0.08


MAX_NORMALIZED_COMPOSITION_DISTANCE = 1.00


MAX_EXPORTED_PAIRS = 2_000_000


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
