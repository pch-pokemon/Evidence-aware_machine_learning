# -*- coding: utf-8 -*-
"""
evidence_core.py
================
Single source of truth for the evidence-aware oxidation-kinetics pipeline.

This module centralizes the MED/PED definitions used by:
- pooled source-aware out-of-fold evaluation;
- temporal evidence-state analysis;
- final full-data calibration;
- prospective candidate screening.

Final metric definitions
------------------------
MED
    MED = 0.5 * D_LEAF + 0.5 * D_SUPPORT

    D_LEAF:
        Mean top-k terminal-leaf Hamming distance.
    D_SUPPORT:
        Inverse terminal-leaf occupancy support.

    The raw quantities are normalized with a leave-one-out training-reference
    distribution. Held-out/query samples NEVER define their own normalization.

PED
    d_phys = sqrt((1 - lambda_protocol) * d_main^2
                  + lambda_protocol * d_protocol^2)

    d_main:
        Weighted standardized distance in composition + oxidation temperature.
        Element weights are based on oxide-formation Gibbs-energy priors and
        oxidation temperature has a fixed largest prior weight.

    d_protocol:
        Weighted standardized differences in heat-treatment numerical variables
        plus oxidation exposure duration (E_Time). Heat-treatment numerical
        differences are route-gated: they contribute only when the corresponding
        treatment step is present in both samples. A fixed penalty is added for
        each heat-treatment route mismatch. E_Time always contributes because it
        is an oxidation-exposure variable rather than a heat-treatment route.

    PED:
        Mean d_phys to the top-k nearest training samples.

Important reproducibility choices
---------------------------------
- All numerical means/stds are estimated from the current training evidence pool.
- Fixed feature weights are loaded from a cross-fold weight JSON.
- The same top-k, lambda_protocol and route-mismatch penalty should be reused
  across retrospective, temporal and prospective analyses.
- PED is exported both as raw PED and ln(1 + PED); the log1p scale is only a
  monotonic visualization/thresholding scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. Canonical feature groups
# =============================================================================

COMPOSITION_FEATURES: Tuple[str, ...] = (
    "x_Co", "x_Al", "x_W", "x_Ni", "x_Cr", "x_Mo", "x_Fe", "x_Nb",
    "x_C", "x_Hf", "x_Si", "x_Ta", "x_Ti", "x_Y", "x_V", "x_B",
    "x_Zr", "x_Mn", "x_Sc", "x_La", "x_Re",
)

MAIN_PHYSICAL_FEATURES: Tuple[str, ...] = COMPOSITION_FEATURES + ("Temperature",)

# Continuous processing/protocol variables.  E_Time is oxidation exposure duration.
PROTOCOL_NUM_FEATURES: Tuple[str, ...] = (
    "solu_temp", "solu_time",
    "middle_temp", "middle_time",
    "aging_temp", "aging_time",
    "E_Time",
)

HT_NUM_FEATURES: Tuple[str, ...] = PROTOCOL_NUM_FEATURES[:-1]
HT_FLAG_FEATURES: Tuple[str, ...] = ("has_solu", "has_middle", "has_aging")

HT_STEPS = {
    "solution": ("has_solu", ("solu_temp", "solu_time")),
    "intermediate": ("has_middle", ("middle_temp", "middle_time")),
    "aging": ("has_aging", ("aging_temp", "aging_time")),
}

EXPOSURE_TIME_FEATURE = "E_Time"


# =============================================================================
# 2. Canonical evidence configuration
# =============================================================================

@dataclass(frozen=True)
class EvidenceMetricConfig:
    topk: int = 10
    lambda_protocol: float = 0.15
    ht_route_mismatch_penalty: float = 1.0
    ped_threshold_scale: str = "log1p"
    clip_training_reference_normalization: bool = True

    def validate(self) -> None:
        if int(self.topk) <= 0:
            raise ValueError("topk must be positive")
        if not 0.0 <= float(self.lambda_protocol) <= 1.0:
            raise ValueError("lambda_protocol must lie in [0, 1]")
        if float(self.ht_route_mismatch_penalty) < 0.0:
            raise ValueError("ht_route_mismatch_penalty must be non-negative")
        if self.ped_threshold_scale not in {"raw", "log1p"}:
            raise ValueError("ped_threshold_scale must be 'raw' or 'log1p'")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


DEFAULT_EVIDENCE_CONFIG = EvidenceMetricConfig()


# =============================================================================
# 3. General utilities
# =============================================================================

def safe_std(df: pd.DataFrame) -> pd.Series:
    """Population std with zero/NaN entries replaced by 1."""
    std = df.astype(float).std(ddof=0)
    return std.replace(0.0, 1.0).fillna(1.0)


def ensure_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def ensure_2d_leaf_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected leaf-matrix shape: {arr.shape}")
    return arr


def topk_indices(distances: np.ndarray, k: int) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("distances must be a non-empty one-dimensional array")
    kk = min(int(k), distances.size)
    if kk <= 0:
        raise ValueError("k must be positive")
    if kk == distances.size:
        return np.argsort(distances)
    part = np.argpartition(distances, kk - 1)[:kk]
    return part[np.argsort(distances[part])]


def minmax_fit(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot fit min-max transform on an empty array")
    return float(np.min(arr)), float(np.max(arr))


def minmax_transform(
    values: np.ndarray,
    vmin: float,
    vmax: float,
    *,
    clip: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if vmax - vmin < eps:
        out = np.zeros_like(arr, dtype=float)
    else:
        out = (arr - vmin) / (vmax - vmin)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def normalize_weight_mapping(
    weights: Mapping[str, float],
    features: Sequence[str],
    *,
    name: str,
) -> pd.Series:
    missing = [f for f in features if f not in weights]
    if missing:
        raise ValueError(f"{name} missing weights for: {missing}")
    series = pd.Series({f: float(weights[f]) for f in features}, dtype=float)
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} contains non-finite values")
    if (series < 0).any() or float(series.sum()) <= 0:
        raise ValueError(f"{name} must contain non-negative weights with positive sum")
    return series / float(series.sum())


def identify_source_column(df: pd.DataFrame) -> str | None:
    for c in ("source_key", "Ref_source_num", "source_id"):
        if c in df.columns:
            return c
    return None


def identify_year_column(df: pd.DataFrame) -> str | None:
    for c in ("Year", "year", "publication_year"):
        if c in df.columns:
            return c
    return None


def calc_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }



# =============================================================================
# 4. MED: leaf distance + leaf support, calibrated on training reference
# =============================================================================

def leaf_hamming_distances(train_leaf: np.ndarray, query_leaf: np.ndarray) -> np.ndarray:
    train_leaf = ensure_2d_leaf_matrix(train_leaf)
    query_leaf = np.asarray(query_leaf)
    if query_leaf.ndim != 1:
        raise ValueError("query_leaf must be one-dimensional")
    if train_leaf.shape[1] != query_leaf.shape[0]:
        raise ValueError("Train/query leaf dimensions do not match")
    return np.mean(train_leaf != query_leaf[None, :], axis=1, dtype=float)


def compute_leaf_raw_query(
    train_leaf: np.ndarray,
    query_leaf: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Raw leaf-distance and occupancy-support quantities for external queries."""
    train_leaf = ensure_2d_leaf_matrix(train_leaf)
    query_leaf = ensure_2d_leaf_matrix(query_leaf)
    if train_leaf.shape[1] != query_leaf.shape[1]:
        raise ValueError("Train/query leaf dimensions differ")

    n_query, n_trees = query_leaf.shape
    d_leaf_raw = np.empty(n_query, dtype=float)
    support_raw = np.empty(n_query, dtype=float)
    nn_indices: List[np.ndarray] = []
    nn_distances: List[np.ndarray] = []

    for i, q in enumerate(query_leaf):
        distances = leaf_hamming_distances(train_leaf, q)
        idx = topk_indices(distances, k)
        selected = distances[idx]
        d_leaf_raw[i] = float(np.mean(selected))

        counts = np.empty(n_trees, dtype=float)
        for t in range(n_trees):
            counts[t] = np.count_nonzero(train_leaf[:, t] == q[t])
        support_raw[i] = float(np.mean(counts))

        nn_indices.append(idx)
        nn_distances.append(selected)

    return d_leaf_raw, support_raw, nn_indices, nn_distances


def compute_leaf_raw_training_loo(
    train_leaf: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-out training-reference distribution.

    Self-neighbor distance is excluded, and the pseudo-query itself is subtracted
    from terminal-leaf occupancy counts.
    """
    train_leaf = ensure_2d_leaf_matrix(train_leaf)
    n_train, n_trees = train_leaf.shape
    if n_train <= int(k):
        raise ValueError(f"Training size {n_train} must be > k={k}")

    d_leaf_raw = np.empty(n_train, dtype=float)
    support_raw = np.empty(n_train, dtype=float)

    for i, q in enumerate(train_leaf):
        distances = leaf_hamming_distances(train_leaf, q)
        distances[i] = np.inf
        idx = topk_indices(distances, k)
        d_leaf_raw[i] = float(np.mean(distances[idx]))

        counts = np.empty(n_trees, dtype=float)
        for t in range(n_trees):
            counts[t] = max(np.count_nonzero(train_leaf[:, t] == q[t]) - 1, 0)
        support_raw[i] = float(np.mean(counts))

    return d_leaf_raw, support_raw


def build_med_from_training_reference(
    train_leaf: np.ndarray,
    query_leaf: np.ndarray,
    config: EvidenceMetricConfig = DEFAULT_EVIDENCE_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[np.ndarray], List[np.ndarray], dict]:
    """
    Build training-reference MED and query MED on the SAME training-derived scale.
    """
    config.validate()
    k = int(config.topk)

    tr_leaf_raw, tr_support_raw = compute_leaf_raw_training_loo(train_leaf, k)
    qu_leaf_raw, qu_support_raw, qu_nn_idx, qu_nn_dist = compute_leaf_raw_query(
        train_leaf, query_leaf, k
    )

    leaf_min, leaf_max = minmax_fit(tr_leaf_raw)
    tr_support_log = np.log1p(np.maximum(tr_support_raw, 0.0))
    qu_support_log = np.log1p(np.maximum(qu_support_raw, 0.0))
    support_min, support_max = minmax_fit(tr_support_log)

    clip = bool(config.clip_training_reference_normalization)
    tr_d_leaf = minmax_transform(tr_leaf_raw, leaf_min, leaf_max, clip=clip)
    qu_d_leaf = minmax_transform(qu_leaf_raw, leaf_min, leaf_max, clip=clip)

    # Larger occupancy = stronger support = smaller distance.
    tr_d_support = 1.0 - minmax_transform(
        tr_support_log, support_min, support_max, clip=clip
    )
    qu_d_support = 1.0 - minmax_transform(
        qu_support_log, support_min, support_max, clip=clip
    )

    tr_med = 0.5 * tr_d_leaf + 0.5 * tr_d_support
    qu_med = 0.5 * qu_d_leaf + 0.5 * qu_d_support

    train_ref = pd.DataFrame({
        "d_leaf_raw": tr_leaf_raw,
        "support_raw": tr_support_raw,
        "D_LEAF": tr_d_leaf,
        "D_SUPPORT": tr_d_support,
        "MED": tr_med,
    })
    query_val = pd.DataFrame({
        "d_leaf_raw": qu_leaf_raw,
        "support_raw": qu_support_raw,
        "D_LEAF": qu_d_leaf,
        "D_SUPPORT": qu_d_support,
        "MED": qu_med,
    })

    norm_params = {
        "leaf_raw_min": float(leaf_min),
        "leaf_raw_max": float(leaf_max),
        "support_log1p_min": float(support_min),
        "support_log1p_max": float(support_max),
        "clip": clip,
    }
    return train_ref, query_val, qu_nn_idx, qu_nn_dist, norm_params


# =============================================================================
# 5. PED: composition/temperature main space + protocol space incl. E_Time
# =============================================================================

def build_physical_arrays(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    main_weights: Mapping[str, float],
    protocol_weights: Mapping[str, float],
) -> dict:
    required = list(MAIN_PHYSICAL_FEATURES) + list(PROTOCOL_NUM_FEATURES) + list(HT_FLAG_FEATURES)
    ensure_columns(train_df, required, "training evidence pool")
    ensure_columns(query_df, required, "query data")

    # ----- Main physical space -----
    main_train = train_df.loc[:, MAIN_PHYSICAL_FEATURES].astype(float)
    main_query = query_df.loc[:, MAIN_PHYSICAL_FEATURES].astype(float)
    main_mu = main_train.mean()
    main_sigma = safe_std(main_train)
    w_main = normalize_weight_mapping(main_weights, MAIN_PHYSICAL_FEATURES, name="main_weights")
    sqrt_w_main = np.sqrt(w_main.to_numpy(dtype=float))
    z_main_train = ((main_train - main_mu) / main_sigma).to_numpy() * sqrt_w_main
    z_main_query = ((main_query - main_mu) / main_sigma).to_numpy() * sqrt_w_main

    # ----- Processing/protocol space -----
    protocol_train = train_df.loc[:, PROTOCOL_NUM_FEATURES].astype(float)
    protocol_query = query_df.loc[:, PROTOCOL_NUM_FEATURES].astype(float)
    protocol_mu = protocol_train.mean()
    protocol_sigma = safe_std(protocol_train)
    w_protocol = normalize_weight_mapping(
        protocol_weights, PROTOCOL_NUM_FEATURES, name="protocol_weights"
    )
    sqrt_w_protocol = np.sqrt(w_protocol.to_numpy(dtype=float))
    z_protocol_train = (
        (protocol_train - protocol_mu) / protocol_sigma
    ).to_numpy() * sqrt_w_protocol
    z_protocol_query = (
        (protocol_query - protocol_mu) / protocol_sigma
    ).to_numpy() * sqrt_w_protocol

    flags_train = train_df.loc[:, HT_FLAG_FEATURES].fillna(0).astype(int).to_numpy()
    flags_query = query_df.loc[:, HT_FLAG_FEATURES].fillna(0).astype(int).to_numpy()

    return {
        "z_main_train": z_main_train,
        "z_main_query": z_main_query,
        "z_protocol_train": z_protocol_train,
        "z_protocol_query": z_protocol_query,
        "flags_train": flags_train,
        "flags_query": flags_query,
        "main_mu": main_mu,
        "main_sigma": main_sigma,
        "protocol_mu": protocol_mu,
        "protocol_sigma": protocol_sigma,
        "w_main": w_main,
        "w_protocol": w_protocol,
    }


def physical_distance_components_for_query(
    arrays: dict,
    query_pos: int,
    config: EvidenceMetricConfig = DEFAULT_EVIDENCE_CONFIG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (d_phys, d2_main, d2_protocol) from one query to all training samples.

    Route-gating rules:
    - E_Time contributes for every pair.
    - For each heat-treatment step:
        both absent -> zero numerical contribution;
        one present / one absent -> fixed route mismatch penalty;
        both present -> weighted standardized temp/time differences.
    """
    config.validate()

    z_main_train = arrays["z_main_train"]
    z_main_query = arrays["z_main_query"]
    z_protocol_train = arrays["z_protocol_train"]
    z_protocol_query = arrays["z_protocol_query"]
    flags_train = arrays["flags_train"]
    flags_query = arrays["flags_query"]

    if not 0 <= int(query_pos) < len(z_main_query):
        raise IndexError("query_pos outside query-array range")

    d2_main = np.sum(
        (z_main_train - z_main_query[query_pos][None, :]) ** 2,
        axis=1,
    )

    n_train = len(z_main_train)
    d2_protocol = np.zeros(n_train, dtype=float)
    protocol_idx = {f: i for i, f in enumerate(PROTOCOL_NUM_FEATURES)}
    flag_idx = {f: i for i, f in enumerate(HT_FLAG_FEATURES)}

    # Oxidation exposure duration is always a valid protocol comparison.
    et_idx = protocol_idx[EXPOSURE_TIME_FEATURE]
    d2_protocol += (
        z_protocol_train[:, et_idx] - z_protocol_query[query_pos, et_idx]
    ) ** 2

    # Heat-treatment variables are route-gated.
    for flag_name, num_names in HT_STEPS.values():
        fi = flag_idx[flag_name]
        train_flag = flags_train[:, fi]
        query_flag = int(flags_query[query_pos, fi])

        mismatch = train_flag != query_flag
        both_present = (train_flag == 1) & (query_flag == 1)

        if np.any(mismatch):
            d2_protocol[mismatch] += float(config.ht_route_mismatch_penalty)

        cols = [protocol_idx[name] for name in num_names]
        if np.any(both_present):
            diff = (
                z_protocol_train[both_present][:, cols]
                - z_protocol_query[query_pos, cols]
            )
            d2_protocol[both_present] += np.sum(diff ** 2, axis=1)

    lam = float(config.lambda_protocol)
    d_phys = np.sqrt((1.0 - lam) * d2_main + lam * d2_protocol)
    return d_phys, d2_main, d2_protocol


def compute_ped_external(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    main_weights: Mapping[str, float],
    protocol_weights: Mapping[str, float],
    config: EvidenceMetricConfig = DEFAULT_EVIDENCE_CONFIG,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], dict]:
    """PED for external/held-out query samples against a training evidence pool."""
    config.validate()
    arrays = build_physical_arrays(train_df, query_df, main_weights, protocol_weights)

    ped = np.empty(len(query_df), dtype=float)
    nn_indices: List[np.ndarray] = []
    nn_distances: List[np.ndarray] = []

    for i in range(len(query_df)):
        d, _, _ = physical_distance_components_for_query(arrays, i, config)
        idx = topk_indices(d, config.topk)
        selected = d[idx]
        ped[i] = float(np.mean(selected))
        nn_indices.append(idx)
        nn_distances.append(selected)

    return ped, nn_indices, nn_distances, arrays


def compute_ped_training_loo(
    train_df: pd.DataFrame,
    main_weights: Mapping[str, float],
    protocol_weights: Mapping[str, float],
    config: EvidenceMetricConfig = DEFAULT_EVIDENCE_CONFIG,
) -> np.ndarray:
    """
    Leave-one-out PED reference distribution inside the current training pool.

    Standardization is fitted once on the training pool. Each record's self-distance
    is excluded before selecting its top-k physical neighbors.
    """
    config.validate()
    if len(train_df) <= int(config.topk):
        raise ValueError(f"Training size {len(train_df)} must be > topk={config.topk}")

    arrays = build_physical_arrays(train_df, train_df, main_weights, protocol_weights)
    ped = np.empty(len(train_df), dtype=float)

    for i in range(len(train_df)):
        d, _, _ = physical_distance_components_for_query(arrays, i, config)
        d[i] = np.inf
        idx = topk_indices(d, config.topk)
        ped[i] = float(np.mean(d[idx]))

    return ped


def transform_ped_for_threshold(
    ped_raw: np.ndarray,
    config: EvidenceMetricConfig = DEFAULT_EVIDENCE_CONFIG,
) -> np.ndarray:
    config.validate()
    arr = np.asarray(ped_raw, dtype=float)
    if config.ped_threshold_scale == "raw":
        return arr
    return np.log1p(arr)


def physical_standardization_to_dict(arrays: dict) -> dict:
    """Small JSON-serializable record of fold-specific physical scaling."""
    return {
        "main_mu": {k: float(v) for k, v in arrays["main_mu"].items()},
        "main_sigma": {k: float(v) for k, v in arrays["main_sigma"].items()},
        "protocol_mu": {k: float(v) for k, v in arrays["protocol_mu"].items()},
        "protocol_sigma": {k: float(v) for k, v in arrays["protocol_sigma"].items()},
        "main_weights": {k: float(v) for k, v in arrays["w_main"].items()},
        "protocol_weights": {k: float(v) for k, v in arrays["w_protocol"].items()},
    }


# =============================================================================
# 6. Evidence summaries, thresholds and Q states
# =============================================================================

def local_evidence_statistics(
    train_df: pd.DataFrame,
    neighbor_indices: Sequence[np.ndarray],
    target_col: str,
) -> pd.DataFrame:
    source_col = identify_source_column(train_df)
    year_col = identify_year_column(train_df)
    rows: List[dict] = []

    for idx in neighbor_indices:
        nn = train_df.iloc[idx]
        y = nn[target_col].astype(float)
        rows.append({
            "n_sources": int(nn[source_col].nunique()) if source_col else np.nan,
            "lgkp_std": float(y.std(ddof=0)),
            "neighbor_lgkp_mean": float(y.mean()),
            "neighbor_lgkp_min": float(y.min()),
            "neighbor_lgkp_max": float(y.max()),
            "neighbor_year_min": float(nn[year_col].min()) if year_col else np.nan,
            "neighbor_year_max": float(nn[year_col].max()) if year_col else np.nan,
        })
    return pd.DataFrame(rows)


def derive_training_thresholds(
    med_reference: Sequence[float],
    ped_reference_scaled: Sequence[float],
    percentiles: Sequence[int] = (40, 45, 50, 55, 60),
) -> pd.DataFrame:
    med = np.asarray(med_reference, dtype=float)
    ped = np.asarray(ped_reference_scaled, dtype=float)
    if med.size == 0 or ped.size == 0:
        raise ValueError("Reference arrays must be non-empty")
    rows = []
    for p in percentiles:
        if not 0 <= int(p) <= 100:
            raise ValueError(f"Invalid percentile: {p}")
        rows.append({
            "percentile": int(p),
            "MED_threshold": float(np.percentile(med, p)),
            "PED_threshold": float(np.percentile(ped, p)),
        })
    return pd.DataFrame(rows)


def assign_quadrants(
    med: Sequence[float],
    ped_scaled: Sequence[float],
    med_threshold: float,
    ped_threshold: float,
) -> np.ndarray:
    med_arr = np.asarray(med, dtype=float)
    ped_arr = np.asarray(ped_scaled, dtype=float)
    if med_arr.shape != ped_arr.shape:
        raise ValueError("MED and PED arrays must have the same shape")

    med_low = med_arr <= float(med_threshold)
    ped_low = ped_arr <= float(ped_threshold)
    return np.select(
        [
            med_low & ped_low,
            med_low & ~ped_low,
            ~med_low & ped_low,
        ],
        ["Q1", "Q2", "Q3"],
        default="Q4",
    )


def quadrant_error_statistics(
    df: pd.DataFrame,
    quadrant_col: str,
    *,
    error_col: str = "error",
    abs_error_col: str = "abs_error",
) -> pd.DataFrame:
    ensure_columns(df, [quadrant_col, error_col, abs_error_col, "MED", "PED_log1p"], "quadrant table")
    rows = []
    n_total = len(df)
    for q in ("Q1", "Q2", "Q3", "Q4"):
        sub = df[df[quadrant_col] == q]
        if len(sub) == 0:
            rows.append({
                "quadrant": q,
                "count": 0,
                "fraction": 0.0,
                "MAE": np.nan,
                "RMSE": np.nan,
                "std_abs_error": np.nan,
                "max_abs_error": np.nan,
                "mean_MED": np.nan,
                "mean_PED_log1p": np.nan,
            })
            continue

        err = sub[error_col].to_numpy(dtype=float)
        rows.append({
            "quadrant": q,
            "count": int(len(sub)),
            "fraction": float(len(sub) / n_total),
            "MAE": float(sub[abs_error_col].mean()),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "std_abs_error": float(sub[abs_error_col].std(ddof=0)),
            "max_abs_error": float(sub[abs_error_col].max()),
            "mean_MED": float(sub["MED"].mean()),
            "mean_PED_log1p": float(sub["PED_log1p"].mean()),
        })
    return pd.DataFrame(rows)


def load_main_weights(json_path: str | Path) -> Dict[str, float]:
    """Load the physical priors; protocol weights are calculated for each fitted model."""
    with Path(json_path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    return normalize_weight_mapping(
        config["main_weights"], MAIN_PHYSICAL_FEATURES, name="main_weights"
    ).to_dict()
