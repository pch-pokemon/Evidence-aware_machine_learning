# -*- coding: utf-8 -*-
"""
Evidence tracing of MED and PED.

Manuscript definitions
----------------------
1) Pairwise leaf-space distance
       d_leaf(x_i, x_j) = (1/T) * sum_t I(l_i,t != l_j,t)

2) Model evidence distance (MED)
       MED_i = mean of d_leaf to the k nearest training samples in leaf space

3) Pairwise physical distance
       d_phys(x_i, x_j) = sqrt((1-lambda_HT) * d_main^2
                               + lambda_HT * d_HT^2)

4) Physical evidence distance (PED)
       PED_i = mean of d_phys to the k nearest training samples in physical space

Important implementation choices
--------------------------------
- All standardization parameters are estimated from the training set only.
- Both raw PED and ln(1+PED) are exported. The latter is only a monotonic
  visualization/thresholding scale when PED_SCALE_FOR_QUADRANT='log1p'.
- The top-k neighbors used for MED and PED are selected independently.
- n_sources and lgkp_std are calculated from the PED top-k neighborhood.
- Element/temperature weights and heat-treatment weights are fixed before PED
  calculation and are kept identical across temporal analyses.

Before running, edit MODEL_DIR, DATA_DIR, FOLD, and threshold settings below.
——————————————————————————
# 均值证据距离与物理证据距离的溯源说明
## 定义
1）两个样本之间叶空间距离
$d_{leaf}(x_i, x_j) = (1/T) \cdot \sum_t \mathbb{I}(l_{i,t} \neq l_{j,t})$

2）模型证据距离（MED）
$MED_i$ 为叶空间中与第 $k$ 个最近训练样本之间叶空间距离的平均值

3）两个样本之间物理距离
$d_{phys}(x_i, x_j) = \sqrt{(1-\lambda_{HT}) \cdot d_{main}^2 + \lambda_{HT} \cdot d_{HT}^2}$

4）物理证据距离（PED）
$PED_i$ 为物理空间中与第 $k$ 个最近训练样本之间物理距离的平均值

## 关键实现说明
- 所有标准化参数仅基于训练集进行估算。
- 程序会同时输出原始物理证据距离（PED）与 $\ln(1+\text{PED})$。当参数 `PED_SCALE_FOR_QUADRANT` 设为 `log1p` 时，后者仅用作单调可视化及阈值划分标尺。
- 计算模型证据距离（MED）和物理证据距离（PED）时，所选取的前 $k$ 个近邻样本相互独立。
- 数据源数量（n_sources）与lgkp标准差（lgkp_std）均基于物理证据距离对应的前 $k$ 近邻区域计算得出。

运行前，请在下方修改模型目录（MODEL_DIR）、数据目录（DATA_DIR）、折数（FOLD）以及各项阈值配置。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# 1. Configuration
# =============================================================================

MODEL_DIR = Path("models/xgb_source_kfold")
DATA_DIR = Path("splits/source_kfold")
FOLD = 4
TOPK = 10

# Contribution of heat-treatment distance to total squared physical distance.
LAMBDA_HT = 0.15

# Penalty for a route mismatch for one heat-treatment step, e.g. one sample has
# solution treatment while the other does not. This value is dimensionless
# because all numerical heat-treatment variables are standardized.
HT_ROUTE_MISMATCH_PENALTY = 1.0

# Final normalized weights used for composition + oxidation temperature.
# Keep these values identical to those used to generate the manuscript results.
FIXED_MAIN_WEIGHTS: Dict[str, float] = {
    "x_Co": 0.020723,
    "x_Al": 0.053486,
    "x_W": 0.036866,
    "x_Ni": 0.020723,
    "x_Cr": 0.043082,
    "x_Mo": 0.035035,
    "x_Fe": 0.036669,
    "x_Nb": 0.043220,
    "x_C": 0.037964,
    "x_Hf": 0.054559,
    "x_Si": 0.047950,
    "x_Ta": 0.045097,
    "x_Ti": 0.048990,
    "x_Y": 0.059208,
    "x_V": 0.038874,
    "x_B": 0.046444,
    "x_Zr": 0.053763,
    "x_Mn": 0.042809,
    "x_Sc": 0.059180,
    "x_La": 0.056943,
    "x_Re": 0.029604,
    "Temperature": 0.088812,
}

         
    
# Fixed relative weights within the heat-treatment subspace.
# These are the mean normalized XGBoost gain importances across the
# five source-aware folds and sum to 1.
FIXED_HT_WEIGHTS: Dict[str, float] = {
    "solu_temp": 0.136008,
    "solu_time": 0.273011,
    "middle_temp": 0.175682,
    "middle_time": 0.094346,
    "aging_temp": 0.124761,
    "aging_time": 0.196192,
}


# "raw" means use PED directly; "log1p" means use ln(1 + PED).
PED_SCALE_FOR_QUADRANT = "log1p"

# For a single ordinary fold, None uses the medians of the current test set.
# For 2022-versus-2026 comparison, set explicit fixed thresholds derived from
# the chosen reference model and use the SAME values in both runs.
MED_THRESHOLD: Optional[float] = None
PED_THRESHOLD: Optional[float] = None

# Optional test indices whose top-k evidence tables will be exported.
CASE_INDICES: Sequence[int] = ()

OUTPUT_DIR = MODEL_DIR / "evidence_tracing_aligned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. Data structures
# =============================================================================

@dataclass(frozen=True)
class EvidenceConfig:
    fold: int
    topk: int
    lambda_ht: float
    ht_route_mismatch_penalty: float
    ped_scale_for_quadrant: str
    med_threshold: Optional[float]
    ped_threshold: Optional[float]


# =============================================================================
# 3. General utilities
# =============================================================================

def safe_std(df: pd.DataFrame) -> pd.Series:
    """Training-set standard deviation with zero/NaN values replaced by 1."""
    std = df.std(ddof=0)
    return std.replace(0.0, 1.0).fillna(1.0)


def topk_indices(distances: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the k smallest distances, sorted from small to large."""
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1:
        raise ValueError("distances must be one-dimensional")
    if distances.size == 0:
        raise ValueError("Cannot select neighbors from an empty training set")

    kk = min(int(k), distances.size)
    if kk <= 0:
        raise ValueError("k must be a positive integer")

    if kk == distances.size:
        return np.argsort(distances)

    partial = np.argpartition(distances, kk - 1)[:kk]
    return partial[np.argsort(distances[partial])]


def identify_source_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("source_key", "Ref_source_num", "source_id"):
        if candidate in df.columns:
            return candidate
    return None


def identify_year_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("Year", "year", "publication_year"):
        if candidate in df.columns:
            return candidate
    return None


def normalize_weights(
    features: Sequence[str],
    weight_dict: Dict[str, float],
) -> pd.Series:
    missing = [feature for feature in features if feature not in weight_dict]
    if missing:
        raise ValueError(f"Missing physical weights for: {missing}")

    weights = pd.Series(
        {feature: float(weight_dict[feature]) for feature in features},
        dtype=float,
    )
    if (weights < 0).any() or float(weights.sum()) <= 0:
        raise ValueError("Physical weights must be non-negative with positive sum")
    return weights / float(weights.sum())


def ensure_2d_leaf_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected leaf-matrix shape: {arr.shape}")
    return arr


# =============================================================================
# 4. MED: model evidence space
# =============================================================================

def pairwise_leaf_distance(
    train_leaf: np.ndarray,
    query_leaf: np.ndarray,
) -> np.ndarray:
    """
    Pairwise normalized Hamming distance in terminal-leaf space.

    d_leaf(x_i, x_j) = fraction of trees in which the two samples are assigned
    to different terminal leaves. The value lies in [0, 1].
    """
    train_leaf = ensure_2d_leaf_matrix(train_leaf)
    query_leaf = np.asarray(query_leaf)

    if query_leaf.ndim != 1:
        raise ValueError("query_leaf must be one-dimensional")
    if train_leaf.shape[1] != query_leaf.shape[0]:
        raise ValueError("Train/query leaf dimensions do not match")

    return np.mean(train_leaf != query_leaf[None, :], axis=1, dtype=float)


def compute_med(
    train_leaf: np.ndarray,
    test_leaf: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    MED is exactly the mean pairwise leaf distance to the k nearest training
    samples. No min-max normalization and no additional support term are used.
    """
    train_leaf = ensure_2d_leaf_matrix(train_leaf)
    test_leaf = ensure_2d_leaf_matrix(test_leaf)

    med = np.empty(test_leaf.shape[0], dtype=float)
    neighbor_indices: List[np.ndarray] = []
    neighbor_distances: List[np.ndarray] = []

    for i, query_leaf in enumerate(test_leaf):
        distances = pairwise_leaf_distance(train_leaf, query_leaf)
        idx = topk_indices(distances, k)
        selected = distances[idx]

        med[i] = float(np.mean(selected))
        neighbor_indices.append(idx)
        neighbor_distances.append(selected)

    return med, neighbor_indices, neighbor_distances


# =============================================================================
# 5. PED: physical evidence space
# =============================================================================

HT_STEP_DEFINITIONS = {
    "solution": {
        "flag_candidates": ("has_solu", "has_solution"),
        "temp_candidates": ("solu_temp", "solution_temp"),
        "time_candidates": ("solu_time", "solution_time"),
    },
    "intermediate": {
        "flag_candidates": ("has_middle", "has_intermediate"),
        "temp_candidates": ("middle_temp", "intermediate_temp"),
        "time_candidates": ("middle_time", "intermediate_time"),
    },
    "aging": {
        "flag_candidates": ("has_aging",),
        "temp_candidates": ("aging_temp",),
        "time_candidates": ("aging_time",),
    },
}


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def resolve_ht_steps(columns: Sequence[str]) -> Dict[str, Dict[str, Optional[str]]]:
    resolved: Dict[str, Dict[str, Optional[str]]] = {}
    for step_name, definitions in HT_STEP_DEFINITIONS.items():
        resolved[step_name] = {
            "flag": first_existing(columns, definitions["flag_candidates"]),
            "temp": first_existing(columns, definitions["temp_candidates"]),
            "time": first_existing(columns, definitions["time_candidates"]),
        }
    return resolved


def standardize_from_training(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_numeric = train_df.loc[:, features].astype(float)
    test_numeric = test_df.loc[:, features].astype(float)

    mean = train_numeric.mean()
    std = safe_std(train_numeric)

    train_z = (train_numeric - mean) / std
    test_z = (test_numeric - mean) / std
    return train_z, test_z, mean, std


def build_main_physical_space(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    main_features: Sequence[str],
    fixed_weights: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series]:
    train_z, test_z, mean, std = standardize_from_training(
        train_df, test_df, main_features
    )
    weights = normalize_weights(main_features, fixed_weights)

    sqrt_weights = np.sqrt(weights.to_numpy(dtype=float))
    train_weighted = train_z.to_numpy(dtype=float) * sqrt_weights
    test_weighted = test_z.to_numpy(dtype=float) * sqrt_weights

    return train_weighted, test_weighted, weights, mean, std


def build_ht_standardization(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    resolved_steps: Dict[str, Dict[str, Optional[str]]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    numerical_features: List[str] = []
    for step in resolved_steps.values():
        for field in ("temp", "time"):
            feature = step[field]
            if feature is not None and feature not in numerical_features:
                numerical_features.append(feature)

    if not numerical_features:
        empty_train = pd.DataFrame(index=train_df.index)
        empty_test = pd.DataFrame(index=test_df.index)
        return empty_train, empty_test, pd.Series(dtype=float), pd.Series(dtype=float), []

    train_z, test_z, mean, std = standardize_from_training(
        train_df, test_df, numerical_features
    )
    return train_z, test_z, mean, std, numerical_features


def infer_flag(
    df: pd.DataFrame,
    step: Dict[str, Optional[str]],
) -> np.ndarray:
    """Use explicit has_* flag when present; otherwise infer it from temp/time."""
    flag = step["flag"]
    if flag is not None:
        return df[flag].fillna(0).astype(int).to_numpy()

    available_numeric = [c for c in (step["temp"], step["time"]) if c is not None]
    if not available_numeric:
        return np.zeros(len(df), dtype=int)

    values = df.loc[:, available_numeric].fillna(0).astype(float)
    return (values.abs().sum(axis=1) > 0).astype(int).to_numpy()


def pairwise_ht_squared_distance(
    train_df: pd.DataFrame,
    test_row_position: int,
    test_df: pd.DataFrame,
    train_ht_z: pd.DataFrame,
    test_ht_z: pd.DataFrame,
    resolved_steps: Dict[str, Dict[str, Optional[str]]],
    route_mismatch_penalty: float,
    ht_weights: pd.Series,
) -> np.ndarray:
    """
    Weighted heat-treatment squared distance.

    Numerical heat-treatment variables use fixed relative weights derived from
    cross-fold-averaged XGBoost gain importance.

    For each heat-treatment step:
    - both absent: contribution = 0;
    - route mismatch: contribution =
          route_mismatch_penalty * sum(weights of that step);
    - both present: contribution =
          sum_h w_h * (z_train,h - z_query,h)^2.

    Because the six numerical heat-treatment weights sum to one, the resulting
    d_HT^2 remains on a stable and directly interpretable scale.
    """
    n_train = len(train_df)
    total = np.zeros(n_train, dtype=float)

    for step in resolved_steps.values():
        flag_train = infer_flag(train_df, step)
        flag_test = int(infer_flag(test_df, step)[test_row_position])

        if step["flag"] is None and step["temp"] is None and step["time"] is None:
            continue

        both_present = (flag_train == 1) & (flag_test == 1)
        mismatch = flag_train != flag_test

        available_numeric = [
            feature
            for feature in (step["temp"], step["time"])
            if feature is not None and feature in ht_weights.index
        ]

        # The route penalty is scaled by the total numerical importance of
        # the corresponding treatment step.
        step_weight = float(ht_weights.reindex(available_numeric).fillna(0.0).sum())

        if np.any(mismatch):
            total[mismatch] += float(route_mismatch_penalty) * step_weight

        if available_numeric and np.any(both_present):
            train_values = train_ht_z.loc[:, available_numeric].to_numpy(dtype=float)
            query_values = (
                test_ht_z.iloc[test_row_position][available_numeric]
                .to_numpy(dtype=float)
            )
            weights = ht_weights.loc[available_numeric].to_numpy(dtype=float)
            weighted_squared = np.sum(
                (train_values - query_values[None, :]) ** 2
                * weights[None, :],
                axis=1,
            )
            total[both_present] += weighted_squared[both_present]

    return total

def compute_ped(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    main_features: Sequence[str],
    fixed_weights: Dict[str, float],
    fixed_ht_weights: Dict[str, float],
    k: int,
    lambda_ht: float,
    route_mismatch_penalty: float,
) -> Tuple[
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    Dict[str, Dict[str, Optional[str]]],
]:
    if not 0.0 <= lambda_ht <= 1.0:
        raise ValueError("lambda_ht must lie in [0, 1]")

    train_main, test_main, weights, main_mean, main_std = build_main_physical_space(
        train_df, test_df, main_features, fixed_weights
    )

    resolved_steps = resolve_ht_steps(list(train_df.columns))
    train_ht_z, test_ht_z, ht_mean, ht_std, ht_features = build_ht_standardization(
        train_df, test_df, resolved_steps
    )

    if ht_features:
        ht_weights = normalize_weights(ht_features, fixed_ht_weights)
    else:
        ht_weights = pd.Series(dtype=float)

    ped = np.empty(len(test_df), dtype=float)
    neighbor_indices: List[np.ndarray] = []
    neighbor_distances: List[np.ndarray] = []

    for i in range(len(test_df)):
        # Weighted squared Euclidean distance in composition-temperature space.
        main_squared = np.sum((train_main - test_main[i][None, :]) ** 2, axis=1)

        ht_squared = pairwise_ht_squared_distance(
            train_df=train_df,
            test_row_position=i,
            test_df=test_df,
            train_ht_z=train_ht_z,
            test_ht_z=test_ht_z,
            resolved_steps=resolved_steps,
            route_mismatch_penalty=route_mismatch_penalty,
            ht_weights=ht_weights,
        )

        physical_distance = np.sqrt(
            (1.0 - lambda_ht) * main_squared + lambda_ht * ht_squared
        )

        idx = topk_indices(physical_distance, k)
        selected = physical_distance[idx]

        ped[i] = float(np.mean(selected))
        neighbor_indices.append(idx)
        neighbor_distances.append(selected)

    return (
        ped,
        neighbor_indices,
        neighbor_distances,
        weights,
        main_mean,
        main_std,
        ht_mean,
        ht_std,
        ht_weights,
        resolved_steps,
    )


# =============================================================================
# 6. Local evidence statistics and quadrants
# =============================================================================

def local_physical_evidence_statistics(
    train_df: pd.DataFrame,
    ped_neighbor_indices: Sequence[np.ndarray],
    target_col: str,
    source_col: Optional[str],
    year_col: Optional[str],
) -> pd.DataFrame:
    records = []

    for indices in ped_neighbor_indices:
        neighbors = train_df.iloc[indices]
        target_values = neighbors[target_col].astype(float)

        record = {
            "n_sources": (
                int(neighbors[source_col].nunique()) if source_col is not None else np.nan
            ),
            "lgkp_std": float(target_values.std(ddof=0)),
            "neighbor_lgkp_mean": float(target_values.mean()),
            "neighbor_lgkp_min": float(target_values.min()),
            "neighbor_lgkp_max": float(target_values.max()),
            "year_min": (
                float(neighbors[year_col].min()) if year_col is not None else np.nan
            ),
            "year_max": (
                float(neighbors[year_col].max()) if year_col is not None else np.nan
            ),
        }
        records.append(record)

    return pd.DataFrame(records)


def transform_ped(ped_raw: np.ndarray, scale: str) -> np.ndarray:
    if scale == "raw":
        return np.asarray(ped_raw, dtype=float)
    if scale == "log1p":
        return np.log1p(np.asarray(ped_raw, dtype=float))
    raise ValueError("PED_SCALE_FOR_QUADRANT must be 'raw' or 'log1p'")


def assign_quadrants(
    med: np.ndarray,
    ped_for_quadrant: np.ndarray,
    med_threshold: float,
    ped_threshold: float,
) -> np.ndarray:
    med_low = med <= med_threshold
    ped_low = ped_for_quadrant <= ped_threshold

    return np.select(
        [
            med_low & ped_low,
            med_low & ~ped_low,
            ~med_low & ped_low,
            ~med_low & ~ped_low,
        ],
        ["Q1", "Q2", "Q3", "Q4"],
        default="unknown",
    )


# =============================================================================
# 7. Case export
# =============================================================================

def export_case_neighbors(
    case_index: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    med_indices: Sequence[np.ndarray],
    med_distances: Sequence[np.ndarray],
    ped_indices: Sequence[np.ndarray],
    ped_distances: Sequence[np.ndarray],
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if not 0 <= case_index < len(test_df):
        raise IndexError(f"CASE index {case_index} is outside test-set range")

    case_dir = output_dir / f"case_{case_index}"
    case_dir.mkdir(parents=True, exist_ok=True)

    query = test_df.iloc[[case_index]].copy()
    for column in summary_df.columns:
        query[f"result_{column}"] = summary_df.iloc[case_index][column]
    query.to_csv(case_dir / "query.csv", index=False, encoding="utf-8-sig")

    model_neighbors = train_df.iloc[med_indices[case_index]].copy()
    model_neighbors.insert(0, "rank", np.arange(1, len(model_neighbors) + 1))
    model_neighbors.insert(1, "leaf_distance", med_distances[case_index])
    model_neighbors.to_csv(
        case_dir / "model_space_neighbors.csv", index=False, encoding="utf-8-sig"
    )

    physical_neighbors = train_df.iloc[ped_indices[case_index]].copy()
    physical_neighbors.insert(0, "rank", np.arange(1, len(physical_neighbors) + 1))
    physical_neighbors.insert(1, "physical_distance", ped_distances[case_index])
    physical_neighbors.to_csv(
        case_dir / "physical_space_neighbors.csv", index=False, encoding="utf-8-sig"
    )


# =============================================================================
# 8. Main workflow
# =============================================================================

def main() -> None:
    config = EvidenceConfig(
        fold=FOLD,
        topk=TOPK,
        lambda_ht=LAMBDA_HT,
        ht_route_mismatch_penalty=HT_ROUTE_MISMATCH_PENALTY,
        ped_scale_for_quadrant=PED_SCALE_FOR_QUADRANT,
        med_threshold=MED_THRESHOLD,
        ped_threshold=PED_THRESHOLD,
    )

    model_path = MODEL_DIR / f"xgb_fold{FOLD}.joblib"
    meta_path = MODEL_DIR / f"xgb_fold{FOLD}_meta.json"
    train_path = DATA_DIR / f"fold{FOLD}_train.csv"
    test_path = DATA_DIR / f"fold{FOLD}_test.csv"

    required_paths = [model_path, meta_path, train_path, test_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing_paths))

    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    feature_cols: List[str] = list(meta["feature_cols"])
    target_col: str = str(meta["target_col"])

    model = joblib.load(model_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_columns = set(feature_cols + [target_col])
    for name, df in (("training data", train_df), ("test data", test_df)):
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    x_train = train_df[feature_cols].to_numpy(dtype=float)
    x_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)
    y_pred = np.asarray(model.predict(x_test), dtype=float)

    # Conventional model metrics.
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # -----------------------------
    # MED
    # -----------------------------
    train_leaf = ensure_2d_leaf_matrix(model.apply(x_train))
    test_leaf = ensure_2d_leaf_matrix(model.apply(x_test))
    med, med_indices, med_distances = compute_med(train_leaf, test_leaf, TOPK)

    # -----------------------------
    # PED
    # -----------------------------
    main_features = [feature for feature in feature_cols if feature.startswith("x_")]
    if "Temperature" in feature_cols:
        main_features.append("Temperature")

    if not main_features:
        raise ValueError("No composition or Temperature features found for PED")

    (
        ped_raw,
        ped_indices,
        ped_distances,
        main_weights,
        main_mean,
        main_std,
        ht_mean,
        ht_std,
        ht_weights,
        resolved_steps,
    ) = compute_ped(
        train_df=train_df,
        test_df=test_df,
        main_features=main_features,
        fixed_weights=FIXED_MAIN_WEIGHTS,
        fixed_ht_weights=FIXED_HT_WEIGHTS,
        k=TOPK,
        lambda_ht=LAMBDA_HT,
        route_mismatch_penalty=HT_ROUTE_MISMATCH_PENALTY,
    )

    ped_log1p = np.log1p(ped_raw)
    ped_for_quadrant = transform_ped(ped_raw, PED_SCALE_FOR_QUADRANT)

    med_threshold = (
        float(np.median(med)) if MED_THRESHOLD is None else float(MED_THRESHOLD)
    )
    ped_threshold = (
        float(np.median(ped_for_quadrant))
        if PED_THRESHOLD is None
        else float(PED_THRESHOLD)
    )

    quadrants = assign_quadrants(
        med=med,
        ped_for_quadrant=ped_for_quadrant,
        med_threshold=med_threshold,
        ped_threshold=ped_threshold,
    )

    source_col = identify_source_column(train_df)
    year_col = identify_year_column(train_df)
    local_stats = local_physical_evidence_statistics(
        train_df=train_df,
        ped_neighbor_indices=ped_indices,
        target_col=target_col,
        source_col=source_col,
        year_col=year_col,
    )

    # -----------------------------
    # Summary table
    # -----------------------------
    preferred_identity_columns = [
        "sample_id",
        "source_key",
        "Ref_source_num",
        "Year",
        "Temperature",
        "x_Al",
        "x_Cr",
    ]
    identity_columns = [c for c in preferred_identity_columns if c in test_df.columns]
    summary = test_df[identity_columns].copy()
    summary.insert(0, "test_idx", np.arange(len(test_df), dtype=int))
    summary["y_true"] = y_test
    summary["y_pred"] = y_pred
    summary["error"] = y_pred - y_test
    summary["abs_error"] = np.abs(y_pred - y_test)
    summary["MED"] = med
    summary["PED_raw"] = ped_raw
    summary["PED_log1p"] = ped_log1p
    summary["PED_for_quadrant"] = ped_for_quadrant
    summary["quadrant"] = quadrants
    summary = pd.concat([summary.reset_index(drop=True), local_stats], axis=1)

    summary_path = OUTPUT_DIR / f"fold{FOLD}_evidence_summary_k{TOPK}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Quadrant error statistics.
    quadrant_stats = (
        summary.groupby("quadrant", as_index=False)
        .agg(
            count=("abs_error", "size"),
            MAE=("abs_error", "mean"),
            std_abs_error=("abs_error", lambda x: float(np.std(x, ddof=0))),
            max_abs_error=("abs_error", "max"),
            mean_MED=("MED", "mean"),
            mean_PED_raw=("PED_raw", "mean"),
            mean_PED_log1p=("PED_log1p", "mean"),
        )
    )
    quadrant_order = pd.CategoricalDtype(["Q1", "Q2", "Q3", "Q4"], ordered=True)
    quadrant_stats["quadrant"] = quadrant_stats["quadrant"].astype(quadrant_order)
    quadrant_stats = quadrant_stats.sort_values("quadrant")
    quadrant_stats.to_csv(
        OUTPUT_DIR / f"fold{FOLD}_quadrant_statistics_k{TOPK}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save all settings and training-derived standardization parameters.
    config_payload = {
        "evidence_config": asdict(config),
        "resolved_thresholds": {
            "MED_threshold": med_threshold,
            "PED_threshold": ped_threshold,
            "PED_scale": PED_SCALE_FOR_QUADRANT,
        },
        "model_metrics": {"R2": r2, "MAE": mae, "RMSE": rmse},
        "source_column": source_col,
        "year_column": year_col,
        "main_features": main_features,
        "main_weights_normalized": main_weights.to_dict(),
        "main_training_mean": main_mean.to_dict(),
        "main_training_std": main_std.to_dict(),
        "heat_treatment_weights_normalized": ht_weights.to_dict(),
        "heat_treatment_training_mean": ht_mean.to_dict(),
        "heat_treatment_training_std": ht_std.to_dict(),
        "resolved_heat_treatment_steps": resolved_steps,
        "definitions": {
            "MED": "mean top-k terminal-leaf Hamming distance",
            "PED": "mean top-k physical distance",
            "local_evidence_statistics": "computed from PED top-k neighbors",
        },
    }
    with (OUTPUT_DIR / f"fold{FOLD}_evidence_config_k{TOPK}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config_payload, file, indent=2, ensure_ascii=False)

    for case_index in CASE_INDICES:
        export_case_neighbors(
            case_index=case_index,
            train_df=train_df,
            test_df=test_df,
            med_indices=med_indices,
            med_distances=med_distances,
            ped_indices=ped_indices,
            ped_distances=ped_distances,
            summary_df=summary,
            output_dir=OUTPUT_DIR,
        )

    print("\n=== Evidence tracing completed ===")
    print(f"Fold: {FOLD}")
    print(f"Train/test size: {len(train_df)} / {len(test_df)}")
    print(f"R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    print(f"MED threshold: {med_threshold:.6f}")
    print(
        f"PED threshold ({PED_SCALE_FOR_QUADRANT} scale): "
        f"{ped_threshold:.6f}"
    )
    print(f"Summary saved to: {summary_path}")
    print("\nQuadrant statistics:")
    print(quadrant_stats.to_string(index=False))


if __name__ == "__main__":
    main()
