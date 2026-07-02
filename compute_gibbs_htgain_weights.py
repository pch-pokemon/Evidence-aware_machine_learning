# -*- coding: utf-8 -*-
"""
compute_gibbs_htgain_weights.py

### Usage
----
Extract gain feature importance from multiple source-aware XGBoost models, and combine it with the thermodynamic prior constructed by the Gibbs free energy of oxide formation to generate a set of fixed and reproducible PED weights. The generated fixed weights shall remain consistent across all retrospective analyses including Fold 4, 2022 time-cut and 2026 full-data, so as to ensure the comparability of PED scales.

### Logic
--------
1. Extract the actual gain of XGBoost from each model.
2. Normalize the gains within each model to a total sum of 1 to eliminate discrepancies in gain magnitude among different models.
3. Calculate the average of normalized gains across the 5 source-aware folds.
4. Perform z-score standardization on the averaged gains among all features.
5. Construct element weights based on Gibbs oxidation affinity.
6. Assign a maximum fixed prior weight to the oxidation temperature.
7. Conduct unified normalization on element and temperature weights to make the sum of weights in the main physical space equal to 1.
8. The weights of numerical heat treatment features are determined solely by the multi-fold averaged gains and shall be normalized separately.

### Notes
----
- This script only calculates and freezes weights, and does not compute MED/PED.
- The generated final_main_weights.json shall be directly read by the retrospective code aligned with the paper.
- Recalculating weights separately for the 2022 and 2026 models is not recommended, as it will alter the measurement scale of PED.
————————————————
用途
----
从多个 source-aware XGBoost 模型中提取 gain 特征重要性，
与氧化物形成 Gibbs 自由能构建的热力学先验结合，生成一套固定、
可复现的 PED 权重。输出的固定权重应在 Fold 4、2022 time-cut、
2026 full-data 等所有追溯分析中保持不变，以保证 PED 尺度可比较。

逻辑
--------
1. 对每个模型提取 XGBoost 的真实 gain；
2. 在每个模型内部将 gain 归一化为和为 1，避免不同模型的 gain 量级不同；
3. 对 5 个 source-aware fold 的归一化 gain 取平均；
4. 将平均 gain 在特征之间进行 z-score 标准化；
5. 以 Gibbs 氧化亲和力构建元素权重；
6. 为氧化温度设置最高固定先验权重；
7. 对元素与温度权重统一归一化，使主物理空间权重之和为 1；
8. 热处理数值特征的权重仅由多折平均 gain 决定，并单独归一化。

注意
----
- 该脚本只“计算并冻结权重”，不计算 MED/PED。
- 生成的 final_main_weights.json 应由论文对齐版追溯代码直接读取。
- 不建议为 2022 和 2026 模型分别重新计算权重，否则 PED 的度量尺度会改变。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. 用户配置
# ============================================================

# source-aware 五折模型目录
MODEL_DIR = Path("models/xgb_source_kfold")

# 用于构建固定权重的参考模型折
REFERENCE_FOLDS = [1, 2, 3, 4, 5]

# 输出目录
OUT_DIR = Path("models/fixed_evidence_weights")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 温度的物理先验基础权重
TEMP_BASE_WEIGHT = 3.0

# 元素热力学先验映射区间
AFFINITY_BASE_MIN = 1.0
AFFINITY_BASE_MAX = 2.0

# 是否压低基体元素，使其不因含量高而主导物理相似性
BACKGROUND_ELEMENT_CAPS = {
    "x_Co": 0.70,
    "x_Ni": 0.70,
}

# 主物理距离特征
MAIN_FEATURES = [
    "x_Co", "x_Al", "x_W", "x_Ni", "x_Cr", "x_Mo", "x_Fe", "x_Nb",
    "x_C", "x_Hf", "x_Si", "x_Ta", "x_Ti", "x_Y", "x_V", "x_B",
    "x_Zr", "x_Mn", "x_Sc", "x_La", "x_Re", "Temperature",
]

# 热处理数值特征
HT_NUM_FEATURES = [
    "solu_temp", "solu_time",
    "middle_temp", "middle_time",
    "aging_temp", "aging_time",
]

# 氧化物形成 Gibbs 自由能数据
# 只要求内部单位一致；越负表示氧化亲和力越强。
# 计算细节：取800-1000℃范围内各个元素的基本氧化物的平均值作为物理距离计算的先验权重
OXIDE_GIBBS = {
    "x_Co": -266.552,
    "x_Al": -869.066,
    "x_W": -362.5706667,
    "x_Ni": -267.1793333,
    "x_Cr": -552.0046667,
    "x_Mo": -306.770,
    "x_Fe": -356.5848333,
    "x_Nb": -556.2266667,
    "x_C": -396.0336667,
    "x_Hf": -901.774,
    "x_Si": -700.352,
    "x_Ta": -613.413,
    "x_Ti": -732.0493333,
    "x_Y": -1043.434667,
    "x_V": -423.789,
    "x_B": -654.472,
    "x_Zr": -877.5156667,
    "x_Mn": -543.6931667,
    "x_Sc": -1042.593333,
    "x_La": -974.4136667,
    "x_Re": -141.277,
}


# ============================================================
# 2. 工具函数
# ============================================================

def load_meta_and_model(
    model_dir: Path,
    fold: int,
):
    """读取指定折的模型和元信息。"""
    model_path = model_dir / f"xgb_fold{fold}.joblib"
    meta_path = model_dir / f"xgb_fold{fold}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = joblib.load(model_path)
    feature_cols = list(meta["feature_cols"])

    return model, feature_cols, model_path, meta_path


def extract_true_gain(
    model,
    feature_cols: List[str],
) -> pd.Series:
    """
    从 XGBoost Booster 提取 importance_type='gain'。

    返回
    ----
    pd.Series:
        index 为 feature_cols，未被树使用的特征 gain 为 0。
    """
    booster = model.get_booster()
    raw_score = booster.get_score(importance_type="gain")

    mapped: Dict[str, float] = {}

    for key, value in raw_score.items():
        key_str = str(key)

        # sklearn wrapper 有时保留真实特征名
        if key_str in feature_cols:
            mapped[key_str] = float(value)
            continue

        # 有时使用 f0, f1, ...
        if key_str.startswith("f"):
            try:
                idx = int(key_str[1:])
            except ValueError:
                continue

            if 0 <= idx < len(feature_cols):
                mapped[feature_cols[idx]] = float(value)

    gain = pd.Series(mapped, dtype=float)
    gain = gain.reindex(feature_cols).fillna(0.0)

    if float(gain.sum()) <= 0:
        raise ValueError(
            "No positive gain values were extracted from the model. "
            "Please check the saved model and feature metadata."
        )

    return gain


def normalize_nonnegative(values: pd.Series) -> pd.Series:
    """将非负向量归一化为和为 1。"""
    values = values.astype(float).clip(lower=0.0)
    total = float(values.sum())

    if total <= 0:
        return pd.Series(
            np.ones(len(values), dtype=float) / len(values),
            index=values.index,
        )

    return values / total


def collect_cross_fold_gain(
    model_dir: Path,
    folds: Iterable[int],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    提取各折 gain，并在每折内部归一化。

    这样避免不同 XGBoost 模型的原始 gain 量纲或总量不同。
    """
    normalized_gain_by_fold: Dict[str, pd.Series] = {}
    reference_feature_cols: List[str] | None = None

    for fold in folds:
        model, feature_cols, _, _ = load_meta_and_model(model_dir, fold)

        if reference_feature_cols is None:
            reference_feature_cols = feature_cols
        elif feature_cols != reference_feature_cols:
            raise ValueError(
                f"Feature order mismatch in fold {fold}. "
                "All reference models must use identical feature columns."
            )

        raw_gain = extract_true_gain(model, feature_cols)
        normalized_gain = normalize_nonnegative(raw_gain)

        normalized_gain_by_fold[f"gain_fold{fold}"] = normalized_gain

    if reference_feature_cols is None:
        raise ValueError("REFERENCE_FOLDS is empty.")

    gain_table = pd.DataFrame(normalized_gain_by_fold)
    gain_table["gain_mean"] = gain_table.mean(axis=1)
    gain_table["gain_std"] = gain_table[
        [c for c in gain_table.columns if c.startswith("gain_fold")]
    ].std(axis=1, ddof=0)

    return gain_table, reference_feature_cols


def build_thermodynamic_base(
    main_features: List[str],
) -> pd.DataFrame:
    """
    为元素与温度构造热力学基础权重。

    元素：
        使用氧化亲和力 -DeltaG，在所有元素之间线性映射到
        [AFFINITY_BASE_MIN, AFFINITY_BASE_MAX]。

    温度：
        使用独立设定 TEMP_BASE_WEIGHT。
    """
    elemental_features = [
        feature for feature in main_features
        if feature != "Temperature"
    ]

    missing_gibbs = [
        feature for feature in elemental_features
        if feature not in OXIDE_GIBBS
    ]
    if missing_gibbs:
        raise ValueError(
            f"Missing Gibbs values for elemental features: {missing_gibbs}"
        )

    affinities = pd.Series(
        {-feature_index: 0 for feature_index in []},
        dtype=float,
    )
    affinities = pd.Series(
        {
            feature: -float(OXIDE_GIBBS[feature])
            for feature in elemental_features
        },
        dtype=float,
    )

    affinity_min = float(affinities.min())
    affinity_max = float(affinities.max())

    if affinity_max - affinity_min <= 1e-12:
        scaled = pd.Series(
            np.full(len(affinities), AFFINITY_BASE_MIN),
            index=affinities.index,
        )
    else:
        scaled = (
            AFFINITY_BASE_MIN
            + (affinities - affinity_min)
            / (affinity_max - affinity_min)
            * (AFFINITY_BASE_MAX - AFFINITY_BASE_MIN)
        )

    base = scaled.copy()

    # 对指定背景元素设置上限
    for feature, cap in BACKGROUND_ELEMENT_CAPS.items():
        if feature in base.index:
            base.loc[feature] = min(float(base.loc[feature]), float(cap))

    if "Temperature" in main_features:
        base.loc["Temperature"] = float(TEMP_BASE_WEIGHT)

    table = pd.DataFrame(index=main_features)
    table["gibbs_value"] = [
        OXIDE_GIBBS.get(feature, np.nan)
        for feature in main_features
    ]
    table["oxidation_affinity"] = [
        -OXIDE_GIBBS[feature]
        if feature in OXIDE_GIBBS else np.nan
        for feature in main_features
    ]
    table["physics_base"] = base.reindex(main_features)

    return table


def build_gibbs_main_weights(
    main_features: List[str],
) -> pd.DataFrame:
    """
    构建主物理空间的固定权重。

    元素权重：
        仅由代表性氧化物形成 Gibbs 自由能所反映的氧化亲和力决定。

    氧化温度权重：
        使用固定的最高先验基础权重 TEMP_BASE_WEIGHT。

    最终：
        对所有元素与温度基础权重统一归一化，使权重之和为 1。
    """
    table = build_thermodynamic_base(main_features)

    raw = table["physics_base"].astype(float)
    total = float(raw.sum())

    if total <= 0:
        raise ValueError("The Gibbs/temperature base weights sum to zero.")

    table["final_weight"] = raw / total

    return table

def build_heat_treatment_weights(
    gain_table: pd.DataFrame,
    ht_features: List[str],
) -> pd.DataFrame:
    """
    热处理温度/时间不使用 Gibbs 先验，只使用多折平均 gain。

    若所有热处理特征 gain 均为 0，则退化为等权。
    """
    existing = [
        feature for feature in ht_features
        if feature in gain_table.index
    ]

    if not existing:
        return pd.DataFrame(
            columns=[
                "gain_mean",
                "gain_std",
                "final_weight",
            ]
        )

    table = gain_table.loc[existing].copy()
    table["final_weight"] = normalize_nonnegative(
        table["gain_mean"]
    )

    return table


def export_results(
    gain_table: pd.DataFrame,
    main_weight_table: pd.DataFrame,
    ht_weight_table: pd.DataFrame,
) -> None:
    """保存 CSV 和 JSON。"""
    gain_csv = OUT_DIR / "cross_fold_normalized_gain.csv"
    main_csv = OUT_DIR / "main_physics_gain_weights.csv"
    ht_csv = OUT_DIR / "heat_treatment_gain_weights.csv"

    gain_table.to_csv(gain_csv, encoding="utf-8-sig")
    main_weight_table.to_csv(main_csv, encoding="utf-8-sig")
    ht_weight_table.to_csv(ht_csv, encoding="utf-8-sig")

    final_main_weights = {
        feature: float(value)
        for feature, value in main_weight_table[
            "final_weight"
        ].items()
    }

    final_ht_weights = {
        feature: float(value)
        for feature, value in ht_weight_table[
            "final_weight"
        ].items()
    }

    config = {
        "description": (
            "Fixed PED weights with Gibbs-energy-based elemental priors, "
            "a fixed highest prior for oxidation temperature, and "
            "cross-fold normalized XGBoost gain for heat treatment."
        ),
        "reference_model_dir": str(MODEL_DIR),
        "reference_folds": list(REFERENCE_FOLDS),
        "gain_aggregation": (
            "Normalize gain to sum=1 within each fold, then average across folds."
        ),
        "main_weight_formula": (
            "Elemental weights are based only on Gibbs-energy-derived "
            "oxidation affinity; oxidation temperature uses a fixed highest "
            "prior; all main-space weights are normalized to sum=1."
        ),
        "temperature_base_weight": float(TEMP_BASE_WEIGHT),
        "affinity_base_min": float(AFFINITY_BASE_MIN),
        "affinity_base_max": float(AFFINITY_BASE_MAX),
        "background_element_caps": BACKGROUND_ELEMENT_CAPS,
        "main_weights": final_main_weights,
        "heat_treatment_weights": final_ht_weights,
        "usage_note": (
            "Reuse these fixed weights for all folds and all temporal models. "
            "Do not recompute them separately for 2022 and 2026."
        ),
    }

    json_path = OUT_DIR / "final_fixed_ped_weights.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 单独保存可直接复制进代码的字典
    dict_path = OUT_DIR / "fixed_weight_dict.py"
    dict_text = (
        "# Automatically generated by compute_gibbs_htgain_weights.py\n\n"
        f"FIXED_MAIN_WEIGHTS = {repr(final_main_weights)}\n\n"
        f"FIXED_HT_WEIGHTS = {repr(final_ht_weights)}\n"
    )
    dict_path.write_text(dict_text, encoding="utf-8")

    print("\nSaved:")
    print(f"  {gain_csv}")
    print(f"  {main_csv}")
    print(f"  {ht_csv}")
    print(f"  {json_path}")
    print(f"  {dict_path}")


# ============================================================
# 3. 主程序
# ============================================================

def main() -> None:
    gain_table, feature_cols = collect_cross_fold_gain(
        model_dir=MODEL_DIR,
        folds=REFERENCE_FOLDS,
    )

    missing_main = [
        feature for feature in MAIN_FEATURES
        if feature not in feature_cols
    ]
    if missing_main:
        raise ValueError(
            f"These MAIN_FEATURES are absent from model metadata: {missing_main}"
        )

    main_weight_table = build_gibbs_main_weights(
        main_features=MAIN_FEATURES,
    )

    ht_weight_table = build_heat_treatment_weights(
        gain_table=gain_table,
        ht_features=HT_NUM_FEATURES,
    )

    export_results(
        gain_table=gain_table,
        main_weight_table=main_weight_table,
        ht_weight_table=ht_weight_table,
    )

    print("\nFinal main-space weights:")
    print(
        main_weight_table[
            [
                "physics_base",
                "final_weight",
            ]
        ].sort_values("final_weight", ascending=False)
    )

    print("\nFinal heat-treatment weights:")
    print(
        ht_weight_table[
            ["gain_mean", "final_weight"]
        ].sort_values("final_weight", ascending=False)
    )

    print(
        "\nCheck: sum(main weights) = "
        f"{main_weight_table['final_weight'].sum():.12f}"
    )

    if len(ht_weight_table) > 0:
        print(
            "Check: sum(HT weights) = "
            f"{ht_weight_table['final_weight'].sum():.12f}"
        )


if __name__ == "__main__":
    main()
