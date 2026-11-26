# -*- coding: utf-8 -*-
"""
compute_fixed_evidence_weights.py
====================================
Construct fixed and reproducible weights for the final PED definition.

Changes relative to the previous weight script
----------------------------------------------
1) The main physical space remains composition + oxidation temperature.
2) E_Time (oxidation exposure duration) is explicitly added to the continuous
   processing/protocol subspace.
3) The continuous protocol weights are derived from the mean normalized XGBoost
   gain across the five source-aware folds and are normalized across:
       solu_temp, solu_time,
       middle_temp, middle_time,
       aging_temp, aging_time,
       E_Time.
4) Heat-treatment route flags are NOT assigned continuous weights; they are used
   later as categorical route gates / mismatch penalties in evidence_core.py.
5) The resulting JSON is intended to be reused without modification in pooled OOF,
   temporal, full-data calibration and prospective analyses.

Prerequisite
------------
python train_xgb_source_kfold_optuna.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from evidence_core import (
    MAIN_PHYSICAL_FEATURES,
    PROTOCOL_NUM_FEATURES,
    HT_FLAG_FEATURES,
)


# =============================================================================
# 1. Configuration
# =============================================================================

MODEL_DIR = Path("models/xgb_source_kfold")
REFERENCE_FOLDS = (1, 2, 3, 4, 5)

OUT_DIR = Path("models/fixed_evidence_weights")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUT_DIR / "final_fixed_evidence_weights.json"

# Main-space physics priors.
TEMP_BASE_WEIGHT = 3.0
AFFINITY_BASE_MIN = 1.0
AFFINITY_BASE_MAX = 2.0
BACKGROUND_ELEMENT_CAPS = {
    "x_Co": 0.70,
    "x_Ni": 0.70,
}

# Representative oxide-formation Gibbs free energies. More negative = stronger
# oxidation affinity. Internal units only need to remain consistent.
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


# =============================================================================
# 2. Model gain utilities
# =============================================================================

def load_meta_and_model(model_dir: Path, fold: int):
    model_path = model_dir / f"xgb_fold{fold}.joblib"
    meta_path = model_dir / f"xgb_fold{fold}_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    model = joblib.load(model_path)
    feature_cols = list(meta["feature_cols"])
    return model, feature_cols, meta


def extract_true_gain(model, feature_cols: List[str]) -> pd.Series:
    """Extract XGBoost gain and align it to model feature order."""
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")
    mapped: Dict[str, float] = {}

    for key, value in raw.items():
        key_str = str(key)
        if key_str in feature_cols:
            mapped[key_str] = float(value)
            continue
        if key_str.startswith("f"):
            try:
                idx = int(key_str[1:])
            except ValueError:
                continue
            if 0 <= idx < len(feature_cols):
                mapped[feature_cols[idx]] = float(value)

    gain = pd.Series(mapped, dtype=float).reindex(feature_cols).fillna(0.0)
    if float(gain.sum()) <= 0:
        raise ValueError("No positive XGBoost gain values were extracted")
    return gain


def normalize_nonnegative(values: pd.Series) -> pd.Series:
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
    per_fold: Dict[str, pd.Series] = {}
    reference_feature_cols: List[str] | None = None

    for fold in folds:
        model, feature_cols, _ = load_meta_and_model(model_dir, fold)
        if reference_feature_cols is None:
            reference_feature_cols = feature_cols
        elif feature_cols != reference_feature_cols:
            raise ValueError(
                f"Feature-order mismatch in fold {fold}. All folds must use identical features."
            )

        gain = extract_true_gain(model, feature_cols)
        per_fold[f"gain_fold{fold}"] = normalize_nonnegative(gain)

    if reference_feature_cols is None:
        raise ValueError("No reference folds supplied")

    table = pd.DataFrame(per_fold)
    fold_cols = [c for c in table.columns if c.startswith("gain_fold")]
    table["gain_mean"] = table[fold_cols].mean(axis=1)
    table["gain_std"] = table[fold_cols].std(axis=1, ddof=0)
    return table, reference_feature_cols


# =============================================================================
# 3. Main-space Gibbs/temperature weights
# =============================================================================

def build_main_physics_weights() -> pd.DataFrame:
    features = list(MAIN_PHYSICAL_FEATURES)
    element_features = [f for f in features if f != "Temperature"]

    missing = [f for f in element_features if f not in OXIDE_GIBBS]
    if missing:
        raise ValueError(f"Missing Gibbs values for: {missing}")

    affinity = pd.Series(
        {f: -float(OXIDE_GIBBS[f]) for f in element_features},
        dtype=float,
    )
    amin = float(affinity.min())
    amax = float(affinity.max())

    if amax - amin <= 1e-12:
        scaled = pd.Series(AFFINITY_BASE_MIN, index=affinity.index, dtype=float)
    else:
        scaled = (
            AFFINITY_BASE_MIN
            + (affinity - amin) / (amax - amin)
            * (AFFINITY_BASE_MAX - AFFINITY_BASE_MIN)
        )

    base = scaled.copy()
    for feature, cap in BACKGROUND_ELEMENT_CAPS.items():
        if feature in base.index:
            base.loc[feature] = min(float(base.loc[feature]), float(cap))
    base.loc["Temperature"] = float(TEMP_BASE_WEIGHT)

    table = pd.DataFrame(index=features)
    table.index.name = "feature"
    table["gibbs_value"] = [OXIDE_GIBBS.get(f, np.nan) for f in features]
    table["oxidation_affinity"] = [
        -OXIDE_GIBBS[f] if f in OXIDE_GIBBS else np.nan for f in features
    ]
    table["physics_base"] = base.reindex(features)
    table["final_weight"] = table["physics_base"] / float(table["physics_base"].sum())
    return table


# =============================================================================
# 4. Protocol weights: HT numerical variables + E_Time
# =============================================================================

def build_protocol_weights(gain_table: pd.DataFrame) -> pd.DataFrame:
    features = list(PROTOCOL_NUM_FEATURES)
    missing = [f for f in features if f not in gain_table.index]
    if missing:
        raise ValueError(
            "The trained XGBoost feature set does not contain all protocol features: "
            f"{missing}. In particular, confirm that E_Time is present in model metadata."
        )

    table = gain_table.loc[features, ["gain_mean", "gain_std"]].copy()
    table["final_weight"] = normalize_nonnegative(table["gain_mean"])
    return table


# =============================================================================
# 5. Export
# =============================================================================

def export_results(
    gain_table: pd.DataFrame,
    main_table: pd.DataFrame,
    protocol_table: pd.DataFrame,
    feature_cols: List[str],
) -> None:
    gain_path = OUT_DIR / "cross_fold_normalized_gain.csv"
    main_path = OUT_DIR / "main_physics_weights.csv"
    protocol_path = OUT_DIR / "protocol_gain_weights.csv"

    gain_table.to_csv(gain_path, encoding="utf-8-sig")
    main_table.to_csv(main_path, encoding="utf-8-sig")
    protocol_table.to_csv(protocol_path, encoding="utf-8-sig")

    main_weights = {
        f: float(v) for f, v in main_table["final_weight"].items()
    }
    protocol_weights = {
        f: float(v) for f, v in protocol_table["final_weight"].items()
    }

    cfg = {
        "version": "final_exposure_time_aware",
        "description": (
            "Fixed PED weights for the exposure-time-aware evidence pipeline. "
            "The main physical space contains composition and oxidation temperature; "
            "the protocol space contains six heat-treatment numerical variables plus "
            "oxidation exposure duration E_Time."
        ),
        "reference_model_dir": str(MODEL_DIR),
        "reference_folds": list(REFERENCE_FOLDS),
        "reference_model_features": feature_cols,
        "gain_aggregation": (
            "Within each source-aware fold, XGBoost gain is normalized to sum=1; "
            "normalized gains are then averaged across the five folds."
        ),
        "main_features": list(MAIN_PHYSICAL_FEATURES),
        "protocol_features": list(PROTOCOL_NUM_FEATURES),
        "route_flag_features": list(HT_FLAG_FEATURES),
        "exposure_time_feature": "E_Time",
        "main_weight_formula": (
            "Elemental weights are derived from representative oxide-formation Gibbs "
            "free-energy affinity; oxidation temperature uses a fixed highest prior; "
            "all main-space weights are normalized to sum=1."
        ),
        "protocol_weight_formula": (
            "Weights for solu_temp, solu_time, middle_temp, middle_time, aging_temp, "
            "aging_time and E_Time are proportional to their mean normalized XGBoost "
            "gain across the five source-aware folds, then normalized to sum=1 within "
            "the protocol subspace."
        ),
        "temperature_base_weight": float(TEMP_BASE_WEIGHT),
        "affinity_base_min": float(AFFINITY_BASE_MIN),
        "affinity_base_max": float(AFFINITY_BASE_MAX),
        "background_element_caps": BACKGROUND_ELEMENT_CAPS,
        "main_weights": main_weights,
        "protocol_weights": protocol_weights,
        "usage_note": (
            "Reuse these fixed weights unchanged for pooled OOF, temporal, full-data "
            "calibration and prospective analyses. Fold-specific means/stds may be fit "
            "from each training evidence pool, but the weights themselves remain fixed."
        ),
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    dict_path = OUT_DIR / "fixed_weight_dict.py"
    dict_path.write_text(
        "# Automatically generated by compute_fixed_evidence_weights.py\n\n"
        f"FIXED_MAIN_WEIGHTS = {repr(main_weights)}\n\n"
        f"FIXED_PROTOCOL_WEIGHTS = {repr(protocol_weights)}\n",
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"  {gain_path}")
    print(f"  {main_path}")
    print(f"  {protocol_path}")
    print(f"  {OUTPUT_JSON}")
    print(f"  {dict_path}")


# =============================================================================
# 6. Main
# =============================================================================

def main() -> None:
    gain_table, feature_cols = collect_cross_fold_gain(MODEL_DIR, REFERENCE_FOLDS)

    required = list(MAIN_PHYSICAL_FEATURES) + list(PROTOCOL_NUM_FEATURES) + list(HT_FLAG_FEATURES)
    missing = [f for f in required if f not in feature_cols]
    if missing:
        raise ValueError(
            "The current source-aware XGBoost models are missing required evidence features: "
            f"{missing}"
        )

    main_table = build_main_physics_weights()
    protocol_table = build_protocol_weights(gain_table)
    export_results(gain_table, main_table, protocol_table, feature_cols)

    print("\nFinal main-space weights:")
    print(main_table[["physics_base", "final_weight"]].sort_values("final_weight", ascending=False))

    print("\nFinal protocol-space weights (includes E_Time):")
    print(protocol_table[["gain_mean", "gain_std", "final_weight"]].sort_values("final_weight", ascending=False))

    print(f"\nCheck: sum(main weights)     = {main_table['final_weight'].sum():.12f}")
    print(f"Check: sum(protocol weights) = {protocol_table['final_weight'].sum():.12f}")
    print(f"E_Time protocol weight       = {protocol_table.loc['E_Time', 'final_weight']:.8f}")


if __name__ == "__main__":
    main()
