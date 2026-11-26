# -*- coding: utf-8 -*-
"""compute_temporal_evidence.py

Retrospective evidence evolution (historical <=2022 vs current), paired with the
FROZEN nested source-aware five-fold OOF models.

NO model fitting, NO Optuna, NO modification to any current/historical source.

Per outer fold i:
  1) Verify SAME held-out queries, source isolation, historical training subset
     and fold-specific theta_i* provenance.
  2) Historical MED: historical fitted XGBoost model, historical training LOO
     normalization AND historical training-reference P40-P60 MED thresholds.
     Current MED and MED thresholds: frozen formal nested OOF / calibration.
     MED is model-relative, so raw values are NOT a common metric across eras.
  3) Historical and current PED: SAME current-fold XGB gain protocol weights,
     SAME main physical weights, SAME current-fold mean/std, SAME metric and
     SAME current-fold PED reference P40-P60 thresholds. Only the physical
     reference pool expands (historical <=2022 subset -> all current records).
  4) Require current PED <= historical PED for every query, and require the
     recalculated current PED, model predictions and all P40-P60 Q assignments
     to reproduce the frozen formal nested OOF, sample by sample.
  5) Report transitions, dimension-wise low/high transitions, P40-P60
     sensitivity, paired error changes, illustrative candidates, and audit.

This is a retrospective fixed-configuration *paired reference* comparison.
Theta_i* was selected using contemporary outer-training records. Historical
and current MED use DIFFERENT trained models and relative reference scales.
Thus Q transitions are changes in relative evidence states, not causal effects
or differences in an absolute common MED scale. This is not an as-of-2022
forward-in-time forecast. Training-reference LOO MED is not cross-fitted.

Run from the project root, beside evidence_core.py:
    python compute_temporal_evidence.py
    !python compute_temporal_evidence.py

Optional:
    --project-root PATH
    --year-cutoff 2022
    --expected-records 613    # 0 disables count for synthetic testing
    --no-excel

Requires: numpy, pandas, joblib, scikit-learn, xgboost, evidence_core.py;
          openpyxl for optional Excel export.
"""
from __future__ import annotations

import argparse
import json
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
    build_physical_arrays,
    calc_regression_metrics,
    derive_training_thresholds,
    ensure_2d_leaf_matrix,
    load_main_weights,
    local_evidence_statistics,
    normalize_weight_mapping,
    physical_distance_components_for_query,
    safe_std,
    topk_indices,
    transform_ped_for_threshold,
)

FOLDS = (1, 2, 3, 4, 5)
PERCENTILES = (40, 45, 50, 55, 60)
STATES = ("Q1", "Q2", "Q3", "Q4")
CFG = EvidenceMetricConfig(
    topk=10,
    lambda_protocol=0.15,
    ht_route_mismatch_penalty=1.0,
    ped_threshold_scale="log1p",
    clip_training_reference_normalization=True,
)
SAMPLE, SOURCE, TARGET, YEAR = "sample_id", "source_key", "lgkp", "Year"


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested source-aware retrospective MED/PED/Q transitions")
    p.add_argument("--project-root", default=None,
                   help="Default: directory containing this script")
    p.add_argument("--year-cutoff", type=int, default=2022)
    p.add_argument("--data-dir", default="splits/source_kfold")
    p.add_argument("--current-model-dir", default="models/xgb_source_nested_kfold")
    p.add_argument("--historical-model-dir", default=None,
                   help="Default models/xgb_nested_temporal_<year-cutoff>")
    p.add_argument("--current-evidence-dir", default="models/nested_evidence_final")
    p.add_argument("--main-weights-json",
                   default="models/fixed_evidence_weights/final_fixed_evidence_weights.json")
    p.add_argument("--output-dir", default=None,
                   help="Default models/nested_temporal_evidence_<year-cutoff>_vs_current")
    p.add_argument("--expected-records", type=int, default=613)
    p.add_argument("--no-excel", action="store_true")
    args = p.parse_args()
    if args.year_cutoff < 1900 or args.expected_records < 0:
        p.error("year-cutoff must be >=1900 and expected-records must be >=0")
    return args


def at(root: Path, path: str) -> Path:
    q = Path(path).expanduser()
    return (q if q.is_absolute() else root / q).resolve()


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_csv(p: Path, frame: pd.DataFrame, *, index: bool = False) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(p, index=index, encoding="utf-8-sig")


def require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} lacks columns: {missing}")


def valid_ids(frame: pd.DataFrame, name: str) -> None:
    require(frame, [SAMPLE, SOURCE], name)
    if frame[SAMPLE].isna().any() or frame[SAMPLE].astype(str).duplicated().any():
        raise RuntimeError(f"{name}: null/duplicate sample_id")
    if frame[SOURCE].isna().any():
        raise RuntimeError(f"{name}: missing source_key")


def close(a, b, label: str, atol: float = 2e-7) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError(f"{label}: invalid shape or nonfinite arrays: {a.shape}, {b.shape}")
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    if d > atol:
        raise RuntimeError(f"{label}: max abs difference {d:.12g} > {atol}")
    return d


def model_x(model, frame: pd.DataFrame, features: list[str]):
    names = model.get_booster().feature_names
    if names is not None:
        if list(names) != features:
            raise RuntimeError("Fitted model feature-name order != frozen metadata")
        return frame.loc[:, features].astype(float)
    return frame.loc[:, features].to_numpy(dtype=float)


def check_fixed_theta(fold: int, current_meta: dict, history_meta: dict,
                      current_model, history_model) -> None:
    if int(current_meta.get("fold", -1)) != fold or int(history_meta.get("fold", -1)) != fold:
        raise RuntimeError(f"Fold {fold}: metadata fold identifiers disagree")
    if current_meta.get("best_params") != history_meta.get("best_params"):
        raise RuntimeError(f"Fold {fold}: historical best_params differ from current theta_i")
    if current_meta.get("fixed_params") != history_meta.get("fixed_params"):
        raise RuntimeError(f"Fold {fold}: historical fixed_params differ from current theta_i")
    if not history_meta.get("hyperparameter_match_current", False):
        raise RuntimeError(f"Fold {fold}: historical hyperparameter-match flag is false")
    keys = set(current_meta["best_params"]) | set(current_meta["fixed_params"])
    for k in keys:
        a, b = current_model.get_params().get(k), history_model.get_params().get(k)
        if k == "n_jobs":
            continue  # runtime thread count may be overridden without changing the model
        if isinstance(a, (float, int)) and isinstance(b, (float, int)):
            if not np.isclose(float(a), float(b), rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"Fold {fold}: current/historical model {k} differs")
        elif a != b:
            raise RuntimeError(f"Fold {fold}: current/historical model {k} differs")
    if list(current_meta["feature_cols"]) != list(history_meta["feature_cols"]):
        raise RuntimeError(f"Fold {fold}: feature ordering differs between models")


def protocol_gain(model, features: list[str]) -> dict[str, float]:
    """Recompute the SAME seven gain weights as the formal Nested OOF builder."""
    raw = model.get_booster().get_score(importance_type="gain")
    mapped = pd.Series(0.0, index=features, dtype=float)
    for name, weight in raw.items():
        name = str(name)
        if name not in mapped.index and name.startswith("f"):
            try:
                j = int(name[1:])
            except ValueError:
                continue
            if 0 <= j < len(features):
                name = features[j]
        if name in mapped.index:
            mapped.loc[name] = float(weight)
    if not np.isfinite(mapped.to_numpy()).all() or (mapped < 0).any() or mapped.sum() <= 0:
        raise RuntimeError("Invalid gain weights from current Nested model")
    mapped /= float(mapped.sum())
    weights = mapped.reindex(PROTOCOL_NUM_FEATURES, fill_value=0).copy()
    if weights.sum() <= 0:
        weights[:] = 1.0 / len(weights)
    else:
        weights /= float(weights.sum())
    return {k: float(weights[k]) for k in PROTOCOL_NUM_FEATURES}


def historical_subset(full: pd.DataFrame, saved: pd.DataFrame, year: int,
                      features: list[str], fold: int) -> tuple[pd.DataFrame, np.ndarray]:
    require(full, [SAMPLE, SOURCE, TARGET, YEAR] + features, "current training")
    require(saved, [SAMPLE, SOURCE, TARGET, YEAR] + features, "historical saved training")
    valid_ids(full, "full train")
    valid_ids(saved, "historical train")
    years = pd.to_numeric(full[YEAR], errors="coerce")
    mask = (years <= year).fillna(False).to_numpy(dtype=bool)
    if not mask.any() or int(mask.sum()) <= CFG.topk:
        raise RuntimeError(f"Fold {fold}: insufficient historical reference records")
    expected = full.loc[mask].reset_index(drop=True)
    a_ids = expected[SAMPLE].astype(str).tolist()
    b = saved.copy()
    b[SAMPLE] = b[SAMPLE].astype(str)
    if set(a_ids) != set(b[SAMPLE]):
        raise RuntimeError(f"Fold {fold}: historical train is NOT exact <= {year} subset")
    b = b.set_index(SAMPLE).loc[a_ids].reset_index()
    for col in [TARGET, YEAR] + features:
        close(expected[col], b[col], f"Fold {fold}: hist/full values for {col}", atol=2e-7)
    if not np.array_equal(expected[SOURCE].astype(str), b[SOURCE].astype(str)):
        raise RuntimeError(f"Fold {fold}: historical/full source IDs differ")
    if (pd.to_numeric(b[YEAR], errors="coerce") > year).any():
        raise RuntimeError(f"Fold {fold}: historical reference contains post-cutoff samples")
    return b, mask


def verify_calibration_scale(fold: int, full: pd.DataFrame, arrays: dict, cal: dict,
                             main_prior: dict, current_model, features: list[str]) -> None:
    if int(cal.get("fold", -1)) != fold:
        raise RuntimeError(f"Fold {fold}: frozen calibration belongs to a different fold")
    if cal.get("config") != CFG.to_dict():
        raise RuntimeError(f"Fold {fold}: frozen evidence configuration differs")
    sc = cal["physical_standardization"]
    for domain, cols in (("main", MAIN_PHYSICAL_FEATURES), ("protocol", PROTOCOL_NUM_FEATURES)):
        fitted_mu = full.loc[:, cols].astype(float).mean()
        fitted_sigma = safe_std(full.loc[:, cols])
        close(fitted_mu.to_numpy(), [sc[f"{domain}_mu"][x] for x in cols],
              f"Fold {fold} frozen {domain} mean", atol=2e-9)
        close(fitted_sigma.to_numpy(), [sc[f"{domain}_sigma"][x] for x in cols],
              f"Fold {fold} frozen {domain} std", atol=2e-9)
        close(arrays[f"{domain}_mu"].to_numpy(), fitted_mu.to_numpy(),
              f"Fold {fold} computed {domain} mean", atol=2e-9)
        close(arrays[f"{domain}_sigma"].to_numpy(), fitted_sigma.to_numpy(),
              f"Fold {fold} computed {domain} std", atol=2e-9)
    for name, expected in main_prior.items():
        close([cal["main_weights"][name]], [expected], f"Fold {fold} main prior {name}", 1e-9)
        close([sc["main_weights"][name]], [expected], f"Fold {fold} main scale weight {name}", 1e-9)
    computed_gain = protocol_gain(current_model, features)
    for name, expected in computed_gain.items():
        close([cal["protocol_weights"][name]], [expected], f"Fold {fold} gain {name}", 1e-9)
        close([sc["protocol_weights"][name]], [expected], f"Fold {fold} scale gain {name}", 1e-9)


def common_ped(full: pd.DataFrame, historical: pd.DataFrame, test: pd.DataFrame,
               hist_mask: np.ndarray, main_w: dict, protocol_w: dict):
    """One physical metric fitted on current full_train, restricted reference pools.

    Crucial: apply exact same standardization & gain weights for BOTH periods.
    This makes monotone PED for historical subset -> current superset rigorous.
    """
    arrays = build_physical_arrays(full, test, main_w, protocol_w)
    hist_pos = np.flatnonzero(hist_mask)
    if len(hist_pos) != len(historical):
        raise RuntimeError("Historical subset length inconsistent with PED mask")
    raw_hist = np.empty(len(test), dtype=float)
    raw_current = np.empty(len(test), dtype=float)
    neighbors_hist = []
    full_test_median_nn = []
    for j in range(len(test)):
        d, _, _ = physical_distance_components_for_query(arrays, j, CFG)
        idx_full = topk_indices(d, CFG.topk)
        h_d = d[hist_pos]
        idx_hist = topk_indices(h_d, CFG.topk)
        raw_hist[j] = float(np.mean(h_d[idx_hist]))
        raw_current[j] = float(np.mean(d[idx_full]))
        neighbors_hist.append(idx_hist)
        full_test_median_nn.append(float(np.median(d[idx_full])))
    return raw_hist, raw_current, neighbors_hist, arrays


def transitions(df: pd.DataFrame, history_col: str, current_col: str):
    counts = pd.crosstab(df[history_col], df[current_col]).reindex(
        index=STATES, columns=STATES, fill_value=0
    )
    counts.index.name = "historical"
    counts.columns.name = "current"
    fractions = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    return counts, fractions


def state_component(q: pd.Series, axis: str) -> np.ndarray:
    vals = q.astype(str).to_numpy()
    if axis == "MED":
        return np.isin(vals, ("Q1", "Q2"))
    if axis == "PED":
        return np.isin(vals, ("Q1", "Q3"))
    raise ValueError(axis)


def med_threshold_map(hist_tr_med: pd.DataFrame, current_cal: dict, fold: int) -> pd.DataFrame:
    hist = derive_training_thresholds(hist_tr_med["MED"], [0.0], PERCENTILES)
    # Historical PED threshold from hist not used: only historical MED thresholds.
    cur = current_cal["thresholds_by_percentile"]
    rows = []
    for row in hist.itertuples(index=False):
        p = int(row.percentile)
        if str(p) not in cur:
            raise RuntimeError(f"Fold {fold}: missing frozen P{p} thresholds")
        rows.append({
            "fold": fold, "percentile": p,
            "historical_MED_threshold": float(row.MED_threshold),
            "current_MED_threshold": float(cur[str(p)]["MED_threshold"]),
            "common_PED_threshold": float(cur[str(p)]["PED_threshold"]),
            "historical_MED_reference": "historical_tree_historical_train_LOO",
            "current_MED_reference": "current_tree_current_train_LOO",
            "PED_reference": "current_fold_train_LOO_same_for_both_periods",
        })
    return pd.DataFrame(rows)


def one_fold(fold: int, paths: dict[str, Path], current_oof: pd.DataFrame,
             training_predictions: pd.DataFrame, main_w: dict, cutoff: int):
    full = pd.read_csv(paths["data"] / f"fold{fold}_train.csv", low_memory=False)
    test = pd.read_csv(paths["data"] / f"fold{fold}_test.csv", low_memory=False)
    saved_hist = pd.read_csv(paths["hist"] / f"fold{fold}_train_{cutoff}.csv", low_memory=False)
    cur_meta = read_json(paths["cur"] / f"nested_outer_fold{fold}_meta.json")
    hist_meta = read_json(paths["hist"] / f"nested_temporal_fold{fold}_meta.json")
    cal = read_json(paths["evidence"] / "weights" / f"fold{fold}_evidence_calibration.json")
    cur_model = joblib.load(paths["cur"] / f"nested_outer_fold{fold}.joblib")
    hist_model = joblib.load(paths["hist"] / f"nested_temporal_fold{fold}.joblib")
    features = list(cur_meta["feature_cols"])
    check_fixed_theta(fold, cur_meta, hist_meta, cur_model, hist_model)
    if int(hist_meta.get("year_cutoff", -1)) != cutoff:
        raise RuntimeError(f"Fold {fold}: historical model year cutoff differs")
    valid_ids(test, f"Fold {fold} test")
    train_sources = set(full[SOURCE].astype(str))
    if train_sources & set(test[SOURCE].astype(str)):
        raise RuntimeError(f"Fold {fold}: training and test literature sources overlap")
    if set(full[SAMPLE].astype(str)) & set(test[SAMPLE].astype(str)):
        raise RuntimeError(f"Fold {fold}: training/test sample IDs overlap")
    historical, mask = historical_subset(full, saved_hist, cutoff, features, fold)
    for name, d in (("current train", full), ("test", test), ("history", historical)):
        require(d, list(MAIN_PHYSICAL_FEATURES) + list(PROTOCOL_NUM_FEATURES) + list(HT_FLAG_FEATURES), name)
        x = d.loc[:, features].to_numpy(dtype=float)
        if np.isinf(x).any():
            raise RuntimeError(f"Fold {fold}: {name} contains infinite model features")
    if hist_meta.get("n_train_historical") != len(historical) or hist_meta.get("n_train_full") != len(full):
        raise RuntimeError(f"Fold {fold}: historical metadata training sizes do not match")

    # Reproduce the frozen current formal OOF, and the historical model's saved OOF.
    ref = current_oof.loc[current_oof.fold.eq(fold)].copy().sort_values("fold_test_idx")
    saved = training_predictions.loc[training_predictions.fold.eq(fold)].copy().sort_values("fold_test_idx")
    if len(ref) != len(test) or len(saved) != len(test):
        raise RuntimeError(f"Fold {fold}: current/historical OOF records do not cover fixed test")
    expected_ids = test[SAMPLE].astype(str).to_numpy()
    for label, frame in (("current evidence", ref), ("temporal predictions", saved)):
        if not np.array_equal(frame[SAMPLE].astype(str).to_numpy(), expected_ids):
            raise RuntimeError(f"Fold {fold}: {label} test order/IDs disagree")
        close(frame["y_true"], test[TARGET], f"Fold {fold}: {label} true targets")
    y_true = test[TARGET].to_numpy(dtype=float)
    yc = np.asarray(cur_model.predict(model_x(cur_model, test, features)), dtype=float)
    yh = np.asarray(hist_model.predict(model_x(hist_model, test, features)), dtype=float)
    close(yc, ref["y_pred"], f"Fold {fold}: current official prediction")
    close(yc, saved["y_pred_current"], f"Fold {fold}: frozen current temporal prediction")
    close(yh, saved[f"y_pred_{cutoff}"], f"Fold {fold}: frozen historical temporal prediction")
    close(y_true, saved["y_true"], f"Fold {fold}: historical true targets")

    # Historical MED calibrated on its OWN model and historical reference.
    hist_ref_leaf = ensure_2d_leaf_matrix(hist_model.apply(model_x(hist_model, historical, features)))
    hist_test_leaf = ensure_2d_leaf_matrix(hist_model.apply(model_x(hist_model, test, features)))
    med_train_hist, med_test_hist, _, _, hist_norm = build_med_from_training_reference(
        hist_ref_leaf, hist_test_leaf, CFG
    )
    if not np.isfinite(med_test_hist["MED"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"Fold {fold}: nonfinite historical MED")
    thresholds = med_threshold_map(med_train_hist, cal, fold)
    # Frozen current MED (and all current P40-P60 Q states) must reproduce.
    for p in PERCENTILES:
        cur_t = cal["thresholds_by_percentile"][str(p)]
        recalculated = assign_quadrants(ref["MED"], ref["PED_for_threshold"],
                                       cur_t["MED_threshold"], cur_t["PED_threshold"])
        if not np.array_equal(recalculated, ref[f"quadrant_P{p}"].to_numpy(dtype=str)):
            raise RuntimeError(f"Fold {fold}: current P{p} quadrants not reproducible")

    # PED uses the current Nested model's gain and currently frozen physical
    # reference scale, not the historical model's gain or historical mean/std.
    proto_w = protocol_gain(cur_model, features)
    ped_h, ped_c, neighbors_hist, arrays = common_ped(full, historical, test, mask, main_w, proto_w)
    verify_calibration_scale(fold, full, arrays, cal, main_w, cur_model, features)
    close(ped_c, ref["PED_raw"], f"Fold {fold}: frozen current PED raw", atol=2e-8)
    close(transform_ped_for_threshold(ped_c, CFG), ref["PED_for_threshold"],
          f"Fold {fold}: frozen current PED threshold-scale", atol=2e-8)
    if np.any(ped_c > ped_h + 1e-10):
        bad = np.flatnonzero(ped_c > ped_h + 1e-10)
        raise RuntimeError(f"Fold {fold}: PED monotonicity violated for {len(bad)} queries")
    if np.any((ped_h < 0) | (ped_c < 0)):
        raise RuntimeError(f"Fold {fold}: negative PED")
    hist_local = local_evidence_statistics(historical, neighbors_hist, TARGET)

    cols = [c for c in [SAMPLE, SOURCE, "Ref_source_num", YEAR, "Temperature", "E_Time",
                        "x_Co", "x_Ni", "x_Cr", "x_Al", "x_Ti", "x_W"] if c in test.columns]
    out = test[cols].copy().reset_index(drop=True)
    out.insert(0, "fold", fold)
    out.insert(1, "fold_test_idx", np.arange(len(test)))
    out["y_true"] = y_true
    out[f"y_pred_{cutoff}"] = yh
    out["y_pred_current"] = yc
    out[f"error_{cutoff}"] = yh - y_true
    out["error_current"] = yc - y_true
    out[f"abs_error_{cutoff}"] = np.abs(yh - y_true)
    out["abs_error_current"] = np.abs(yc - y_true)
    out[f"D_LEAF_{cutoff}"] = med_test_hist["D_LEAF"].to_numpy(dtype=float)
    out[f"D_SUPPORT_{cutoff}"] = med_test_hist["D_SUPPORT"].to_numpy(dtype=float)
    out[f"MED_{cutoff}"] = med_test_hist["MED"].to_numpy(dtype=float)
    out[f"d_leaf_raw_{cutoff}"] = med_test_hist["d_leaf_raw"].to_numpy(dtype=float)
    out[f"support_raw_{cutoff}"] = med_test_hist["support_raw"].to_numpy(dtype=float)
    out[f"PED_raw_{cutoff}"] = ped_h
    out[f"PED_log1p_{cutoff}"] = np.log1p(ped_h)
    out[f"PED_for_threshold_{cutoff}"] = transform_ped_for_threshold(ped_h, CFG)
    out[f"n_sources_{cutoff}"] = hist_local["n_sources"].to_numpy()
    for col in ("lgkp_std", "neighbor_lgkp_mean", "neighbor_lgkp_min", "neighbor_lgkp_max",
                "neighbor_year_min", "neighbor_year_max"):
        out[f"{col}_{cutoff}"] = hist_local[col].to_numpy()

    current_fields = ("y_pred", "error", "abs_error", "D_LEAF", "D_SUPPORT", "MED",
                      "d_leaf_raw", "support_raw", "PED_raw", "PED_log1p", "PED_for_threshold",
                      "n_sources", "lgkp_std", "neighbor_lgkp_mean", "neighbor_lgkp_min",
                      "neighbor_lgkp_max", "neighbor_year_min", "neighbor_year_max")
    for col in current_fields:
        if col in ref.columns:
            out[f"{col}_current"] = ref[col].to_numpy()
    close(out["y_pred_current"], ref["y_pred"], f"Fold {fold}: merged frozen prediction")
    for row in thresholds.itertuples(index=False):
        p = int(row.percentile)
        out[f"quadrant_P{p}_{cutoff}"] = assign_quadrants(
            out[f"MED_{cutoff}"], out[f"PED_for_threshold_{cutoff}"],
            row.historical_MED_threshold, row.common_PED_threshold)
        out[f"quadrant_P{p}_current"] = ref[f"quadrant_P{p}"].to_numpy()
    out[f"quadrant_{cutoff}"] = out[f"quadrant_P50_{cutoff}"]
    out["quadrant_current"] = out["quadrant_P50_current"]
    if not np.array_equal(out["quadrant_current"], ref["quadrant"].to_numpy()):
        raise RuntimeError(f"Fold {fold}: current primary Q state mismatch")
    p50 = thresholds.loc[thresholds.percentile.eq(50)].iloc[0]
    out[f"MED_threshold_P50_{cutoff}"] = float(p50.historical_MED_threshold)
    out["MED_threshold_P50_current"] = float(p50.current_MED_threshold)
    out["PED_threshold_P50_common"] = float(p50.common_PED_threshold)
    out[f"MED_minus_threshold_P50_{cutoff}"] = (
        out[f"MED_{cutoff}"] - float(p50.historical_MED_threshold))
    out["MED_minus_threshold_P50_current"] = (
        out["MED_current"] - float(p50.current_MED_threshold))
    out[f"PED_minus_threshold_P50_{cutoff}"] = (
        out[f"PED_for_threshold_{cutoff}"] - float(p50.common_PED_threshold))
    out["PED_minus_threshold_P50_current"] = (
        out["PED_for_threshold_current"] - float(p50.common_PED_threshold))
    out["delta_PED_raw_current_minus_historical"] = ped_c - ped_h
    out["delta_PED_log1p_current_minus_historical"] = (
        out["PED_log1p_current"] - out[f"PED_log1p_{cutoff}"])
    out["PED_improvement_historical_minus_current"] = ped_h - ped_c
    out["delta_abs_error_current_minus_historical"] = (
        out["abs_error_current"] - out[f"abs_error_{cutoff}"])
    out["delta_prediction_current_minus_historical"] = yc - yh
    out["delta_local_sources_current_minus_historical"] = (
        out["n_sources_current"] - out[f"n_sources_{cutoff}"])

    historical_q = out[f"quadrant_{cutoff}"].astype(str)
    current_q = out["quadrant_current"].astype(str)
    out["state_changed"] = historical_q.ne(current_q)
    out["transition"] = historical_q + "->" + current_q
    med_hist_low = state_component(historical_q, "MED")
    med_cur_low = state_component(current_q, "MED")
    ped_hist_low = state_component(historical_q, "PED")
    ped_cur_low = state_component(current_q, "PED")
    out[f"MED_low_{cutoff}"] = med_hist_low
    out["MED_low_current"] = med_cur_low
    out[f"PED_low_{cutoff}"] = ped_hist_low
    out["PED_low_current"] = ped_cur_low
    out["MED_boundary_changed"] = med_hist_low != med_cur_low
    out["PED_boundary_changed"] = ped_hist_low != ped_cur_low
    if np.any(ped_hist_low & (~ped_cur_low)):
        raise RuntimeError(f"Fold {fold}: low PED unexpectedly becomes high with expanded reference")
    out["transition_driver"] = np.select(
        [~out.MED_boundary_changed & ~out.PED_boundary_changed,
         out.MED_boundary_changed & ~out.PED_boundary_changed,
         ~out.MED_boundary_changed & out.PED_boundary_changed,
         out.MED_boundary_changed & out.PED_boundary_changed],
        ["stable", "MED_only", "PED_only", "both"], default="unknown")
    out["transition_class"] = np.select(
        [~out.state_changed,
         historical_q.ne("Q1") & current_q.eq("Q1"),
         historical_q.eq("Q1") & current_q.ne("Q1"),
         historical_q.ne("Q4") & current_q.eq("Q4"),
         historical_q.eq("Q4") & current_q.ne("Q4")],
        ["stable", "into_Q1", "out_of_Q1", "into_Q4", "out_of_Q4"],
        default="other_transition")
    # No across-era delta_MED! Different fitted trees and self-reference scales.
    historical_met = calc_regression_metrics(y_true, yh)
    current_met = calc_regression_metrics(y_true, yc)
    return out, thresholds, historical_met, current_met, {
        "fold": fold,
        "historical_train_rows": len(historical),
        "current_train_rows": len(full),
        "historical_sources": int(historical[SOURCE].nunique()),
        "current_sources": int(full[SOURCE].nunique()),
        "test_rows": len(test),
        "test_sources": int(test[SOURCE].nunique()),
        "historical_MED_normalization": hist_norm,
        "P50_historical_MED": float(p50.historical_MED_threshold),
        "P50_current_MED": float(p50.current_MED_threshold),
        "P50_common_PED": float(p50.common_PED_threshold),
        "n_state_changed": int(out.state_changed.sum()),
        "n_MED_boundary_changed": int(out.MED_boundary_changed.sum()),
        "n_PED_boundary_changed": int(out.PED_boundary_changed.sum()),
        "n_PED_monotonicity_violations": int(np.sum(ped_c > ped_h + 1e-10)),
        "max_abs_current_PED_reproduction_error": close(ped_c, ref["PED_raw"], "current PED", 2e-8),
        "historical_prediction_metrics": historical_met,
        "current_prediction_metrics": current_met,
    }


def export_excel(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in frames.items():
            sheet = name[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="17365D")
            for i, col in enumerate(df.columns, 1):
                ws.column_dimensions[get_column_letter(i)].width = min(45, max(12, len(str(col)) + 2))


def main() -> None:
    args = arguments()
    CFG.validate()
    root = Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd()
    hist_name = args.historical_model_dir or f"models/xgb_nested_temporal_{args.year_cutoff}"
    out_name = args.output_dir or f"models/nested_temporal_evidence_{args.year_cutoff}_vs_current"
    paths = {
        "data": at(root, args.data_dir),
        "cur": at(root, args.current_model_dir),
        "hist": at(root, hist_name),
        "evidence": at(root, args.current_evidence_dir),
        "priors": at(root, args.main_weights_json),
        "out": at(root, out_name),
    }
    for key in ("data", "cur", "hist", "evidence", "priors"):
        p = paths[key].parent if key == "priors" else paths[key]
        if paths["out"] == p or p in paths["out"].parents or paths["out"] in p.parents:
            raise RuntimeError(f"Output directory overlaps protected {key} directory: {p}")
    requisite = [paths["evidence"] / "pooled_source_oof_evidence.csv",
                 paths["hist"] / "temporal_oof_predictions.csv", paths["priors"]]
    for f in FOLDS:
        requisite.extend([
            paths["data"] / f"fold{f}_train.csv",
            paths["data"] / f"fold{f}_test.csv",
            paths["cur"] / f"nested_outer_fold{f}.joblib",
            paths["cur"] / f"nested_outer_fold{f}_meta.json",
            paths["hist"] / f"nested_temporal_fold{f}.joblib",
            paths["hist"] / f"nested_temporal_fold{f}_meta.json",
            paths["hist"] / f"fold{f}_train_{args.year_cutoff}.csv",
            paths["evidence"] / "weights" / f"fold{f}_evidence_calibration.json",
        ])
    missing = [str(p) for p in requisite if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))
    main_w = load_main_weights(paths["priors"])
    cur = pd.read_csv(requisite[0], low_memory=False)
    saved = pd.read_csv(requisite[1], low_memory=False)
    for name, frame in (("formal current OOF", cur), ("historical prediction OOF", saved)):
        require(frame, ["fold", "fold_test_idx", SAMPLE, SOURCE, "y_true"], name)
        valid_ids(frame, name)
    if len(cur) != len(saved):
        raise RuntimeError("Current and historical OOF row counts differ")
    if args.expected_records and len(cur) != args.expected_records:
        raise RuntimeError(f"Current OOF has {len(cur)} rows, expected {args.expected_records}")
    for name, frame in (("current", cur), ("historical", saved)):
        if frame.groupby(SOURCE)["fold"].nunique().max() != 1:
            raise RuntimeError(f"{name}: literature source appears in multiple held-out folds")
    print("=" * 102)
    print("NESTED RETROSPECTIVE TEMPORAL OOF EVIDENCE (NO TRAINING)")
    print(f"Project: {root}")
    print(f"Historical cutoff: Year <= {args.year_cutoff} | SAME held-out queries in both periods")
    print(f"Current evidence: {paths['evidence']}")
    print(f"Historical trained models: {paths['hist']}")
    print(f"Output: {paths['out']}")
    print(f"Evidence config: {CFG.to_dict()}")
    print("Historical MED: own historical tree + own LOO normalization + own P40-P60 thresholds")
    print("Current MED: frozen Nested OOF + current training reference thresholds")
    print("Both PED: frozen CURRENT-fold scale, gains and P40-P60 PED thresholds")
    print("NO raw MED difference is used to characterize temporal evidence gain")
    print("=" * 102, flush=True)

    parts, thresholds, fold_metrics, details = [], [], [], []
    heldout_sources = set()
    for f in FOLDS:
        print(f"\nProcessing paired historical/current fold {f}/5...", flush=True)
        t, thr, hist_m, curr_m, info = one_fold(f, paths, cur, saved, main_w, args.year_cutoff)
        sources_f = set(t[SOURCE].astype(str))
        if heldout_sources & sources_f:
            raise RuntimeError(f"Fold {f}: test literature sources also occur in other outer folds")
        heldout_sources.update(sources_f)
        parts.append(t)
        thresholds.append(thr)
        fold_metrics.append({"fold": f, "n_test": len(t),
                             "n_train_historical": info["historical_train_rows"],
                             "n_train_current": info["current_train_rows"],
                             **{f"historical_{k}": v for k, v in hist_m.items()},
                             **{f"current_{k}": v for k, v in curr_m.items()}})
        details.append(info)
        counts = t[f"quadrant_{args.year_cutoff}"].value_counts().reindex(STATES, fill_value=0)
        print(f"  Historical/current R2={hist_m['R2']:.5f}/{curr_m['R2']:.5f}; "
              f"hist Q={counts.to_dict()}; transitions={int(t.state_changed.sum())}/{len(t)}; "
              f"PED monotonicity PASS", flush=True)

    comparison = pd.concat(parts, ignore_index=True)
    threshold_table = pd.concat(thresholds, ignore_index=True)
    metric_table = pd.DataFrame(fold_metrics)
    if args.expected_records and len(comparison) != args.expected_records:
        raise RuntimeError("Pooled result differs from expected OOF record count")
    valid_ids(comparison, "pooled temporal evidence")
    if not set(comparison[SAMPLE].astype(str)).issubset(set(cur[SAMPLE].astype(str))):
        raise RuntimeError("Temporal output contains sample IDs not in frozen current OOF")
    if set(comparison[SAMPLE].astype(str)) != set(cur[SAMPLE].astype(str)):
        raise RuntimeError("Temporal output does not cover every frozen current OOF sample")
    qh, qc = f"quadrant_{args.year_cutoff}", "quadrant_current"
    transition_counts, transition_frac = transitions(comparison, qh, qc)
    hist_perf = calc_regression_metrics(comparison.y_true, comparison[f"y_pred_{args.year_cutoff}"])
    curr_perf = calc_regression_metrics(comparison.y_true, comparison.y_pred_current)
    changed = int(comparison.state_changed.sum())
    drivers = comparison.transition_driver.value_counts().reindex(
        ["stable", "MED_only", "PED_only", "both"], fill_value=0).rename_axis("driver").reset_index(name="count")
    drivers["fraction"] = drivers["count"] / len(comparison)
    axis_rows = []
    for axis in ("MED", "PED"):
        old = comparison[f"{axis}_low_{args.year_cutoff}"].astype(bool)
        new = comparison[f"{axis}_low_current"].astype(bool)
        axis_rows.append({"axis": axis, "high_to_low": int((~old & new).sum()),
                          "low_to_high": int((old & ~new).sum()),
                          "stable_low": int((old & new).sum()),
                          "stable_high": int((~old & ~new).sum())})
    axis_transition = pd.DataFrame(axis_rows)
    sensitivity = []
    for p in PERCENTILES:
        h = comparison[f"quadrant_P{p}_{args.year_cutoff}"]
        c = comparison[f"quadrant_P{p}_current"]
        matrix, _ = transitions(comparison, f"quadrant_P{p}_{args.year_cutoff}", f"quadrant_P{p}_current")
        sensitivity.append({
            "percentile": p,
            "historical_Q1": int((h == "Q1").sum()),
            "historical_Q2": int((h == "Q2").sum()),
            "historical_Q3": int((h == "Q3").sum()),
            "historical_Q4": int((h == "Q4").sum()),
            "current_Q1": int((c == "Q1").sum()),
            "current_Q2": int((c == "Q2").sum()),
            "current_Q3": int((c == "Q3").sum()),
            "current_Q4": int((c == "Q4").sum()),
            "n_changed": int((h != c).sum()),
            "fraction_changed": float((h != c).mean()),
            "Q4_to_Q1": int(matrix.loc["Q4", "Q1"]),
            "Q1_to_Q4": int(matrix.loc["Q1", "Q4"]),
        })
    sensitivity = pd.DataFrame(sensitivity)
    fold_transition = comparison.groupby("fold", as_index=False).agg(
        n_test=(SAMPLE, "size"), n_changed=("state_changed", "sum"),
        n_MED_change=("MED_boundary_changed", "sum"),
        n_PED_change=("PED_boundary_changed", "sum"),
        median_PED_improvement=("PED_improvement_historical_minus_current", "median"),
    )
    fold_transition["fraction_changed"] = fold_transition.n_changed / fold_transition.n_test
    class_summary = comparison.groupby("transition_class", as_index=False).agg(
        count=(SAMPLE, "size"),
        median_PED_improvement=("PED_improvement_historical_minus_current", "median"),
        mean_delta_abs_error=("delta_abs_error_current_minus_historical", "mean"),
        mean_delta_n_sources=("delta_local_sources_current_minus_historical", "mean"),
    )
    class_summary["fraction"] = class_summary["count"] / len(comparison)
    y = args.year_cutoff
    q4_to_q1 = comparison.loc[(comparison[qh] == "Q4") & (comparison[qc] == "Q1")].sort_values(
        ["PED_improvement_historical_minus_current", "sample_id"], ascending=[False, True])
    into_q1 = comparison.loc[(comparison[qh] != "Q1") & (comparison[qc] == "Q1")].sort_values(
        ["PED_improvement_historical_minus_current", "sample_id"], ascending=[False, True])
    into_q4 = comparison.loc[(comparison[qh] != "Q4") & (comparison[qc] == "Q4")].sort_values(
        ["PED_improvement_historical_minus_current", "sample_id"], ascending=[False, True])
    print("\n" + "=" * 102)
    print("FINAL NESTED TEMPORAL EVIDENCE SUMMARY")
    print("=" * 102)
    print("Historical pooled metrics:", {k: round(v, 5) for k, v in hist_perf.items()})
    print("Current pooled metrics   :", {k: round(v, 5) for k, v in curr_perf.items()})
    print("Historical Q counts:", comparison[qh].value_counts().reindex(STATES, fill_value=0).to_dict())
    print("Current Q counts   :", comparison[qc].value_counts().reindex(STATES, fill_value=0).to_dict())
    print("Historical -> current transition matrix:")
    print(transition_counts.to_string())
    print(f"Changed: {changed}/{len(comparison)} = {changed/len(comparison):.2%}")
    print("Dimension-specific transitions:")
    print(axis_transition.to_string(index=False))
    print("Transition drivers:")
    print(drivers.to_string(index=False))
    print("P40-P60 sensitivity:")
    print(sensitivity.to_string(index=False))
    print("PED monotonicity: PASS (all current <= historical)")
    print("Frozen current predictions/PED/P40-P60 Q: PASS")
    print("Historical saved OOF predictions: PASS")
    print("No cross-model raw MED difference computed.")

    # Save only after all checks have succeeded (existing model/evidence unaffected).
    out = paths["out"]
    out.mkdir(parents=True, exist_ok=True)
    historical_cols = (["fold", "fold_test_idx", SAMPLE, SOURCE, "y_true", f"y_pred_{y}",
                        f"error_{y}", f"abs_error_{y}", f"MED_{y}", f"PED_raw_{y}",
                        f"PED_log1p_{y}", f"quadrant_{y}"] +
                       [f"quadrant_P{p}_{y}" for p in PERCENTILES])
    historical_only = comparison[[c for c in historical_cols if c in comparison.columns]].copy()
    outputs = {
        "temporal_comparison.csv": comparison,
        f"historical_{y}_pooled_oof.csv": historical_only,
        "temporal_thresholds_by_fold.csv": threshold_table,
        "temporal_fold_metrics.csv": metric_table,
        "fold_transition_summary.csv": fold_transition,
        "transition_summary.csv": class_summary,
        "P40_P60_temporal_sensitivity.csv": sensitivity,
        "dimension_transition_summary.csv": axis_transition,
        "transition_driver_summary.csv": drivers,
        "q4_to_q1_candidates.csv": q4_to_q1,
        "into_q1_candidates.csv": into_q1,
        "into_q4_candidates.csv": into_q4,
    }
    for name, data in outputs.items():
        save_csv(out / name, data)
    save_csv(out / "transition_matrix_counts.csv", transition_counts, index=True)
    save_csv(out / "transition_matrix_row_fraction.csv", transition_frac, index=True)
    for f in FOLDS:
        save_csv(out / f"fold{f}_historical_evidence.csv", comparison.loc[comparison.fold.eq(f)])
        save_json(out / "calibrations" / f"fold{f}_historical_med_config.json", details[f-1])
    config = {
        "version": "nested_source_aware_temporal_retro_fixed_theta",
        "year_cutoff": y,
        "n_paired_heldout": len(comparison),
        "n_unique_test_sources": int(comparison[SOURCE].nunique()),
        "config": CFG.to_dict(),
        "model_and_reference_rules": {
            "theta": "Nested theta_i selected on full contemporary outer train, fixed in historical/current pair",
            "historical_MED": "historical model leaf space; own historical training LOO normalization/P40-P60 thresholds",
            "current_MED": "frozen current model leaf space; own current training LOO normalization/P40-P60 thresholds",
            "ped": "same current-fold main/protocol weights and mean/std, distinct historical subset vs current superset",
            "ped_thresholds": "same CURRENT fold training LOO P40-P60 for both periods",
            "interpretation": "relative evidence state transitions, not a common absolute MED metric or causal literature gain",
        },
        "historical_pooled_metrics": hist_perf,
        "current_pooled_metrics": curr_perf,
        "historical_P50_state_counts": comparison[qh].value_counts().reindex(STATES, fill_value=0).to_dict(),
        "current_P50_state_counts": comparison[qc].value_counts().reindex(STATES, fill_value=0).to_dict(),
        "n_states_changed": changed,
        "fraction_states_changed": changed / len(comparison),
        "PED_monotonicity_violations": 0,
        "checks": "all test IDs, theta, saved predictions, current PED and P40-P60 Q passed",
        "paths": {k: str(v) for k, v in paths.items()},
    }
    save_json(out / "run_config.json", config)
    if not args.no_excel:
        try:
            export_excel(out / "nested_temporal_evidence.xlsx", {
                "Summary": pd.DataFrame([{"n": len(comparison), "changed": changed,
                                          "changed_fraction": changed/len(comparison),
                                          **{f"historical_{k}":v for k,v in hist_perf.items()},
                                          **{f"current_{k}":v for k,v in curr_perf.items()}}]),
                "Transition_Counts": transition_counts.reset_index(),
                "Thresholds": threshold_table,
                "Sensitivity": sensitivity,
                "Dimension_Changes": axis_transition,
                "Transition_Drivers": drivers,
                "Fold_Summary": fold_transition,
                "Paired_Records": comparison,
            })
            print("Excel saved:", out / "nested_temporal_evidence.xlsx")
        except ImportError:
            print("openpyxl not found; CSV output completed, Excel skipped")
    print("ALL CHECKS PASSED. Output:", out)
    print("Primary comparison:", out / "temporal_comparison.csv", flush=True)


if __name__ == "__main__":
    main()
