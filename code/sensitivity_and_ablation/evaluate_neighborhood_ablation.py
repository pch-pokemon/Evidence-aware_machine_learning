# -*- coding: utf-8 -*-
"""Neighborhood-size sensitivity and applicability-score ablation.

Run from the oxidation-project root (where evidence_core.py and splits/ exist):
    python evaluate_neighborhood_ablation.py
    # Jupyter: !python evaluate_neighborhood_ablation.py

This script DOES NOT fit or tune XGBoost. It loads frozen nested outer-fold models.
All normalization and evidence thresholds come from the matching OUTER TRAIN fold.

Required (default relative paths):
  pooled_source_oof_evidence.csv
      OR models/nested_evidence_final/pooled_source_oof_evidence.csv
  splits/source_kfold/fold{i}_train.csv, fold{i}_test.csv
  models/xgb_source_nested_kfold/nested_outer_fold{i}.joblib
  models/xgb_source_nested_kfold/nested_outer_fold{i}_meta.json
  models/fixed_evidence_weights/final_fixed_evidence_weights.json
  models/nested_evidence_final/weights/protocol_fold_weights.csv
  evidence_core.py

Outputs: models/neighborhood_ablation/*.csv, summary.txt, analysis_protocol.json,
         risk_coverage.png (when matplotlib is installed).

Methods:
  1. Vary k=5,10,15,20 separately for MED, PED, and both. Recompute the
     per-fold training LOO normalization/median and held-out evidence states.
  2. Baseline controls at k=10: MED-only, PED-only, standardized Euclidean
     nearest-neighbor distance over original 32 input features, and joint
     max(MED-reference-percentile, PED-reference-percentile).
  3. Rank held-out samples WITHOUT their errors, compare MAE at identical
     retained counts. Also show P50 single-distance AD groups and Q1--Q4.
  4. Source-cluster bootstrap intervals condition on frozen predictions,
     selected sets, and source identities. They are descriptive, NOT an
     independent test or uncertainty about the XGBoost fitting procedure.

The joint MAX-percentile score is used ONLY as an auxiliary fair-ranking
comparison; it does NOT replace the official Q1--Q4 assignment.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from evidence_core import (
    EvidenceMetricConfig,
    MAIN_PHYSICAL_FEATURES,
    PROTOCOL_NUM_FEATURES,
    assign_quadrants,
    build_physical_arrays,
    load_main_weights,
    physical_distance_components_for_query,
)

FOLDS = (1, 2, 3, 4, 5)
K_GRID = (5, 10, 15, 20)
K0 = 10
STATES = ("Q1", "Q2", "Q3", "Q4")
BASE_CFG = EvidenceMetricConfig(
    topk=10, lambda_protocol=0.15,
    ht_route_mismatch_penalty=1.0,
    ped_threshold_scale="log1p",
    clip_training_reference_normalization=True,
)
SCENARIOS = ("MED_k_only", "PED_k_only", "both_k")
METHODS = ("MED_only", "PED_only", "Euclidean_kNN", "Joint_MED_PED")


def get_args():
    p = argparse.ArgumentParser(description="Nested k sensitivity / evidence ablation; frozen XGBoost")
    p.add_argument("--project-root", default=".", help="Path containing splits/, models/, evidence_core.py")
    p.add_argument("--oof", default="pooled_source_oof_evidence.csv",
                   help="Official frozen pooled OOF CSV; searches nested_evidence_final/ as fallback")
    p.add_argument("--data-dir", default="splits/source_kfold")
    p.add_argument("--model-dir", default="models/xgb_source_nested_kfold")
    p.add_argument("--evidence-dir", default="models/nested_evidence_final")
    p.add_argument("--main-weights-json", default="models/fixed_evidence_weights/final_fixed_evidence_weights.json")
    p.add_argument("--output-dir", default="models/neighborhood_ablation")
    p.add_argument("--bootstrap", type=int, default=1000, help="Source bootstrap draws; 0 to disable")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected-n", type=int, default=613, help="0 disables 613-record check")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    if a.bootstrap < 0 or a.expected_n < 0:
        p.error("--bootstrap and --expected-n must be nonnegative")
    return a


def path(root: Path, name: str) -> Path:
    p = Path(name).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def write_csv(df: pd.DataFrame, name: str, out: Path):
    df.to_csv(out / name, index=False, encoding="utf-8-sig")


def validate_inputs(oof: pd.DataFrame, expected_n: int):
    need = {"fold", "sample_id", "source_key", "y_true", "y_pred", "abs_error",
            "D_LEAF", "D_SUPPORT", "MED", "PED_log1p", "quadrant",
            "MED_threshold_P50_fold", "PED_threshold_P50_fold"}
    missing = need - set(oof)
    if missing:
        raise ValueError("Official OOF missing columns: " + str(sorted(missing)))
    if expected_n and len(oof) != expected_n:
        raise ValueError(f"Expected {expected_n} held-out records; found {len(oof)}")
    if oof.sample_id.isna().any() or oof.sample_id.astype(str).duplicated().any():
        raise ValueError("OOF sample_id is missing or duplicated")
    if set(oof.fold.astype(int)) != set(FOLDS):
        raise ValueError("OOF does not contain folds 1..5")
    if oof.source_key.isna().any() or oof.groupby("source_key").fold.nunique().max() != 1:
        raise ValueError("OOF contains missing sources or the same source in multiple outer folds")
    if not np.allclose(oof.MED, 0.5*(oof.D_LEAF + oof.D_SUPPORT), atol=1e-9):
        raise ValueError("Frozen MED != 0.5*D_LEAF + 0.5*D_SUPPORT")
    if not np.allclose(oof.abs_error, (oof.y_true-oof.y_pred).abs(), atol=1e-8):
        raise ValueError("OOF absolute errors disagree with saved y_true/y_pred")
    med_low = oof.MED.to_numpy(dtype=float) <= oof.MED_threshold_P50_fold.to_numpy(dtype=float)
    ped_low = oof.PED_log1p.to_numpy(dtype=float) <= oof.PED_threshold_P50_fold.to_numpy(dtype=float)
    base_q = np.select([med_low & ped_low, med_low & ~ped_low, ~med_low & ped_low],
                       ["Q1", "Q2", "Q3"], default="Q4")
    if not np.array_equal(base_q, oof.quadrant.astype(str).to_numpy()):
        raise ValueError("Frozen OOF Q states disagree with its P50 thresholds")


def read_protocol_weights(weights: pd.DataFrame, fold: int) -> dict[str, float]:
    w = weights.loc[weights.fold.astype(int).eq(fold), ["feature", "weight"]]
    if len(w) != len(PROTOCOL_NUM_FEATURES) or w.feature.duplicated().any():
        raise RuntimeError(f"fold {fold}: missing/duplicated protocol weights")
    result = dict(zip(w.feature.astype(str), w.weight.astype(float)))
    if set(result) != set(PROTOCOL_NUM_FEATURES) or not np.isclose(sum(result.values()), 1, atol=1e-6):
        raise RuntimeError(f"fold {fold}: protocol weights invalid")
    return result


def med_components_for_all_k(train_leaf: np.ndarray, test_leaf: np.ndarray, ks: tuple[int, ...]):
    """Exact leaf Hamming + occupancy definitions from evidence_core, efficiently cached."""
    tr = np.asarray(train_leaf)
    te = np.asarray(test_leaf)
    if tr.ndim == 1: tr = tr.reshape(-1, 1)
    if te.ndim == 1: te = te.reshape(-1, 1)
    if tr.shape[1] != te.shape[1]:
        raise ValueError("Train/test leaf dimensions mismatch")
    n, n_trees = tr.shape
    if n <= max(ks):
        raise ValueError(f"Training pool size {n} not greater than largest k={max(ks)}")
    # scipy cdist 'hamming' = average number of non-identical terminal leaves
    dist_tr = cdist(tr.astype(float), tr.astype(float), metric="hamming")
    dist_te = cdist(te.astype(float), tr.astype(float), metric="hamming")
    np.fill_diagonal(dist_tr, np.inf)

    occ_train = np.zeros(n, dtype=float)
    occ_test = np.zeros(len(te), dtype=float)
    for t in range(n_trees):
        labels, inv, counts = np.unique(tr[:, t], return_inverse=True, return_counts=True)
        occ_train += counts[inv] - 1
        idx = np.searchsorted(labels, te[:, t])
        good = idx < len(labels)
        found = np.zeros(len(te), dtype=float)
        found[good] = (labels[idx[good]] == te[good, t]) * counts[idx[good]]
        occ_test += found
    occ_train /= n_trees
    occ_test /= n_trees
    tr_log = np.log1p(np.maximum(occ_train, 0.0))
    te_log = np.log1p(np.maximum(occ_test, 0.0))
    lo, hi = float(tr_log.min()), float(tr_log.max())
    if hi - lo <= 1e-12:
        tr_ds = np.ones(n)  # core: 1 - minmax_transform -> 1.0
        te_ds = np.ones(len(te))
    else:
        tr_ds = 1 - np.clip((tr_log - lo)/(hi-lo), 0.0, 1.0)
        te_ds = 1 - np.clip((te_log - lo)/(hi-lo), 0.0, 1.0)
    result = {}
    for k in ks:
        tr_raw = np.partition(dist_tr, k-1, axis=1)[:, :k].mean(axis=1)
        te_raw = np.partition(dist_te, k-1, axis=1)[:, :k].mean(axis=1)
        lo, hi = float(tr_raw.min()), float(tr_raw.max())
        if hi - lo <= 1e-12:
            tr_dl, te_dl = np.zeros(n), np.zeros(len(te))
        else:
            tr_dl = np.clip((tr_raw-lo)/(hi-lo), 0.0, 1.0)
            te_dl = np.clip((te_raw-lo)/(hi-lo), 0.0, 1.0)
        result[k] = {"train": 0.5*(tr_dl+tr_ds), "test": 0.5*(te_dl+te_ds),
                     "train_leaf": tr_dl, "test_leaf": te_dl,
                     "train_support": tr_ds, "test_support": te_ds}
    return result


def physical_ped_for_all_k(train: pd.DataFrame, test: pd.DataFrame,
                           main_weights: dict, protocol_weights: dict,
                           ks: tuple[int, ...]):
    """Reuse evidence_core's standardization, oxidation time, and route-gating."""
    train_arrays = build_physical_arrays(train, train, main_weights, protocol_weights)
    test_arrays = build_physical_arrays(train, test, main_weights, protocol_weights)
    n, m = len(train), len(test)
    d_tr = np.empty((n, n), dtype=np.float64)
    d_te = np.empty((m, n), dtype=np.float64)
    for i in range(n):
        distances, _, _ = physical_distance_components_for_query(train_arrays, i, BASE_CFG)
        d_tr[i] = distances
    for i in range(m):
        distances, _, _ = physical_distance_components_for_query(test_arrays, i, BASE_CFG)
        d_te[i] = distances
    np.fill_diagonal(d_tr, np.inf)
    out = {}
    for k in ks:
        tr_raw = np.partition(d_tr, k-1, axis=1)[:, :k].mean(axis=1)
        te_raw = np.partition(d_te, k-1, axis=1)[:, :k].mean(axis=1)
        out[k] = {"train": np.log1p(tr_raw), "test": np.log1p(te_raw)}
    return out


def vanilla_knn_distance(train: pd.DataFrame, test: pd.DataFrame,
                         features: list[str], k: int):
    """Equal-weight Euclidean 32-feature kNN; train-only imputation and scaling."""
    x_tr = train.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    x_te = test.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    med = x_tr.median().fillna(0.0)
    x_tr = x_tr.fillna(med)
    x_te = x_te.fillna(med)
    mu = x_tr.mean()
    sig = x_tr.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    z_tr = ((x_tr-mu)/sig).to_numpy(dtype=float)
    z_te = ((x_te-mu)/sig).to_numpy(dtype=float)
    d_tr = cdist(z_tr, z_tr, metric="euclidean")
    d_te = cdist(z_te, z_tr, metric="euclidean")
    np.fill_diagonal(d_tr, np.inf)
    tr_score = np.partition(d_tr, k-1, axis=1)[:, :k].mean(axis=1)
    te_score = np.partition(d_te, k-1, axis=1)[:, :k].mean(axis=1)
    return tr_score, te_score


def train_percentile_rank(train_ref: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(train_ref, dtype=float))
    q = np.asarray(query_scores, dtype=float)
    if len(ref) == 0 or not np.isfinite(ref).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite/empty training-reference distance for percentile transform")
    return np.searchsorted(ref, q, side="right") / len(ref)


def ci_q4_minus_q1(data: pd.DataFrame, q_name: str, draws: np.ndarray | None):
    if draws is None: return np.nan, np.nan
    source = data["_source_idx"].to_numpy(dtype=int)
    ae = data["abs_error"].to_numpy(dtype=float)
    q = data[q_name].astype(str).to_numpy()
    n_sources = draws.shape[1]
    mask1 = (q == "Q1")
    mask4 = (q == "Q4")
    num1 = np.bincount(source, weights=ae*mask1, minlength=n_sources)
    den1 = np.bincount(source, weights=mask1.astype(float), minlength=n_sources)
    num4 = np.bincount(source, weights=ae*mask4, minlength=n_sources)
    den4 = np.bincount(source, weights=mask4.astype(float), minlength=n_sources)
    cnt1, cnt4 = draws @ den1, draws @ den4
    good = (cnt1 > 0) & (cnt4 > 0)
    if good.sum() < 20: return np.nan, np.nan
    dif = (draws[good] @ num4)/cnt4[good] - (draws[good] @ num1)/cnt1[good]
    return tuple(map(float, np.percentile(dif, [2.5, 97.5])))


def k_sensitivity_table(all_details: pd.DataFrame, frozen: pd.DataFrame,
                        draws: np.ndarray | None):
    rows = []
    for (scenario, k), sub in all_details.groupby(["scenario", "k"], sort=False):
        sub = sub.set_index("sample_id").loc[frozen.sample_id.astype(str)].reset_index()
        q = sub["state"].to_numpy(dtype=str)
        counts = {s: int(np.sum(q == s)) for s in STATES}
        mae1 = float(sub.loc[q == "Q1", "abs_error"].mean()) if counts["Q1"] else np.nan
        mae4 = float(sub.loc[q == "Q4", "abs_error"].mean()) if counts["Q4"] else np.nan
        ci_lo, ci_hi = ci_q4_minus_q1(sub, "state", draws)
        rows.append({"scenario": scenario, "k": int(k),
                     **{f"{s}_count": counts[s] for s in STATES},
                     "state_agreement_pct": 100 * np.mean(q == frozen.quadrant.to_numpy(dtype=str)),
                     "Q1_MAE": mae1, "Q4_MAE": mae4,
                     "Q4_minus_Q1_MAE": mae4 - mae1,
                     "delta_CI95_low": ci_lo, "delta_CI95_high": ci_hi})
    return pd.DataFrame(rows).sort_values(["scenario", "k"]).reset_index(drop=True)


def state_summary(frozen: pd.DataFrame, scores: pd.DataFrame):
    ae = frozen.abs_error.to_numpy(dtype=float)
    state = frozen.quadrant.to_numpy(dtype=str)
    rows = []
    for q in STATES:
        sel = state == q
        rows.append({"method": "Joint_MED_PED", "group": q, "N": int(sel.sum()),
                     "fraction_pct": 100 * sel.mean(), "MAE": float(ae[sel].mean())})
    for method in ("MED_only", "PED_only", "Euclidean_kNN"):
        low = scores[method + "_lowP50"].to_numpy(dtype=bool)
        for name, sel in (("P50_supported",low),("P50_outside",~low)):
            rows.append({"method": method, "group": name, "N": int(sel.sum()),
                         "fraction_pct": 100*sel.mean(),
                         "MAE": float(ae[sel].mean()) if sel.any() else np.nan})
    return pd.DataFrame(rows)


def ranking_table(frozen: pd.DataFrame, scores: pd.DataFrame,
                  rates: tuple[float, ...]):
    ae = frozen.abs_error.to_numpy(dtype=float)
    q1 = frozen.quadrant.astype(str).eq("Q1").to_numpy()
    ids = frozen.sample_id.astype(str).to_numpy()
    n = len(frozen)
    rows = []
    selects = {}
    for method in METHODS:
        vals = scores[method].to_numpy(dtype=float)
        order = np.lexsort((ids, vals))  # independent of abs_error, reproducible tie handling
        for frac in rates:
            n_keep = min(n, max(1, int(round(n*frac))))
            key = (method, n_keep)
            if key in selects: continue
            select = np.zeros(n, dtype=bool)
            select[order[:n_keep]] = True
            selects[key] = select
            rows.append({"method": method, "n_retained": n_keep,
                         "retained_pct": 100*n_keep/n,
                         "MAE": float(ae[select].mean()),
                         "RMSE": float(np.sqrt(np.mean((frozen.y_pred.to_numpy(dtype=float)[select]
                                                        - frozen.y_true.to_numpy(dtype=float)[select])**2))),
                         "N_official_Q1_in_retained": int((select & q1).sum()),
                         "overlap_with_Q1_pct": 100 * (select & q1).sum() / max(n_keep,1)})
    return pd.DataFrame(rows).sort_values(["n_retained", "method"]).reset_index(drop=True), selects


def paired_bootstrap_at_n(frozen: pd.DataFrame, selections: dict, n_keep: int,
                          draws: np.ndarray | None):
    if draws is None: return pd.DataFrame()
    source_idx = frozen._source_idx.to_numpy(dtype=int)
    ae = frozen.abs_error.to_numpy(dtype=float)
    ns = draws.shape[1]
    base = selections[("Joint_MED_PED", n_keep)]
    def drawn_mae(mask):
        num = np.bincount(source_idx, weights=ae*mask, minlength=ns)
        den = np.bincount(source_idx, weights=mask.astype(float), minlength=ns)
        n = draws @ num
        d = draws @ den
        return np.divide(n,d,out=np.full_like(n, np.nan, dtype=float),where=d>0)
    joint_mae = drawn_mae(base)
    rows = []
    for method in METHODS:
        if method == "Joint_MED_PED": continue
        sel = selections[(method, n_keep)]
        diff = drawn_mae(sel) - joint_mae  # positive means joint MAE lower
        valid = diff[np.isfinite(diff)]
        ci = np.percentile(valid, [2.5,97.5]) if len(valid) >= 20 else [np.nan,np.nan]
        rows.append({"comparison": method + " minus Joint_MED_PED",
                     "n_retained": n_keep, "difference_in_MAE": float(ae[sel].mean()-ae[base].mean()),
                     "bootstrap_CI95_low": float(ci[0]), "bootstrap_CI95_high": float(ci[1]),
                     "valid_draws": len(valid),
                     "interpretation": "Positive = lower MAE for joint; conditional bootstrap"})
    return pd.DataFrame(rows)


def error_ranking_diagnostics(frozen: pd.DataFrame, scores: pd.DataFrame):
    ae = frozen.abs_error.to_numpy(dtype=float)
    high = ae >= np.quantile(ae, 0.75)
    rows=[]
    for method in METHODS:
        x = scores[method].to_numpy(dtype=float)
        rho, p = spearmanr(x,ae)
        auc = roc_auc_score(high, x) if high.any() and (~high).any() else np.nan
        rows.append({"method":method,"spearman_rho_distance_vs_abs_error":rho,
                     "spearman_p_nominal_nonclustered": p,
                     "AUROC_for_top_quartile_abs_error": auc})
    return pd.DataFrame(rows)


def create_plot(risk: pd.DataFrame, output: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Plot skipped: {exc}")
        return
    fig, ax = plt.subplots(figsize=(7.5,4.8))
    for method in METHODS:
        x = risk[risk.method == method].sort_values("retained_pct")
        ax.plot(x.retained_pct, x.MAE, marker="o", linewidth=1.5, markersize=3.5,
                label=method.replace("_", " "))
    ax.set_xlabel("Retained held-out predictions (%)")
    ax.set_ylabel("MAE of retained predictions (lgkp)")
    ax.set_title("Risk–coverage comparison (fixed nested OOF predictions)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=260)
    plt.close(fig)


def main():
    args = get_args()
    root = Path(args.project_root).expanduser().resolve()
    oof_path = path(root,args.oof)
    if not oof_path.is_file() and args.oof == "pooled_source_oof_evidence.csv":
        oof_path = path(root,"models/nested_evidence_final/pooled_source_oof_evidence.csv")
    if not oof_path.is_file():
        raise FileNotFoundError(f"Missing official pooled OOF: {oof_path}")
    data_dir = path(root,args.data_dir)
    model_dir = path(root,args.model_dir)
    ev_dir = path(root,args.evidence_dir)
    weight_path = path(root,args.main_weights_json)
    out = path(root,args.output_dir)
    input_paths = [oof_path, data_dir, model_dir, ev_dir, weight_path]
    if out in input_paths or any(out == p or out in p.parents for p in input_paths):
        raise ValueError("Output directory must not overwrite an input file/folder")
    out.mkdir(parents=True, exist_ok=True)

    frozen = pd.read_csv(oof_path, low_memory=False)
    validate_inputs(frozen,args.expected_n)
    frozen.sample_id = frozen.sample_id.astype(str)
    frozen.source_key = frozen.source_key.astype(str)
    main_w = load_main_weights(weight_path)
    weights_path = ev_dir / "weights" / "protocol_fold_weights.csv"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Fold-specific protocol weight file not found: {weights_path}")
    weights = pd.read_csv(weights_path)
    source_names, source_idx = np.unique(frozen.source_key.to_numpy(), return_inverse=True)
    frozen["_source_idx"] = source_idx
    rng = np.random.default_rng(args.seed)
    draws = rng.multinomial(len(source_names), [1.0/len(source_names)]*len(source_names),
                            size=args.bootstrap) if args.bootstrap else None

    oof_scenarios, oof_scores, audit_rows, run_notes = [], [], [], []
    print("="*94)
    print("NESTED NEIGHBORHOOD SENSITIVITY AND APPLICABILITY ABLATION")
    print(f"OOF={oof_path} | N={len(frozen)} | sources={len(source_names)}")
    print(f"k grid={K_GRID}, baseline={K0}; bootstrap={args.bootstrap} source draws")
    print("XGBoost outer models are FROZEN. Every reference threshold is TRAIN-only.")
    print("="*94,flush=True)

    for fold in FOLDS:
        tr_path = data_dir / f"fold{fold}_train.csv"
        te_path = data_dir / f"fold{fold}_test.csv"
        mp = model_dir / f"nested_outer_fold{fold}.joblib"
        meta_path = model_dir / f"nested_outer_fold{fold}_meta.json"
        for p in (tr_path,te_path,mp,meta_path):
            if not p.is_file():
                raise FileNotFoundError("Missing fold file: " + str(p))
        tr = pd.read_csv(tr_path, low_memory=False)
        te = pd.read_csv(te_path, low_memory=False)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feats = list(meta["feature_cols"])
        if len(tr) <= max(K_GRID):
            raise ValueError(f"Fold {fold}: train size too small for k={max(K_GRID)}")
        if (tr.sample_id.isna().any() or te.sample_id.isna().any()
            or tr.sample_id.astype(str).duplicated().any() or te.sample_id.astype(str).duplicated().any()):
            raise RuntimeError(f"Fold {fold}: missing or duplicate sample ids")
        if set(tr.source_key.astype(str)) & set(te.source_key.astype(str)):
            raise RuntimeError(f"Fold {fold}: held-out source leaked into training")
        official = frozen.loc[frozen.fold.astype(int).eq(fold)].copy()
        if (len(official) != len(te) or
                official.sample_id.tolist() != te.sample_id.astype(str).tolist()):
            raise RuntimeError(f"Fold {fold}: official pooled OOF not aligned with split test")
        if not np.allclose(official.y_true.to_numpy(dtype=float),te.lgkp.to_numpy(dtype=float),atol=1e-9):
            raise RuntimeError(f"Fold {fold}: target mismatch against frozen held-out set")
        model = joblib.load(mp)
        if model.get_booster().feature_names is not None and list(model.get_booster().feature_names)!=feats:
            raise RuntimeError(f"Fold {fold}: XGBoost feature order differs from metadata")
        X_tr = tr.loc[:,feats].astype(float)
        X_te = te.loc[:,feats].astype(float)
        pred_check = model.predict(X_te)
        max_pred_diff = float(np.max(np.abs(pred_check-official.y_pred.to_numpy(dtype=float))))
        if max_pred_diff > 2e-5:
            raise RuntimeError(f"Fold {fold}: frozen XGB prediction mismatch {max_pred_diff:.6g}")
        print(f"Fold {fold}: train={len(tr)}, held-out={len(te)}; calculating pairwise distances...",flush=True)
        meds = med_components_for_all_k(model.apply(X_tr),model.apply(X_te),K_GRID)
        peds = physical_ped_for_all_k(tr,te,main_w,read_protocol_weights(weights,fold),K_GRID)
        knn_tr, knn_te = vanilla_knn_distance(tr,te,feats,K0)
        ref_med, te_med = meds[K0]["train"], meds[K0]["test"]
        ref_ped, te_ped = peds[K0]["train"], peds[K0]["test"]
        if not np.allclose(te_med,official.MED.to_numpy(dtype=float),atol=2e-8):
            raise RuntimeError(f"Fold {fold}: MED k=10 failed frozen-reproduction; maxdiff="
                               f"{np.max(np.abs(te_med-official.MED.to_numpy(dtype=float))):.4g}")
        if not np.allclose(te_ped,official.PED_log1p.to_numpy(dtype=float),atol=2e-8):
            raise RuntimeError(f"Fold {fold}: PED k=10 failed frozen-reproduction; maxdiff="
                               f"{np.max(np.abs(te_ped-official.PED_log1p.to_numpy(dtype=float))):.4g}")
        thr_m0 = float(np.percentile(ref_med,50))
        thr_p0 = float(np.percentile(ref_ped,50))
        if (not np.isclose(thr_m0,official.MED_threshold_P50_fold.iloc[0],atol=2e-8)
            or not np.isclose(thr_p0,official.PED_threshold_P50_fold.iloc[0],atol=2e-8)):
            raise RuntimeError(f"Fold {fold}: k=10 training thresholds differ from frozen model")
        q_check = assign_quadrants(te_med,te_ped,thr_m0,thr_p0)
        if not np.array_equal(q_check,official.quadrant.to_numpy(dtype=str)):
            raise RuntimeError(f"Fold {fold}: k=10 Q states differ from frozen model")

        fold_scores = official.loc[:,["sample_id","fold","source_key","abs_error"]].copy()
        fold_scores["MED_only"] = train_percentile_rank(ref_med,te_med)
        fold_scores["PED_only"] = train_percentile_rank(ref_ped,te_ped)
        fold_scores["Euclidean_kNN"] = train_percentile_rank(knn_tr,knn_te)
        fold_scores["Joint_MED_PED"] = np.maximum(fold_scores.MED_only,fold_scores.PED_only)
        fold_scores["MED_only_lowP50"] = te_med <= thr_m0
        fold_scores["PED_only_lowP50"] = te_ped <= thr_p0
        fold_scores["Euclidean_kNN_lowP50"] = knn_te <= np.percentile(knn_tr,50)
        oof_scores.append(fold_scores)

        for scenario in SCENARIOS:
            for k in K_GRID:
                km = k if scenario in ("MED_k_only","both_k") else K0
                kp = k if scenario in ("PED_k_only","both_k") else K0
                tr_m, te_m = meds[km]["train"],meds[km]["test"]
                tr_p, te_p = peds[kp]["train"],peds[kp]["test"]
                tm = float(np.percentile(tr_m,50))
                tp = float(np.percentile(tr_p,50))
                q = assign_quadrants(te_m,te_p,tm,tp)
                if k==K0 and not np.array_equal(q,official.quadrant.astype(str).to_numpy()):
                    raise RuntimeError(f"Fold {fold}: scenario {scenario} at k10 does not match baseline")
                row = official.loc[:,["sample_id","source_key","fold","abs_error"]].copy()
                row["scenario"],row["k"] = scenario,k
                row["MED_variant"],row["PED_variant"] = te_m,te_p
                row["MED_P50_training"],row["PED_P50_training"] = tm,tp
                row["state"] = q
                row["_source_idx"] = official._source_idx.to_numpy(dtype=int)
                oof_scenarios.append(row)
                audit_rows.append({"fold":fold,"scenario":scenario,"k":k,
                                   "MED_threshold":tm,"PED_threshold":tp,"N_test":len(te)})
        run_notes.append({"fold":fold,"max_abs_frozen_prediction_diff":max_pred_diff,
                          "MED_k10_max_abs_score_diff":float(np.max(np.abs(te_med-official.MED))),
                          "PED_k10_max_abs_score_diff":float(np.max(np.abs(te_ped-official.PED_log1p))),
                          "reproduced_official_Q_states":True})
        print(f"  Fold {fold} PASS: k10 MED/PED/P50/Q states reproduce frozen results.",flush=True)

    all_k = pd.concat(oof_scenarios,ignore_index=True)
    scores = pd.concat(oof_scores,ignore_index=True)
    scores = scores.set_index("sample_id").loc[frozen.sample_id].reset_index()
    if len(scores)!=len(frozen):
        raise RuntimeError("Scores do not cover official OOF")
    k_summary = k_sensitivity_table(all_k,frozen,draws)
    n_q1 = int(frozen.quadrant.eq("Q1").sum())
    n = len(frozen)
    rates = tuple(sorted(set((0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0,n_q1/n))))
    risk, selections = ranking_table(frozen,scores,rates)
    p50 = state_summary(frozen,scores)
    diag = error_ranking_diagnostics(frozen,scores)
    paired = paired_bootstrap_at_n(frozen,selections,n_q1,draws)
    q_discordant = (frozen.quadrant.astype(str).isin(["Q2","Q3"])).sum()
    audit = pd.DataFrame(run_notes)
    output_protocol = {
        "input":str(oof_path),"n_predictions":n,"n_sources":len(source_names),
        "model_status":"pretrained nested source-aware outer models; no refitting",
        "k_grid":K_GRID,"baseline_k":K0,"baseline_lambda":BASE_CFG.lambda_protocol,
        "med_weight":"0.5 D_LEAF + 0.5 D_SUPPORT",
        "reference":"per-outer-fold training LOO references, with fitted models (NOT cross-fitted reference MED)",
        "one_factor_at_a_time_scenarios":SCENARIOS,
        "conventional_baseline":"Mean Euclidean distance to k=10 outer-training neighbors over equally weighted standardized 32 XGBoost input columns; training-only median imputation",
        "ranking":"Training LOO empirical CDF percentile; joint_score=max(MED_percentile,PED_percentile)",
        "official_Q_states_unchanged":True,
        "bootstrap":"Source-cluster resampling conditional on frozen models, errors, selections and states; no model retraining",
        "bootstrap_draws":args.bootstrap,"bootstrap_seed":args.seed,
        "evaluation_scope":"descriptive retrospective OOF comparison; not optimized by held-out errors",
        "note":"Joint ranking is exploratory, not a change in the manuscript's Q1-Q4 definition"
    }
    write_csv(k_summary,"k_sensitivity_summary.csv",out)
    write_csv(pd.DataFrame(audit_rows),"k_fold_threshold_audit.csv",out)
    write_csv(all_k.drop(columns=["_source_idx"]),"k_scenario_oof_states.csv",out)
    write_csv(scores,"ablation_oof_scores.csv",out)
    write_csv(p50,"ablation_P50_and_Q_states.csv",out)
    write_csv(risk,"ablation_risk_coverage.csv",out)
    write_csv(paired,"ablation_matched_Q1count_bootstrap.csv",out)
    write_csv(diag,"ablation_error_ranking.csv",out)
    write_csv(audit,"baseline_reproduction_audit.csv",out)
    (out/"analysis_protocol.json").write_text(json.dumps(output_protocol,indent=2,ensure_ascii=False),encoding="utf-8")
    if not args.no_plot: create_plot(risk,out/"risk_coverage.png")
    # human-readable log intentionally contains everything needed for interpretation of the analysis
    columns=["scenario","k","Q1_count","Q2_count","Q3_count","Q4_count",
             "state_agreement_pct","Q1_MAE","Q4_MAE","Q4_minus_Q1_MAE",
             "delta_CI95_low","delta_CI95_high"]
    risk_main = risk.loc[risk.n_retained==n_q1, ["method","n_retained","MAE","RMSE",
                                                 "N_official_Q1_in_retained"]]
    lines=["NEIGHBORHOOD SENSITIVITY AND ABLATION",
           f"Frozen source-aware nested OOF: N={n}, sources={len(source_names)}; Q1={n_q1}; Q2+Q3={q_discordant}",
           "\n1) NEAREST-NEIGHBOR k SENSITIVITY (source-cluster CI conditional on frozen OOF)",
           k_summary[columns].to_string(index=False,float_format=lambda v:f"{v:.4f}"),
           "\n2) SINGLE-DISTANCE P50 AD vs JOINT Q1-Q4 (descriptive)",
           p50.to_string(index=False,float_format=lambda v:f"{v:.4f}"),
           f"\n3) MATCHED RETAINED COUNT = OFFICIAL Q1 COUNT ({n_q1}/{n})",
           risk_main.to_string(index=False,float_format=lambda v:f"{v:.4f}"),
           "\n4) SOURCE-CLUSTER BOOTSTRAP: other method MAE minus joint MAE (fixed selections)",
           paired.to_string(index=False,float_format=lambda v:f"{v:.4f}") if len(paired) else "Bootstrap disabled (--bootstrap 0)",
           "\n5) ERROR RANKING DIAGNOSTICS (not model predictive accuracy improvements)",
           diag.to_string(index=False,float_format=lambda v:f"{v:.4f}"),
           "\n6) OFFICIAL BASELINE REPRODUCTION CHECKS", audit.to_string(index=False),
           "\nNOTES: A low MAE difference does not imply improved XGBoost predictions; scores do not change y_pred.",
           "The joint max-percentile ranking is a retrospective comparison only; official four-state assignment is unchanged.",
           "Intervals condition on fixed models/samples; no uncertainty from retraining or model selection."]
    text="\n".join(lines)+"\n"
    (out/"summary.txt").write_text(text,encoding="utf-8")
    print("\n"+text)
    print("Outputs:",out,flush=True)


if __name__=="__main__":
    main()
