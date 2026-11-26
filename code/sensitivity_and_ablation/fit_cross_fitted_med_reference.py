# -*- coding: utf-8 -*-
"""fit_cross_fitted_med_reference.py

Source-disjoint fits for MED calibration sensitivity.

For each of the five SAVED source-aware Nested-CV outer models:
  1. Audit outer training/test source separation and reproduce held-out predictions.
  2. Compare fitted-model training-reference LOO MED with truly external MED on
     its OUTER held-out, source-disjoint records (observational diagnostic ONLY).
  3. Reuse that outer model's FIXED, already-selected hyperparameters to fit four
     auxiliary models on 3/4 of its OUTER TRAIN sources; assess the remaining 1/4
     sources as external, source-held-out MED queries.
  4. For every auxiliary model, compare LOO MED vs external MED on the SAME model,
     with the SAME reference-derived normalization, and compute P40-P60 exceedance.
  5. Optionally repeat step 3 with ordinary random inner KFold as a control to
     investigate which differences may be associated with literature-source shift.

This script intentionally DOES NOT:
  * perform Optuna or change model hyperparameters;
  * overwrite or train over the saved official Nested outer models;
  * combine raw MED scores across different tree models to set a deployment cutoff;
  * use outer held-out data to choose a final calibration rule;
  * assert that an external >P50 frequency must equal 50%.

CAVEAT: Auxiliary models reuse theta_i chosen on the COMPLETE OUTER training data.
The source-held-out auxiliary samples did not participate in fitting their own
auxiliary TREES, but they DID participate indirectly in choosing theta_i through
Nested CV. This is a model-fitting reference-gap DIAGNOSTIC, not a fully selection-
independent double-nested calibration procedure. External/LOO differences may also
reflect legitimate source-domain shift and auxiliary-vs-full training-set size.

Run from your project root (beside evidence_core.py):
    python fit_cross_fitted_med_reference.py
or:
    %run fit_cross_fitted_med_reference.py

Optional:
    python fit_cross_fitted_med_reference.py --include-random-control
    # Resume AFTER previous GBK console crash without re-fitting 20 source models:
    python fit_cross_fitted_med_reference.py --reuse-source-results
    # Optional recommended next control (fits only 20 random models):
    python fit_cross_fitted_med_reference.py --reuse-source-results --include-random-control
    python fit_cross_fitted_med_reference.py --n-jobs 4
    python fit_cross_fitted_med_reference.py --project-root "C:/..."

Read-only inputs:
    splits/source_kfold/fold{i}_{train,test}.csv
    models/xgb_source_nested_kfold/nested_outer_fold{i}.joblib
    models/xgb_source_nested_kfold/nested_outer_fold{i}_meta.json
    models/xgb_source_nested_kfold/nested_oof_predictions.csv (if present)
    evidence_core.py

Output ONLY:
    models/nested_med_crossfit_diagnostic/
        run_config.json
        per_inner_summary.csv
        source_crossfit_heldout_MED.csv
        outer_reference_vs_test_summary.csv
        outer_heldout_MED.csv
        per_outer_crossfit_summary.csv
        P40_P60_exceedance.csv
        global_summary.json
        random_control_* (ONLY if requested)

No MED calibration is changed by this diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from evidence_core import (
    EvidenceMetricConfig,
    build_med_from_training_reference,
    ensure_2d_leaf_matrix,
)

OUTER_FOLDS = (1, 2, 3, 4, 5)
PERCENTILES = (40, 45, 50, 55, 60)
CONF = EvidenceMetricConfig(
    topk=10,
    lambda_protocol=0.15,
    ht_route_mismatch_penalty=1.0,
    ped_threshold_scale="log1p",
    clip_training_reference_normalization=True,
)


def args_parse():
    p = argparse.ArgumentParser(description="Nested source-aware MED in-sample LOO vs external cross-fit diagnostic")
    p.add_argument("--project-root", default=None, help="Default: directory of this script")
    p.add_argument("--data-dir", default="splits/source_kfold")
    p.add_argument("--nested-model-dir", default="models/xgb_source_nested_kfold")
    p.add_argument("--output-dir", default="models/nested_med_crossfit_diagnostic")
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--include-random-control", action="store_true",
                   help="Also fit 4 random KFold auxiliaries per outer fold (20 more fits)")
    p.add_argument("--skip-outer-audit", action="store_true",
                   help="Skip full-model outer test MED comparison (not recommended)")
    p.add_argument("--reuse-source-results", action="store_true",
                   help="Reuse previously saved source crossfit CSV files; only audit saved outer models and optionally fit random controls")
    a = p.parse_args()
    if a.inner_folds < 2 or a.n_jobs < 1:
        p.error("--inner-folds >= 2 and --n-jobs >= 1 are required")
    return a


def located(root: Path, p: str) -> Path:
    path = Path(p).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def num(x: Any) -> float:
    return float(np.asarray(x, dtype=float))


def median(x: Any) -> float:
    return float(np.median(np.asarray(x, dtype=float)))


def pct(x: Any, q: int | float) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), q))


def frac(cond: Any) -> float:
    return float(np.mean(np.asarray(cond, dtype=bool)))


def summarize_predictions(y_true, y_pred) -> dict[str, float]:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"R2": float(r2_score(y, p)), "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}


def check_frame(df: pd.DataFrame, role: str, features: list[str]) -> None:
    necessary = set(features + ["sample_id", "source_key", "lgkp"])
    missing = sorted(necessary.difference(df.columns))
    if missing:
        raise RuntimeError(f"{role}: missing columns: {missing}")
    if df.sample_id.isna().any() or df.sample_id.duplicated().any():
        raise RuntimeError(f"{role}: null/duplicated sample_id")
    if df.source_key.isna().any():
        raise RuntimeError(f"{role}: null source_key")
    if df[features].isnull().any().any() or not np.isfinite(df[features].to_numpy(float)).all():
        raise RuntimeError(f"{role}: invalid model input features")
    if not np.isfinite(df.lgkp.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{role}: nonfinite lgkp")


def med_from_fitted_model(model, train_x: np.ndarray, query_x: np.ndarray):
    """Reference and query always use ONE fitted model and its OWN train-only scaling."""
    tr_leaves = ensure_2d_leaf_matrix(model.apply(train_x))
    qu_leaves = ensure_2d_leaf_matrix(model.apply(query_x))
    if tr_leaves.shape[0] != len(train_x) or qu_leaves.shape[0] != len(query_x):
        raise RuntimeError("Leaf rows do not match input rows")
    ref, held, _, _, norm = build_med_from_training_reference(tr_leaves, qu_leaves, CONF)
    if not np.isfinite(ref.select_dtypes('number').to_numpy(float)).all():
        raise RuntimeError("Nonfinite training MED")
    if not np.isfinite(held.select_dtypes('number').to_numpy(float)).all():
        raise RuntimeError("Nonfinite external MED")
    return ref, held, norm


def output_heldout_rows(
    frame: pd.DataFrame, metrics: pd.DataFrame, ref: pd.DataFrame,
    norm: Mapping[str, Any], scheme: str, outer_fold: int, inner_fold: int,
    model_role: str, prediction: np.ndarray | None = None,
) -> pd.DataFrame:
    """Store the held-out values and TRAIN-reference quantile labels, never hidden test labels in calibration."""
    use_cols = [c for c in ("sample_id", "source_key", "Year", "Ref_source_num") if c in frame]
    out = frame[use_cols].copy().reset_index(drop=True)
    out.insert(0, "outer_fold", outer_fold)
    out.insert(1, "inner_fold", inner_fold)
    out.insert(2, "scheme", scheme)
    out.insert(3, "model_role", model_role)
    out["y_true"] = frame.lgkp.to_numpy(float)
    if prediction is not None:
        out["y_pred"] = np.asarray(prediction, float)
        out["abs_error"] = np.abs(out.y_pred.to_numpy() - out.y_true.to_numpy())
    for col in ("d_leaf_raw", "support_raw", "D_LEAF", "D_SUPPORT", "MED"):
        out[col] = metrics[col].to_numpy(float)
    ref_med = np.asarray(ref.MED, float)
    ref_sorted = np.sort(ref_med)
    out["training_ref_MED_P50"] = pct(ref_med, 50)
    out["MED_minus_train_P50"] = out.MED - out.training_ref_MED_P50
    out["MED_percentile_vs_train_LOO"] = 100.0 * np.searchsorted(
        ref_sorted, out.MED.to_numpy(float), side="right") / len(ref_sorted)
    out["D_LEAF_raw_above_reference_max"] = (
        out.d_leaf_raw.to_numpy(float) > float(norm["leaf_raw_max"]) + 1e-12)
    out["support_log_below_reference_min"] = (
        np.log1p(np.maximum(out.support_raw.to_numpy(float), 0)) < float(norm["support_log1p_min"]) - 1e-12)
    return out


def summarize_reference_gap(ref: pd.DataFrame, held: pd.DataFrame,
                            norm: Mapping[str, Any]) -> dict[str, Any]:
    """Same-model paired *distributions*, not paired training/held-out records."""
    result = {}
    for name in ("MED", "D_LEAF", "D_SUPPORT", "d_leaf_raw", "support_raw"):
        train_arr = ref[name].to_numpy(float)
        held_arr = held[name].to_numpy(float)
        result[f"train_{name}_median"] = median(train_arr)
        result[f"heldout_{name}_median"] = median(held_arr)
        result[f"delta_median_{name}"] = median(held_arr) - median(train_arr)
        result[f"train_{name}_mean"] = float(np.mean(train_arr))
        result[f"heldout_{name}_mean"] = float(np.mean(held_arr))
    for q in PERCENTILES:
        cutoff = pct(ref.MED, q)
        result[f"train_MED_P{q}"] = cutoff
        result[f"heldout_above_train_P{q}_fraction"] = frac(held.MED.to_numpy(float) > cutoff)
    # The reference itself is LOO; the query is genuinely excluded from tree fitting.
    result["heldout_MED_P50"] = pct(held.MED, 50)
    result["heldout_MED_P50_minus_train_MED_P50"] = (
        result["heldout_MED_P50"] - result["train_MED_P50"])
    result["heldout_D_LEAF_above_ref_max_fraction"] = frac(
        held.d_leaf_raw.to_numpy(float) > float(norm["leaf_raw_max"]) + 1e-12)
    result["heldout_support_below_ref_min_fraction"] = frac(
        np.log1p(np.maximum(held.support_raw.to_numpy(float), 0))
        < float(norm["support_log1p_min"]) - 1e-12)
    result["heldout_D_LEAF_at_max_fraction"] = frac(held.D_LEAF.to_numpy(float) >= 1 - 1e-10)
    result["heldout_D_SUPPORT_at_max_fraction"] = frac(held.D_SUPPORT.to_numpy(float) >= 1 - 1e-10)
    return result


def audit_outer_model(outer_fold: int, outer_model, train, test, features,
                      official_oof: pd.DataFrame | None):
    tr_x = train[features].to_numpy(float)
    te_x = test[features].to_numpy(float)
    predict = np.asarray(outer_model.predict(te_x), dtype=float)
    if official_oof is not None:
        sub = official_oof.loc[official_oof.fold.eq(outer_fold)]
        if "sample_id" not in sub or "y_pred_nested" not in sub:
            raise RuntimeError("Saved Nested OOF missing sample_id/y_pred_nested")
        joined = pd.DataFrame({"sample_id": test.sample_id, "prediction": predict}).merge(
            sub[["sample_id", "y_pred_nested"]], on="sample_id", how="outer",
            validate="one_to_one", indicator=True)
        if len(joined) != len(test) or not joined._merge.eq("both").all():
            raise RuntimeError(f"Outer {outer_fold}: saved Nested OOF identities mismatch")
        max_err = float(np.max(np.abs(joined.prediction - joined.y_pred_nested)))
        if max_err > 2e-6:
            raise RuntimeError(f"Outer {outer_fold}: saved Nested model predictions differ: {max_err}")
    ref, held, norm = med_from_fitted_model(outer_model, tr_x, te_x)
    summary = {"outer_fold": outer_fold, "scheme": "outer_nested_model",
               "n_train": len(train), "n_heldout": len(test),
               "n_train_sources": int(train.source_key.nunique()),
               "n_heldout_sources": int(test.source_key.nunique()),
               "source_overlap": 0,
               **summarize_reference_gap(ref, held, norm),
               **{f"pred_{k}": v for k,v in summarize_predictions(test.lgkp, predict).items()}}
    held_rows = output_heldout_rows(test, held, ref, norm,
                                   "outer_source_heldout", outer_fold, 0,
                                   "saved_nested_outer_model", predict)
    return summary, held_rows


def split_iterator(scheme: str, X, y, groups, inner_folds: int, outer_fold: int):
    if scheme == "source":
        folds = GroupKFold(n_splits=inner_folds).split(X, y, groups=groups)
    elif scheme == "random_control":
        folds = KFold(n_splits=inner_folds, shuffle=True, random_state=42 + outer_fold).split(X, y)
    else:
        raise ValueError(f"Unknown scheme {scheme}")
    splits = list(folds)
    heldout_idx = np.concatenate([idx_hold for _, idx_hold in splits])
    if len(heldout_idx) != len(y) or set(heldout_idx.tolist()) != set(range(len(y))):
        raise RuntimeError(f"{scheme} inner splits do not partition outer training records exactly once")
    for j, (idx_train, idx_hold) in enumerate(splits, 1):
        if len(idx_train) <= CONF.topk:
            raise RuntimeError(f"Inner {scheme} fold {j} reference smaller than k={CONF.topk}")
        if scheme == "source" and set(groups[idx_train]) & set(groups[idx_hold]):
            raise RuntimeError(f"Inner source-aware fold {j} has source overlap")
    return splits


def crossfit_outer(outer_fold: int, train: pd.DataFrame, features: list[str],
                   saved_model, inner_folds: int, n_jobs: int, scheme: str):
    X = train[features].to_numpy(dtype=float)
    y = train.lgkp.to_numpy(dtype=float)
    groups = train.source_key.astype(str).to_numpy()
    splits = split_iterator(scheme, X, y, groups, inner_folds, outer_fold)
    meta_rows = []
    heldout_rows = []
    percentile_rows = []

    # get_params() is the EXACT hyperparameter set of the SAVED outer model.
    # Unlike reading only best_params, this keeps XGBoost fixed training options.
    locked_params = saved_model.get_params()
    locked_params["n_jobs"] = n_jobs
    if locked_params.get("n_estimators") is None:
        raise RuntimeError(f"Outer {outer_fold}: missing saved n_estimators")

    for inner, (idx_tr, idx_held) in enumerate(splits, start=1):
        tr_frame = train.iloc[idx_tr].reset_index(drop=True)
        held_frame = train.iloc[idx_held].reset_index(drop=True)
        common_sources = set(tr_frame.source_key.astype(str)) & set(held_frame.source_key.astype(str))
        if scheme == "source" and common_sources:
            raise RuntimeError("Unexpected source overlap in source-aware auxiliary fit")

        auxiliary_model = xgb.XGBRegressor(**locked_params)
        auxiliary_model.fit(X[idx_tr], y[idx_tr])
        predict = np.asarray(auxiliary_model.predict(X[idx_held]), dtype=float)
        ref, external, norm = med_from_fitted_model(auxiliary_model, X[idx_tr], X[idx_held])
        stats = summarize_reference_gap(ref, external, norm)
        meta_row = {
            "outer_fold": outer_fold, "inner_fold": inner, "scheme": scheme,
            "n_train": len(idx_tr), "n_heldout": len(idx_held),
            "n_train_sources": tr_frame.source_key.nunique(),
            "n_heldout_sources": held_frame.source_key.nunique(),
            "overlapping_sources": len(common_sources),
            "n_trees": int(ensure_2d_leaf_matrix(auxiliary_model.apply(X[idx_tr][:1])).shape[1]),
            **stats,
            **{f"pred_{k}": v for k,v in summarize_predictions(y[idx_held], predict).items()},
        }
        meta_rows.append(meta_row)
        heldout_rows.append(output_heldout_rows(
            held_frame, external, ref, norm, scheme, outer_fold, inner,
            "auxiliary_training_excludes_heldout_records", predict,
        ))
        for q in PERCENTILES:
            percentile_rows.append({
                "outer_fold": outer_fold, "inner_fold": inner, "scheme": scheme,
                "percentile": q, "n_train": len(idx_tr), "n_heldout": len(idx_held),
                "train_reference_threshold": stats[f"train_MED_P{q}"],
                "heldout_exceedance_fraction": stats[f"heldout_above_train_P{q}_fraction"],
            })
        print(f"    {scheme} inner {inner}/{inner_folds}: train={len(idx_tr)} held={len(idx_held)}"
              f" sources-overlap={len(common_sources)}"
              f" | train P50={stats['train_MED_P50']:.5f}"
              f" | held P50={stats['heldout_MED_P50']:.5f}"
              f" | held > train P50={100*stats['heldout_above_train_P50_fraction']:.1f}%"
              f" | delta DLeaf={stats['delta_median_D_LEAF']:+.4f}"
              f" | delta DSupport={stats['delta_median_D_SUPPORT']:+.4f}", flush=True)
        del auxiliary_model

    return pd.DataFrame(meta_rows), pd.concat(heldout_rows, ignore_index=True), pd.DataFrame(percentile_rows)


def per_outer_summary(fold_summary: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    """Use original held-out record counts, not equal inner fold weight."""
    summary = []
    for outer, sub in fold_summary.groupby("outer_fold", sort=True):
        val = heldout.loc[heldout.outer_fold.eq(outer)]
        q = {"outer_fold": int(outer), "n_crossfit_heldout": len(val),
             "n_inner_models": len(sub),
             "median_train_LOO_P50_across_inner_models": median(sub.train_MED_P50),
             "median_external_MED": median(val.MED),
             "median_external_minus_its_ref_P50": median(val.MED_minus_train_P50),
             "fraction_external_above_own_train_P50": frac(val.MED_minus_train_P50 > 0),
             "mean_percentile_vs_own_ref": float(val.MED_percentile_vs_train_LOO.mean()),
             "median_percentile_vs_own_ref": median(val.MED_percentile_vs_train_LOO),
             "median_delta_D_LEAF_across_models": median(sub.delta_median_D_LEAF),
             "median_delta_D_SUPPORT_across_models": median(sub.delta_median_D_SUPPORT),
             "weighted_pred_MAE": float(val.abs_error.mean()),
             "source_count_within_outer_train": val.source_key.nunique()}
        summary.append(q)
    return pd.DataFrame(summary)


def _validate_reused_source_results(inner: pd.DataFrame,
                                    held: pd.DataFrame, percentiles: pd.DataFrame,
                                    data_dir: Path, inner_folds: int) -> None:
    """Prevent stale/mixed-up CSVs from silently being reused for summary."""
    if len(inner) != 5 * inner_folds:
        raise RuntimeError(f"Saved source models count {len(inner)} != {5 * inner_folds}")
    if len(percentiles) != 5 * inner_folds * len(PERCENTILES):
        raise RuntimeError("Saved P40-P60 source percentile rows count mismatch")
    if not set(["outer_fold", "inner_fold", "scheme", "train_MED_P50",
                "delta_median_D_LEAF", "delta_median_D_SUPPORT"]).issubset(inner.columns):
        raise RuntimeError("Saved source summary is missing required columns")
    need = {"outer_fold", "inner_fold", "scheme", "sample_id", "source_key",
            "MED_minus_train_P50", "MED_percentile_vs_train_LOO", "MED", "abs_error"}
    if not need.issubset(held.columns):
        raise RuntimeError(f"Saved held-out evidence missing fields {sorted(need - set(held.columns))}")
    if set(inner.scheme.unique()) != {"source"} or set(held.scheme.unique()) != {"source"} or set(percentiles.scheme.unique()) != {"source"}:
        raise RuntimeError("Saved source CSVs mix experimental schemes")
    if inner.duplicated(["outer_fold", "inner_fold"]).any():
        raise RuntimeError("Duplicate inner-fold summary rows")
    if held.duplicated(["outer_fold", "sample_id"]).any():
        raise RuntimeError("Duplicate held-out samples within outer folds")
    if set(inner.outer_fold) != set(OUTER_FOLDS):
        raise RuntimeError("Saved source summary outer fold IDs incorrect")
    for fold in OUTER_FOLDS:
        tr = pd.read_csv(data_dir / f"fold{fold}_train.csv")
        tr_ids = set(tr.sample_id.astype(str))
        fold_held = held.loc[held.outer_fold.eq(fold)]
        seen_ids = set(fold_held.sample_id.astype(str))
        if len(fold_held) != len(tr) or tr_ids != seen_ids:
            raise RuntimeError(f"Saved source held-out sample membership mismatch for outer fold {fold}")
        fold_inner = inner.loc[inner.outer_fold.eq(fold)]
        if len(fold_inner) != inner_folds or set(fold_inner.inner_fold) != set(range(1, inner_folds + 1)):
            raise RuntimeError(f"Saved inner fold labels incorrect for outer fold {fold}")
        # GroupKFold requires every literature source to belong to exactly one inner held-out fold.
        if (fold_held.groupby("source_key", dropna=False).inner_fold.nunique() != 1).any():
            raise RuntimeError(f"Saved source samples cross inner held-out folds for outer {fold}")
        for _, inner_row in fold_inner.iterrows():
            j = int(inner_row.inner_fold)
            held_sources = set(fold_held.loc[fold_held.inner_fold.eq(j), "source_key"].astype(str))
            other_sources = set(fold_held.loc[fold_held.inner_fold.ne(j), "source_key"].astype(str))
            if held_sources & other_sources:
                raise RuntimeError(f"Source leakage found in saved inner fold {fold}-{j}")
            n = int((fold_held.inner_fold == j).sum())
            if n != int(inner_row.n_heldout):
                raise RuntimeError(f"Saved held-out counts mismatch at outer={fold}, inner={j}")
            threshold = fold_held.loc[fold_held.inner_fold.eq(j), "training_ref_MED_P50"].to_numpy(float)
            if threshold.size == 0 or not np.allclose(threshold, float(inner_row.train_MED_P50), atol=1e-9):
                raise RuntimeError(f"Saved reference P50 mismatch at outer={fold}, inner={j}")
        if not np.isfinite(fold_held[["MED", "MED_minus_train_P50"]].to_numpy(float)).all():
            raise RuntimeError(f"Nonfinite saved MED at outer fold {fold}")
    if percentiles.duplicated(["outer_fold", "inner_fold", "percentile"]).any():
        raise RuntimeError("Duplicate saved percentile rows")


def pooled_report(scheme: str, inner_summary: pd.DataFrame,
                  held: pd.DataFrame, outer_summary: pd.DataFrame) -> dict:
    return {
        "scheme": scheme,
        "n_auxiliary_models": int(len(inner_summary)),
        "n_crossfit_heldout_records_including_repeated_outer_train_membership": int(len(held)),
        "fraction_external_above_own_train_P50": frac(held.MED_minus_train_P50 > 0),
        "median_external_minus_own_train_P50": median(held.MED_minus_train_P50),
        "median_external_MED_percentile_vs_own_train_LOO": median(held.MED_percentile_vs_train_LOO),
        "median_delta_MED_across_auxiliary_models": median(inner_summary.delta_median_MED),
        "median_delta_D_LEAF_across_auxiliary_models": median(inner_summary.delta_median_D_LEAF),
        "median_delta_D_SUPPORT_across_auxiliary_models": median(inner_summary.delta_median_D_SUPPORT),
        "outer_fold_exceedance_min": float(outer_summary.fraction_external_above_own_train_P50.min()),
        "outer_fold_exceedance_max": float(outer_summary.fraction_external_above_own_train_P50.max()),
        "interpretation": (
            "Distribution diagnostics, NOT an unbiased population calibration estimate. "
            "Crossfit records reappear in other outer training folds, and auxiliary "
            "theta was selected using all samples in the corresponding outer training set."
        ),
    }


def main():
    a = args_parse()
    root = Path(a.project_root).expanduser().resolve() if a.project_root else Path.cwd()
    data_dir = located(root, a.data_dir)
    nested_dir = located(root, a.nested_model_dir)
    output_dir = located(root, a.output_dir)
    if any(output_dir == protected or protected in output_dir.parents for protected in (data_dir, nested_dir)):
        raise RuntimeError("Refusing to write outputs inside read-only model/data directories")
    if output_dir == root:
        raise RuntimeError("Output directory must not be project root")
    needed = []
    for outer in OUTER_FOLDS:
        needed += [
            data_dir / f"fold{outer}_train.csv",
            data_dir / f"fold{outer}_test.csv",
            nested_dir / f"nested_outer_fold{outer}.joblib",
            nested_dir / f"nested_outer_fold{outer}_meta.json",
        ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(map(str, missing)))
    CONF.validate()
    out = output_dir
    out.mkdir(parents=True, exist_ok=True)

    if a.reuse_source_results:
        paths = [out / "per_inner_summary.csv", out / "crossfit_heldout_MED.csv",
                 out / "per_outer_crossfit_summary.csv", out / "P40_P60_exceedance.csv"]
        missing_cache = [str(p) for p in paths if not p.is_file()]
        if missing_cache:
            raise FileNotFoundError(
                "--reuse-source-results requires already completed source crossfit CSV files:\n"
                + "\n".join(missing_cache))
        print("REUSE SOURCE RESULTS enabled: saved 20 auxiliary source fits will NOT run again.", flush=True)

    config = {
        "purpose": "Source-disjoint MED reference sensitivity",
        "project": str(root), "data_dir": str(data_dir),
        "nested_model_dir": str(nested_dir), "output_dir": str(out),
        "inner_folds": a.inner_folds,
        "fixed_outer_model_hyperparameters": True,
        "auxiliary_optuna": False,
        "optional_random_control": bool(a.include_random_control),
        "reuse_saved_source_crossfit": bool(a.reuse_source_results),
        "outer_test_data_used_for_model_fitting_or_calibration": False,
        "metric_config": CONF.to_dict(),
        "main_caveat": (
            "Each auxiliary model is fit on its inner-training records only, BUT "
            "its theta_i was selected on the full outer-training set, including "
            "those inner-held-out labels. This is not completely selection-independent. "
            "Reference-vs-heldout differences can also reflect source shift and model sample size. "
            "No new official MED threshold is inferred from this diagnostic."
        ),
    }
    save_json(config, out / "run_config.json")

    print("="*97)
    print("NESTED MED FITTED-LOO vs SOURCE-HELDOUT CROSSFIT DIAGNOSTIC")
    print("Project:", root)
    print("Nested models:", nested_dir)
    print("Output:", out)
    print(f"Outer=5 / inner={a.inner_folds} GroupKFold, fixed saved theta_i, no Optuna")
    print(f"Random control: {a.include_random_control} | model n_jobs={a.n_jobs}")
    print("IMPORTANT: aux theta_i is selected using entire outer train; diagnostic, NOT selection-independent calibration")
    print("IMPORTANT: no official MED/PED/Q definition is changed")
    print("="*97, flush=True)

    official_oof_path = nested_dir / "nested_oof_predictions.csv"
    official_oof = pd.read_csv(official_oof_path) if official_oof_path.is_file() else None
    if official_oof is not None:
        if official_oof.sample_id.duplicated().any():
            raise RuntimeError("Saved source-aware nested_oof_predictions.csv has duplicate sample_id")
    inner_parts = {"source": []}
    held_parts = {"source": []}
    percentile_parts = {"source": []}
    if a.include_random_control:
        for key in ("random_control",):
            inner_parts[key], held_parts[key], percentile_parts[key] = [], [], []
    outer_rows, outer_held_parts = [], []
    test_ids, test_source_fold, per_fold_partition = [], [], []

    for outer in OUTER_FOLDS:
        print(f"\n{'='*30} OUTER FOLD {outer}/5 {'='*30}", flush=True)
        train = pd.read_csv(data_dir / f"fold{outer}_train.csv")
        test = pd.read_csv(data_dir / f"fold{outer}_test.csv")
        meta = json.loads((nested_dir / f"nested_outer_fold{outer}_meta.json").read_text(encoding="utf-8"))
        model = joblib.load(nested_dir / f"nested_outer_fold{outer}.joblib")
        features = list(meta["feature_cols"])
        check_frame(train, f"outer {outer} training", features)
        check_frame(test, f"outer {outer} test", features)
        shared = set(train.source_key.astype(str)) & set(test.source_key.astype(str))
        if shared:
            raise RuntimeError(f"Outer {outer} source leakage: {len(shared)} shared sources")
        if len(train.source_key.unique()) < a.inner_folds:
            raise RuntimeError(f"Outer {outer} not enough sources for inner folds")
        if int(meta.get("fold", outer)) != outer:
            raise RuntimeError(f"Outer {outer}: wrong model meta fold")
        for pname, pvalue in meta.get("best_params", {}).items():
            saved = model.get_params().get(pname)
            if saved is None or not np.isclose(float(saved), float(pvalue), rtol=1e-7, atol=1e-9):
                raise RuntimeError(f"Outer {outer}: model/meta hyperparameter mismatch on {pname}")
        test_ids += test.sample_id.astype(str).tolist()
        test_source_fold.append(set(test.source_key.astype(str)))
        per_fold_partition.append((outer, set(train.sample_id.astype(str)), set(test.sample_id.astype(str))))
        print(f"Nested model: train={len(train)}, held-out outer={len(test)},"
              f" train/test sources={train.source_key.nunique()}/{test.source_key.nunique()},"
              " overlap=0", flush=True)

        if not a.skip_outer_audit:
            outer_stats, outer_query = audit_outer_model(outer, model, train, test, features, official_oof)
            outer_rows.append(outer_stats)
            outer_held_parts.append(outer_query)
            print(f"  Saved outer model train LOO P50={outer_stats['train_MED_P50']:.5f}"
                  f" | outer held P50={outer_stats['heldout_MED_P50']:.5f}"
                  f" | outer held > train P50={100*outer_stats['heldout_above_train_P50_fraction']:.1f}%", flush=True)

        schemes = ["source"] + (["random_control"] if a.include_random_control else [])
        for scheme in schemes:
            if scheme == "source" and a.reuse_source_results:
                print("    source crossfit: SKIPPED (using saved CSVs)", flush=True)
                continue
            stats, queries, thresholds = crossfit_outer(
                outer, train, features, model, a.inner_folds, a.n_jobs, scheme)
            inner_parts[scheme].append(stats)
            held_parts[scheme].append(queries)
            percentile_parts[scheme].append(thresholds)

    if len(test_ids) != len(set(test_ids)):
        raise RuntimeError("Outer held-out sample IDs must be unique across all five folds")
    full_ids = set(test_ids)
    for outer, train_ids, held_ids in per_fold_partition:
        if train_ids & held_ids or train_ids | held_ids != full_ids:
            raise RuntimeError(
                f"Outer fold {outer}: train/test IDs are not complementary partitions "
                "of the same full dataset")
    for ix in range(len(test_source_fold)):
        for jx in range(ix + 1, len(test_source_fold)):
            if test_source_fold[ix] & test_source_fold[jx]:
                raise RuntimeError(f"Outer test sources appear in fold {ix+1} and fold {jx+1}")

    global_summary = {"source_crossfit": None, "outer_nested": None, "random_control": None,
                      "caveat": config["main_caveat"]}
    for scheme in inner_parts:
        if scheme == "source" and a.reuse_source_results:
            all_inner = pd.read_csv(out / "per_inner_summary.csv")
            all_held = pd.read_csv(out / "crossfit_heldout_MED.csv")
            all_thresholds = pd.read_csv(out / "P40_P60_exceedance.csv")
            _validate_reused_source_results(
                all_inner, all_held, all_thresholds, data_dir, a.inner_folds)
            print("Validated saved source crossfit CSVs against all five outer training sets.", flush=True)
        else:
            all_inner = pd.concat(inner_parts[scheme], ignore_index=True)
            all_held = pd.concat(held_parts[scheme], ignore_index=True)
            all_thresholds = pd.concat(percentile_parts[scheme], ignore_index=True)
        outer_summary = per_outer_summary(all_inner, all_held)
        name = "" if scheme == "source" else "random_control_"
        save_csv(all_inner, out / f"{name}per_inner_summary.csv")
        save_csv(all_held, out / f"{name}crossfit_heldout_MED.csv")
        save_csv(outer_summary, out / f"{name}per_outer_crossfit_summary.csv")
        save_csv(all_thresholds, out / f"{name}P40_P60_exceedance.csv")
        final_summ = pooled_report(scheme, all_inner, all_held, outer_summary)
        global_summary["source_crossfit" if scheme == "source" else "random_control"] = final_summ

        print("\n" + "-"*97)
        print(f"CROSSFIT SUMMARY: {scheme.upper()}")
        print(outer_summary[["outer_fold", "n_crossfit_heldout",
            "fraction_external_above_own_train_P50",
            "median_external_minus_its_ref_P50",
            "median_delta_D_LEAF_across_models",
            "median_delta_D_SUPPORT_across_models"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
        print("Overall held-out > OWN train P50: "
              f"{100*final_summ['fraction_external_above_own_train_P50']:.2f}%")
        print("Median external MED - own train P50: "
              f"{final_summ['median_external_minus_own_train_P50']:+.5f}")
        print("Median across inner models: delta DLeaf, delta DSupport = "
              f"{final_summ['median_delta_D_LEAF_across_auxiliary_models']:+.5f}, "
              f"{final_summ['median_delta_D_SUPPORT_across_auxiliary_models']:+.5f}")

    if outer_rows:
        outer_table = pd.DataFrame(outer_rows)
        outer_held = pd.concat(outer_held_parts, ignore_index=True)
        save_csv(outer_table, out / "outer_reference_vs_test_summary.csv")
        save_csv(outer_held, out / "outer_heldout_MED.csv")
        global_summary["outer_nested"] = {
            "outer_fold_exceedance_fractions": outer_table[
                ["outer_fold", "heldout_above_train_P50_fraction"]].to_dict(orient="records"),
            "overall_heldout_above_own_train_P50_fraction": frac(
                outer_held.MED_minus_train_P50 > 0),
            "overall_median_heldout_minus_own_train_P50": median(
                outer_held.MED_minus_train_P50),
            "n_outer_heldout_records": len(outer_held),
        }
        print("\n" + "-"*97)
        print("SAVED NESTED OUTER-MODEL REFERENCE vs HELD-OUT")
        print(outer_table[["outer_fold", "n_train", "n_heldout", "train_MED_P50",
                           "heldout_MED_P50", "heldout_above_train_P50_fraction",
                           "delta_median_D_LEAF", "delta_median_D_SUPPORT"]].to_string(
                               index=False, float_format=lambda v: f"{v:.5f}"))
        print("Total outer held-out > own trained reference P50: "
              f"{100*global_summary['outer_nested']['overall_heldout_above_own_train_P50_fraction']:.2f}%")

    save_json(global_summary, out / "global_summary.json")
    print("\n" + "="*97)
    print("ALL DIAGNOSTIC CHECKS PASSED. Saved to:", out)
    print("IMPORTANT: Do NOT reinterpret higher external MED as pure in-sample overfitting:")
    print("            source-domain shift and auxiliary sample-size differences also contribute.")
    print("IMPORTANT: Theta_i was selected on all outer training records (crossfit labels indirectly used).")
    print("IMPORTANT: NO new final MED/PED threshold or Q-state assignments were produced.")
    print("="*97)


if __name__ == "__main__":
    main()
