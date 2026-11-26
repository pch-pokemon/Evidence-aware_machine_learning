# -*- coding: utf-8 -*-
"""Audit paired historical/current Nested OOF evidence transitions (no training).

Designed to follow compute_temporal_evidence.py.

What it does
------------
1. Verify 1:1 pairing, source-aware folds, P40-P60 states, MED/PED boundaries,
   frozen current evidence, and PED non-increase on reference pool expansion.
2. Source-cluster paired percentile bootstrap (frozen models and Q assignments):
   paired MAE change, transition prevalence, direction of MED/PED boundary
   changes, Q4->Q1 prevalence, and transition-driver prevalence.
3. Audit ALL changed cases; rate exact transition stability across P40-P60
   and distance to the P50 boundaries (MED distances are model-relative).
4. Pick, for Fig. 5(d) *inspection*, a Q4->Q1 improvement case and an
   opposite MED-only transition (Q1->Q3 preferred; Q2->Q4 fallback).
   Choices prioritize transition robustness, NOT target errors.
5. Trace the exact physical top-k reference samples and source_key labels
   for the selected cases and top-ranked cases in each transition category,
   using the FROZEN current-fold weights and standardization. Reproduces
   PED_raw and n_sources from the official temporal results.

IMPORTANT SCIENTIFIC LIMITS
---------------------------
- Historical/current MED use different XGBoost models and independent
  training-LOO normalizations. NEVER compare/subtract raw MED across eras.
- PED is held on one current-fold physical scale, and only its reference
  pool changes. Current PED <= historical PED is checked record by record.
- A local top-k source COUNT need NOT increase after reference pool expansion:
  nearby neighbors may be replaced by records of fewer distinct sources.
- Bootstrap resamples distinct `source_key` literature clusters, reusing
  fixed-model predictions and classifications. It is not a model-retraining
  uncertainty interval or a prospective time-split evaluation.
- The reported number of changed Q states is a conditional descriptive
  statistic, not a causal estimate of the effect of new literature.

Run from the project root (beside evidence_core.py):
    python summarize_temporal_transitions.py
    python summarize_temporal_transitions.py --bootstrap 2000 --seed 42
    python summarize_temporal_transitions.py --skip-neighbor-trace

For small synthetic smoke tests:
    python summarize_temporal_transitions.py --expected-records 0

Requires numpy, pandas, and (for neighbor tracing) evidence_core.py.
Optional openpyxl for Excel export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PERCENTILES = (40, 45, 50, 55, 60)
STATES = ("Q1", "Q2", "Q3", "Q4")
CLASSES = ("Q4->Q1", "Q1->Q3", "Q2->Q4", "Q4->Q2", "Q2->Q1", "Q4->Q3", "Q2->Q3")


def parse_args():
    p = argparse.ArgumentParser(description="Nested historical/current transition audit (NO training)")
    p.add_argument("--project-root", default=None, help="Directory containing script by default")
    p.add_argument("--cutoff", type=int, default=2022)
    p.add_argument("--temporal-dir", default=None,
                   help="Default models/nested_temporal_evidence_<cutoff>_vs_current")
    p.add_argument("--current-evidence-dir", default="models/nested_evidence_final")
    p.add_argument("--data-dir", default="splits/source_kfold")
    p.add_argument("--output-dir", default="models/nested_temporal_transition_audit")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected-records", type=int, default=613,
                   help="0 disables sample count check")
    p.add_argument("--trace-top-per-class", type=int, default=4,
                   help="Trace top N candidate records per major transition type")
    p.add_argument("--trace-sample-ids", nargs="+", default=[], help="Additional fixed query IDs to trace")
    p.add_argument("--skip-neighbor-trace", action="store_true",
                   help="Skip reference sample/source reconstruction")
    p.add_argument("--no-excel", action="store_true")
    a = p.parse_args()
    if a.bootstrap < 100 or a.trace_top_per_class < 0 or a.expected_records < 0:
        p.error("bootstrap>=100, trace-top-per-class>=0, expected-records>=0 required")
    return a


def path_under(root: Path, user_path: str) -> Path:
    path = Path(user_path).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def require(df: pd.DataFrame, cols, label: str):
    missing = [x for x in cols if x not in df.columns]
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def save_csv(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def to_float(df: pd.DataFrame, name: str) -> np.ndarray:
    v = pd.to_numeric(df[name], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(v).all():
        raise RuntimeError(f"Invalid numeric values in column {name}")
    return v


def quadrant_from_low(med_low, ped_low):
    return np.select([med_low & ped_low, med_low & ~ped_low, ~med_low & ped_low],
                     ["Q1", "Q2", "Q3"], default="Q4")


def states_boolean(q, axis: str):
    vals = np.asarray(q, dtype=str)
    return np.isin(vals, ("Q1", "Q2") if axis == "MED" else ("Q1", "Q3"))


def check_close(a, b, label, tol=2e-7):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise RuntimeError(f"{label}: nonfinite or mismatched arrays")
    max_diff = float(np.max(np.abs(x-y))) if x.size else 0.0
    if max_diff > tol:
        raise RuntimeError(f"{label}: max error {max_diff:.8g} > {tol:.8g}")
    return max_diff


def load_and_verify(args, paths):
    csv = paths["temporal"] / "temporal_comparison.csv"
    thr_csv = paths["temporal"] / "temporal_thresholds_by_fold.csv"
    current_csv = paths["current"] / "pooled_source_oof_evidence.csv"
    for p in (csv, thr_csv, current_csv):
        if not p.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {p}")
    df = pd.read_csv(csv, low_memory=False)
    thr = pd.read_csv(thr_csv, low_memory=False)
    frozen = pd.read_csv(current_csv, low_memory=False)
    y = args.cutoff
    needed = ["sample_id", "source_key", "fold", "fold_test_idx", "y_true",
              f"y_pred_{y}", "y_pred_current", f"abs_error_{y}", "abs_error_current",
              f"MED_{y}", "MED_current", f"PED_raw_{y}", "PED_raw_current",
              f"PED_for_threshold_{y}", "PED_for_threshold_current",
              f"MED_minus_threshold_P50_{y}", "MED_minus_threshold_P50_current",
              f"PED_minus_threshold_P50_{y}", "PED_minus_threshold_P50_current",
              "MED_threshold_P50_current", f"MED_threshold_P50_{y}",
              "PED_threshold_P50_common", "quadrant_current", f"quadrant_{y}",
              "state_changed", "transition_driver", "transition",
              "PED_improvement_historical_minus_current", "n_sources_current", f"n_sources_{y}",
              "delta_abs_error_current_minus_historical"]
    needed += [f"quadrant_P{p}_{era}" for p in PERCENTILES for era in (str(y), "current")]
    require(df, needed, "temporal comparison")
    require(frozen, ["sample_id", "source_key", "fold", "y_true", "y_pred", "MED", "PED_raw", "quadrant"], "frozen current OOF")
    require(thr, ["fold", "percentile", "historical_MED_threshold", "current_MED_threshold", "common_PED_threshold"], "temporal thresholds")
    if args.expected_records and len(df) != args.expected_records:
        raise RuntimeError(f"Expected {args.expected_records} records, received {len(df)}")
    for label, frame in (("temporal", df), ("frozen", frozen)):
        if frame.sample_id.isna().any() or frame.sample_id.astype(str).duplicated().any():
            raise RuntimeError(f"{label}: null or repeated sample_id")
        if frame.source_key.isna().any() or frame.groupby("source_key").fold.nunique().max() != 1:
            raise RuntimeError(f"{label}: null sources or source_key spans multiple folds")
    if set(df.sample_id.astype(str)) != set(frozen.sample_id.astype(str)):
        raise RuntimeError("Frozen current and temporal results do not contain the same sample_ids")
    merged = df[["sample_id", "fold", "source_key", "y_true", "y_pred_current", "MED_current", "PED_raw_current", "quadrant_current"]].copy()
    chk = merged.merge(frozen[["sample_id", "fold", "source_key", "y_true", "y_pred", "MED", "PED_raw", "quadrant"]],
                       on="sample_id", validate="one_to_one", suffixes=("_temp", "_frozen"))
    if (chk.fold_temp != chk.fold_frozen).any() or (chk.source_key_temp.astype(str) != chk.source_key_frozen.astype(str)).any():
        raise RuntimeError("Temporal and frozen results disagree on source and fold identities")
    for left, right in (("y_true_temp", "y_true_frozen"), ("y_pred_current", "y_pred"),
                        ("MED_current", "MED"), ("PED_raw_current", "PED_raw")):
        check_close(chk[left], chk[right], f"frozen comparison {left}")
    if (chk.quadrant_current.astype(str) != chk.quadrant.astype(str)).any():
        raise RuntimeError("Temporal and frozen current Q states disagree")
    if not set(df.quadrant_current.astype(str)).issubset(STATES) or not set(df[f"quadrant_{y}"].astype(str)).issubset(STATES):
        raise RuntimeError("Q states are not Q1-Q4")
    if (df.fold_test_idx.isna().any() or df.duplicated(["fold", "fold_test_idx"]).any()):
        raise RuntimeError("Repeated fold_test_idx per fold")
    actual_folds = sorted(df.fold.unique().tolist())
    if actual_folds != [1,2,3,4,5]:
        raise RuntimeError(f"Unexpected folds {actual_folds}; expected 1..5")
    if len(thr) != 5*len(PERCENTILES) or thr.duplicated(["fold", "percentile"]).any():
        raise RuntimeError("Threshold file must have exactly one record per (fold, percentile)")
    expect_keys = {(f,p) for f in range(1,6) for p in PERCENTILES}
    if set(zip(thr.fold.astype(int), thr.percentile.astype(int))) != expect_keys:
        raise RuntimeError("Threshold grid incomplete")
    ped_h, ped_c = to_float(df, f"PED_raw_{y}"), to_float(df, "PED_raw_current")
    if (ped_c > ped_h + 1e-10).any():
        raise RuntimeError("Expanded reference increased some PED values")
    check_close(to_float(df, "PED_improvement_historical_minus_current"), ped_h-ped_c,
                "temporal PED improvement")
    abs_h = np.abs(to_float(df, f"y_pred_{y}")-to_float(df, "y_true"))
    abs_c = np.abs(to_float(df, "y_pred_current")-to_float(df, "y_true"))
    check_close(abs_h, to_float(df, f"abs_error_{y}"), "historical absolute errors")
    check_close(abs_c, to_float(df, "abs_error_current"), "current absolute errors")
    check_close(abs_c-abs_h, to_float(df, "delta_abs_error_current_minus_historical"),
                "paired absolute error")
    # Check each Q against the exact era-specific MED threshold and frozen common PED threshold.
    for row in thr.itertuples(index=False):
        f, p = int(row.fold), int(row.percentile)
        ii = np.flatnonzero(df.fold.to_numpy() == f)
        hmed = to_float(df.iloc[ii], f"MED_{y}")
        cmed = to_float(df.iloc[ii], "MED_current")
        hped = to_float(df.iloc[ii], f"PED_for_threshold_{y}")
        cped = to_float(df.iloc[ii], "PED_for_threshold_current")
        qh = quadrant_from_low(hmed<=float(row.historical_MED_threshold), hped<=float(row.common_PED_threshold))
        qc = quadrant_from_low(cmed<=float(row.current_MED_threshold), cped<=float(row.common_PED_threshold))
        if not np.array_equal(qh,df.iloc[ii][f"quadrant_P{p}_{y}"].to_numpy(dtype=str)):
            raise RuntimeError(f"Fold {f} historical P{p} Q mismatches")
        if not np.array_equal(qc,df.iloc[ii][f"quadrant_P{p}_current"].to_numpy(dtype=str)):
            raise RuntimeError(f"Fold {f} current P{p} Q mismatches")
    qh = df[f"quadrant_{y}"].astype(str).to_numpy()
    qc = df.quadrant_current.astype(str).to_numpy()
    if not np.array_equal(qh,df[f"quadrant_P50_{y}"].astype(str).to_numpy()) or not np.array_equal(qc,df.quadrant_P50_current.astype(str).to_numpy()):
        raise RuntimeError("Primary states do not match P50")
    changed = qh != qc
    if not np.array_equal(changed, df.state_changed.astype(bool).to_numpy()):
        raise RuntimeError("state_changed column is incorrect")
    med_change = states_boolean(qh,"MED") != states_boolean(qc,"MED")
    ped_change = states_boolean(qh,"PED") != states_boolean(qc,"PED")
    derived_driver = np.select([~med_change & ~ped_change, med_change & ~ped_change,
                                ~med_change & ped_change, med_change & ped_change],
                               ["stable", "MED_only", "PED_only", "both"], default="bad")
    if not np.array_equal(derived_driver,df.transition_driver.astype(str).to_numpy()):
        raise RuntimeError("transition_driver does not match independently derived axis transitions")
    # NumPy 1.x may return object arrays for pandas strings; np.char.add does
    # not accept mixed object / Unicode dtypes in those versions.  Build the
    # transition labels with ordinary Python strings for cross-version safety.
    expected_transitions = np.asarray(
        [f"{historical}->{current}" for historical, current in zip(qh, qc)],
        dtype=str,
    )
    if not np.array_equal(expected_transitions, df.transition.astype(str).to_numpy()):
        raise RuntimeError("transition labels do not match")
    # Verify frozen P50 thresholds and era-specific margin fields for each record.
    p50 = thr[thr.percentile.eq(50)].set_index("fold")
    hmed_t = df.fold.map(p50.historical_MED_threshold).to_numpy(dtype=float)
    cmed_t = df.fold.map(p50.current_MED_threshold).to_numpy(dtype=float)
    ped_t = df.fold.map(p50.common_PED_threshold).to_numpy(dtype=float)
    check_close(hmed_t, df[f"MED_threshold_P50_{y}"], "historical P50 boundary")
    check_close(cmed_t, df["MED_threshold_P50_current"], "current P50 boundary")
    check_close(ped_t, df["PED_threshold_P50_common"], "common PED P50 boundary")
    for a, b, th, label in ((f"MED_{y}",f"MED_minus_threshold_P50_{y}",hmed_t,"hist MED"),
                             ("MED_current","MED_minus_threshold_P50_current",cmed_t,"current MED"),
                             (f"PED_for_threshold_{y}",f"PED_minus_threshold_P50_{y}",ped_t,"hist PED"),
                             ("PED_for_threshold_current","PED_minus_threshold_P50_current",ped_t,"current PED")):
        check_close(to_float(df,a)-th,to_float(df,b),label+" margin")
    return df,thr,{"n":int(len(df)),"n_sources":int(df.source_key.nunique()),
                   "n_changed":int(changed.sum()),"n_PED_increase":int(np.sum(ped_c>ped_h+1e-10)),
                   "frozen_current_reproduction":"PASS",
                   "P40_P60_Q_reproduction":"PASS","paired_records":"PASS"}


def statistics_bootstrap(df: pd.DataFrame, cutoff: int, repeats: int, seed: int):
    # Single group index for paired data; draw literature CLUSTERS with replacement.
    cluster_codes, clusters = pd.factorize(df.source_key.astype(str), sort=True)
    n_groups = len(clusters)
    if n_groups < 2:
        raise RuntimeError("At least two source clusters required for bootstrap")
    n = len(df)
    qh, qc = df[f"quadrant_{cutoff}"].astype(str).to_numpy(), df.quadrant_current.astype(str).to_numpy()
    delta = to_float(df,"abs_error_current")-to_float(df,f"abs_error_{cutoff}")
    med_h = states_boolean(qh,"MED"); med_c = states_boolean(qc,"MED")
    ped_h = states_boolean(qh,"PED"); ped_c = states_boolean(qc,"PED")
    indicators = {
        "paired_delta_MAE_current_minus_historical": delta,
        "fraction_Q_state_changed": (qh != qc).astype(float),
        "fraction_Q4_to_Q1": ((qh == "Q4") & (qc == "Q1")).astype(float),
        "fraction_Q1_to_Q3": ((qh == "Q1") & (qc == "Q3")).astype(float),
        "fraction_MED_high_to_low": ((~med_h) & med_c).astype(float),
        "fraction_MED_low_to_high": (med_h & (~med_c)).astype(float),
        "fraction_PED_high_to_low": ((~ped_h) & ped_c).astype(float),
        "fraction_PED_low_to_high": (ped_h & (~ped_c)).astype(float),
        "fraction_MED_only_transition": ((med_h != med_c) & (ped_h == ped_c)).astype(float),
        "fraction_PED_only_transition": ((med_h == med_c) & (ped_h != ped_c)).astype(float),
        "fraction_both_axes_transition": ((med_h != med_c) & (ped_h != ped_c)).astype(float),
        "fraction_historical_Q1": (qh == "Q1").astype(float),
        "fraction_current_Q1": (qc == "Q1").astype(float),
        "fraction_historical_Q4": (qh == "Q4").astype(float),
        "fraction_current_Q4": (qc == "Q4").astype(float),
    }
    indicators["delta_fraction_Q1_current_minus_historical"] = indicators["fraction_current_Q1"] - indicators["fraction_historical_Q1"]
    indicators["delta_fraction_Q4_current_minus_historical"] = indicators["fraction_current_Q4"] - indicators["fraction_historical_Q4"]
    X=np.column_stack([indicators[name] for name in indicators])
    names=list(indicators)
    rng=np.random.default_rng(seed)
    groupsizes=np.bincount(cluster_codes, minlength=n_groups)
    group_sums=np.stack([np.bincount(cluster_codes,weights=X[:,j],minlength=n_groups)
                         for j in range(X.shape[1])],axis=1)
    result=np.empty((repeats,X.shape[1]),dtype=float)
    # Multinomial source counts == independent draws of n_groups sources with replacement.
    for i in range(repeats):
        multiplicities=rng.multinomial(n_groups, np.full(n_groups,1.0/n_groups))
        denom=float(np.dot(multiplicities,groupsizes))
        result[i]=multiplicities @ group_sums / denom
    rows=[]
    for j,name in enumerate(names):
        ci=np.quantile(result[:,j],[0.025,0.975])
        rows.append({"metric":name,"estimate":float(np.mean(X[:,j])),
                     "ci95_low":float(ci[0]),"ci95_high":float(ci[1]),
                     "bootstrap_valid":repeats,"fraction_positive_bootstrap":float(np.mean(result[:,j]>0)),
                     "conditioning":"fixed_nested_models_and_original_OOF_states",
                     "resampling_unit":"source_key"})
    return pd.DataFrame(rows)


def candidate_ranking(df:pd.DataFrame, thr:pd.DataFrame, cutoff:int):
    base=df.copy()
    source=base[f"quadrant_{cutoff}"].astype(str)
    current=base["quadrant_current"].astype(str)
    base["n_matching_transition_P40_P60"]=np.zeros(len(base),dtype=int)
    base["n_matching_driver_P40_P60"]=np.zeros(len(base),dtype=int)
    p50transition=source+"->"+current
    med_h0=states_boolean(source,"MED"); med_c0=states_boolean(current,"MED")
    ped_h0=states_boolean(source,"PED"); ped_c0=states_boolean(current,"PED")
    for p in PERCENTILES:
        h=base[f"quadrant_P{p}_{cutoff}"].astype(str); c=base[f"quadrant_P{p}_current"].astype(str)
        base["n_matching_transition_P40_P60"]+=(h+"->"+c).eq(p50transition).to_numpy(dtype=int)
        med_h=states_boolean(h,"MED");med_c=states_boolean(c,"MED")
        ped_h=states_boolean(h,"PED");ped_c=states_boolean(c,"PED")
        base["n_matching_driver_P40_P60"]+=((med_h!=med_c)==(med_h0!=med_c0)) & ((ped_h!=ped_c)==(ped_h0!=ped_c0))
    base["same_transition_all_P40_P60"]=(base.n_matching_transition_P40_P60==len(PERCENTILES))
    base["same_driver_all_P40_P60"]=(base.n_matching_driver_P40_P60==len(PERCENTILES))
    for name in (f"MED_minus_threshold_P50_{cutoff}","MED_minus_threshold_P50_current",
                 f"PED_minus_threshold_P50_{cutoff}","PED_minus_threshold_P50_current"):
        base[f"abs_{name}"]=np.abs(to_float(base,name))
    margin_cols=[f"abs_{k}" for k in (f"MED_minus_threshold_P50_{cutoff}","MED_minus_threshold_P50_current",
                 f"PED_minus_threshold_P50_{cutoff}","PED_minus_threshold_P50_current")]
    base["min_abs_four_P50_margins"] = base[margin_cols].min(axis=1)
    base["min_abs_MED_P50_margin"] = base[[margin_cols[0],margin_cols[1]]].min(axis=1)
    base["min_abs_PED_P50_margin"] = base[[margin_cols[2],margin_cols[3]]].min(axis=1)
    # Convert each own-threshold P50 margin into units of its own P40-P60
    # reference width, avoiding arbitrary comparisons of raw MED/PED scales.
    threshold_widths = {}
    for field in ("historical_MED_threshold", "current_MED_threshold", "common_PED_threshold"):
        pivot = thr.pivot(index="fold", columns="percentile", values=field)
        width = (pivot[60] - pivot[40]).abs()
        # A degenerate reference interval has no stable percentile scale.
        # Use 1.0 conservatively rather than amplifying a zero width.
        width = width.where(width > 1e-9, 1.0)
        threshold_widths[field] = base.fold.map(width).to_numpy(dtype=float)
    margin_scales = [threshold_widths["historical_MED_threshold"],
                     threshold_widths["current_MED_threshold"],
                     threshold_widths["common_PED_threshold"],
                     threshold_widths["common_PED_threshold"]]
    base["min_relative_P50_clearance_P40P60_units"] = np.min(
        np.stack([base[k].to_numpy(dtype=float)/scale for k,scale in zip(margin_cols,margin_scales)],axis=1),axis=1
    )
    base["candidate_class"] = np.where(base.state_changed,base.transition,"unchanged")
    base["rank_within_transition"] = np.nan
    for cls, group in base[base.state_changed].groupby("transition"):
        order=group.sort_values(["n_matching_transition_P40_P60","min_relative_P50_clearance_P40P60_units",
                                 "PED_improvement_historical_minus_current","sample_id"],
                                ascending=[False,False,False,True],kind="stable").index
        base.loc[order,"rank_within_transition"]=np.arange(1,len(order)+1)
    changed=base.loc[base.state_changed].sort_values(["candidate_class","rank_within_transition","sample_id"])
    return base,changed


def select_examples(cases:pd.DataFrame):
    """Only candidate recommendations, not automatic factual interpretation."""
    selected=[]; used_sources=set()
    priorities=[("A: combined shift into Q1",["Q4->Q1"]),
                ("B: MED-only reverse shift",["Q1->Q3","Q2->Q4"])]
    for role, classes in priorities:
        pool=cases[cases.transition.isin(classes)].copy()
        if role.startswith("B"):
            pool=pool[pool.transition_driver.eq("MED_only")]
        if pool.empty:
            continue
        pool["already_source"] = pool.source_key.astype(str).isin(used_sources)
        pool["priority_class"] = pool.transition.astype(str).map({c:i for i,c in enumerate(classes)}).fillna(len(classes))
        pool=pool.sort_values(["already_source","priority_class","n_matching_transition_P40_P60",
                               "min_relative_P50_clearance_P40P60_units","PED_improvement_historical_minus_current","sample_id"],
                              ascending=[True,True,False,False,False,True],kind="stable")
        pick=pool.iloc[0].copy()
        pick["fig5d_role"]=role
        pick["selection_reason"]="Prioritize P40-P60 transition stability and own-reference P50 clearance; never rank on prediction error"
        pick["selection_caution"]=("Requires qualitative source/physics review" if int(pick["n_matching_transition_P40_P60"])>=3
                                    else "LOW P40-P60 stability: illustrative only, consider another case")
        selected.append(pick)
        used_sources.add(str(pick.source_key))
    if not selected:
        return pd.DataFrame(columns=list(cases.columns)+["fig5d_role","selection_reason"])
    return pd.DataFrame(selected).reset_index(drop=True)


def reconstruct_neighbor_traces(paths, cases:pd.DataFrame, selected:pd.DataFrame, args):
    if cases.empty and selected.empty:
        return pd.DataFrame(),pd.DataFrame()
    # Import only for neighbor tracing, not for bootstrap or candidate selection.
    from evidence_core import (EvidenceMetricConfig, build_physical_arrays,
                               physical_distance_components_for_query, topk_indices)
    ids=set(selected.sample_id.astype(str)) if not selected.empty else set()
    ids.update(args.trace_sample_ids)
    for cls in CLASSES:
        sub=cases.loc[cases.transition.eq(cls)].sort_values("rank_within_transition")
        ids.update(sub.sample_id.astype(str).head(args.trace_top_per_class))
    if not ids:
        return pd.DataFrame(),pd.DataFrame()
    targets=cases.loc[cases.sample_id.astype(str).isin(ids)].copy()
    by_fold=targets.groupby("fold")
    neighbor_rows=[];summary_rows=[]
    for fold,group in by_fold:
        fold=int(fold)
        train_path=paths["data"]/f"fold{fold}_train.csv"
        test_path=paths["data"]/f"fold{fold}_test.csv"
        cal_path=paths["current"]/"weights"/f"fold{fold}_evidence_calibration.json"
        for p in (train_path,test_path,cal_path):
            if not p.exists():
                raise FileNotFoundError(f"Required for top-k trace: {p}")
        train=pd.read_csv(train_path,low_memory=False)
        test=pd.read_csv(test_path,low_memory=False)
        with open(cal_path,encoding="utf-8") as handle:
            calibration=json.load(handle)
        if int(calibration.get("fold",-1))!=fold:
            raise RuntimeError("Calibration fold mismatch")
        config=EvidenceMetricConfig(**calibration["config"])
        config.validate()
        if config.topk!=10:
            print(f"NOTICE: Fold {fold} topk from frozen config is {config.topk}, not hardcoded 10")
        main_w=calibration["main_weights"]
        protocol_w=calibration["protocol_weights"]
        arrays=build_physical_arrays(train,test,main_w,protocol_w)
        years=pd.to_numeric(train["Year"],errors="coerce")
        hist_mask=(years<=args.cutoff).fillna(False).to_numpy()
        hist_pos=np.flatnonzero(hist_mask)
        if len(hist_pos)<config.topk:
            raise RuntimeError(f"Fold {fold}: historical pool has fewer than topk records")
        if set(train.source_key.astype(str)) & set(test.source_key.astype(str)):
            raise RuntimeError(f"Fold {fold} trace data have source leakage")
        for cand in group.itertuples(index=False):
            pos=int(cand.fold_test_idx)
            if not 0<=pos<len(test) or str(test.iloc[pos].sample_id)!=str(cand.sample_id):
                raise RuntimeError(f"Fold {fold}: traced sample_id/test index mismatch: {cand.sample_id}")
            dist,_,_=physical_distance_components_for_query(arrays,pos,config)
            ind_cur=topk_indices(dist,config.topk)
            ind_hist=hist_pos[topk_indices(dist[hist_pos],config.topk)]
            neighbor_h=train.iloc[ind_hist]
            neighbor_c=train.iloc[ind_cur]
            check_close([np.mean(dist[ind_hist])],[getattr(cand,f"PED_raw_{args.cutoff}")],
                        f"Fold {fold} {cand.sample_id} historical PED",2e-7)
            check_close([np.mean(dist[ind_cur])],[cand.PED_raw_current],
                        f"Fold {fold} {cand.sample_id} current PED",2e-7)
            n_hist=int(neighbor_h.source_key.astype(str).nunique())
            n_curr=int(neighbor_c.source_key.astype(str).nunique())
            if n_hist!=int(getattr(cand,f"n_sources_{args.cutoff}")) or n_curr!=int(cand.n_sources_current):
                raise RuntimeError(f"Fold {fold} {cand.sample_id} source counts not reproducible")
            s_hist=set(neighbor_h.source_key.astype(str))
            s_curr=set(neighbor_c.source_key.astype(str))
            summary_rows.append({"sample_id":str(cand.sample_id),"fold":fold,
                                 "source_key_query":str(cand.source_key),"transition":str(cand.transition),
                                 "rank_within_transition":cand.rank_within_transition,
                                 "n_hist_neighbors":len(ind_hist),"n_current_neighbors":len(ind_cur),
                                 "n_sources_hist":n_hist,"n_sources_current":n_curr,
                                 "source_keys_entered_topk":";".join(sorted(s_curr-s_hist)),
                                 "source_keys_exited_topk":";".join(sorted(s_hist-s_curr)),
                                 "n_sources_entered_topk":len(s_curr-s_hist),
                                 "n_sources_exited_topk":len(s_hist-s_curr),
                                 "n_current_neighbors_post_cutoff":int((pd.to_numeric(neighbor_c.Year,errors="coerce")>args.cutoff).sum()),
                                 "shared_neighbor_record_count":len(set(neighbor_h.sample_id.astype(str))&set(neighbor_c.sample_id.astype(str))),
                                 "PED_raw_hist_rebuilt":float(np.mean(dist[ind_hist])),
                                 "PED_raw_current_rebuilt":float(np.mean(dist[ind_cur]))})
            for era,neighbors in ((f"historical_{args.cutoff}",ind_hist),("current",ind_cur)):
                for rank,index in enumerate(neighbors,1):
                    row=train.iloc[int(index)]
                    yr=pd.to_numeric(pd.Series([row.get("Year",np.nan)]),errors="coerce").iloc[0]
                    neighbor_rows.append({"query_sample_id":str(cand.sample_id),"fold":fold,
                                          "query_source_key":str(cand.source_key),"transition":str(cand.transition),
                                          "reference_period":era,"neighbor_rank":rank,
                                          "neighbor_sample_id":str(row.sample_id),
                                          "neighbor_source_key":str(row.source_key),
                                          "neighbor_Ref_source_num":str(row.get("Ref_source_num","")),
                                          "neighbor_Year":yr,
                                          "neighbor_lgkp":float(row.lgkp),
                                          "physical_distance":float(dist[int(index)]),
                                          "is_post_cutoff_reference":bool(pd.notna(yr) and yr>args.cutoff),
                                          "included_in_historical_pool":bool(hist_mask[int(index)])})
        print(f"  Neighbor trace fold {fold}: {len(group)} candidate records, PED and n_sources reproducible",flush=True)
    return pd.DataFrame(summary_rows),pd.DataFrame(neighbor_rows)


def export_excel(path:Path, frames:dict[str,pd.DataFrame]):
    from openpyxl.styles import PatternFill,Font
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path,engine="openpyxl") as writer:
        for sheet,data in frames.items():
            if data is None or data.empty:
                continue
            data.to_excel(writer,sheet_name=sheet[:31],index=False)
            ws=writer.sheets[sheet[:31]]
            ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
            for cell in ws[1]:
                cell.font=Font(bold=True,color="FFFFFF")
                cell.fill=PatternFill("solid",fgColor="17365D")
            for j,col in enumerate(data.columns,1):
                ws.column_dimensions[get_column_letter(j)].width=min(41,max(12,len(str(col))+2))


def main():
    args=parse_args()
    root=Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd()
    temporal_dir=args.temporal_dir or f"models/nested_temporal_evidence_{args.cutoff}_vs_current"
    paths={"temporal":path_under(root,temporal_dir),
           "current":path_under(root,args.current_evidence_dir),
           "data":path_under(root,args.data_dir),
           "out":path_under(root,args.output_dir)}
    for key in ("temporal","current","data"):
        protected=paths[key]
        if paths["out"]==protected or paths["out"] in protected.parents or protected in paths["out"].parents:
            raise RuntimeError(f"Output overlaps protected input directory {protected}")
    print("="*100)
    print("NESTED HISTORICAL/CURRENT TRANSITION AUDIT (NO TRAINING)")
    print(f"Project: {root}")
    print(f"Temporal comparison: {paths['temporal']}")
    print(f"Frozen current OOF: {paths['current']}")
    print(f"Output: {paths['out']}")
    print(f"Cluster bootstrap={args.bootstrap}, seed={args.seed}, trace={not args.skip_neighbor_trace}")
    print("Historical/current MED use own model-relative thresholds; common current-fold PED metric")
    print("="*100,flush=True)
    df,thr,checks=load_and_verify(args,paths)
    print(f"PAIRED DATA CHECK: PASS | records={len(df)}, distinct sources={df.source_key.nunique()}",flush=True)
    boot=statistics_bootstrap(df,args.cutoff,args.bootstrap,args.seed)
    all_rows,changed=candidate_ranking(df,thr,args.cutoff)
    selected=select_examples(changed)
    p_summary=[]
    for p in PERCENTILES:
        h=df[f"quadrant_P{p}_{args.cutoff}"].astype(str)
        c=df[f"quadrant_P{p}_current"].astype(str)
        p_summary.append({"percentile":p,"n_changed":int((h!=c).sum()),
                          "fraction_changed":float((h!=c).mean()),
                          "Q4_to_Q1":int(((h=="Q4")&(c=="Q1")).sum()),
                          "Q1_to_Q3":int(((h=="Q1")&(c=="Q3")).sum())})
    ps=pd.DataFrame(p_summary)
    # Only genuinely changed transitions belong in the transition-class table.
    # Including stable Qx->Qx rows here would make fraction_among_changed invalid.
    class_counts=changed.transition.value_counts().rename_axis("transition").reset_index(name="n")
    class_counts["fraction_overall"]=class_counts.n/len(df)
    class_counts["fraction_among_changed"]=class_counts.n/len(changed) if len(changed) else np.nan
    neighbor_summary,neighbor_detail=pd.DataFrame(),pd.DataFrame()
    if not args.skip_neighbor_trace:
        neighbor_summary,neighbor_detail=reconstruct_neighbor_traces(paths,changed,selected,args)
        if not selected.empty and not set(selected.sample_id.astype(str)).issubset(set(neighbor_summary.sample_id.astype(str))):
            raise RuntimeError("A selected Fig. 5(d) candidate has no audited physical references")
    out=paths["out"]
    out.mkdir(parents=True,exist_ok=True)
    exports={
        "bootstrap_source_cluster_CI.csv":boot,
        "all_transition_candidates_audit.csv":changed,
        "fig5d_representative_case_candidates.csv":selected,
        "P40_P60_transition_robustness.csv":ps,
        "transition_counts.csv":class_counts,
    }
    if not neighbor_summary.empty:
        exports["traced_topk_neighbor_summary.csv"]=neighbor_summary
    if not neighbor_detail.empty:
        exports["traced_topk_neighbor_records.csv"]=neighbor_detail
    for name,frame in exports.items():
        save_csv(out/name,frame)
    checks.update({"bootstrap_replicates":args.bootstrap,
                   "bootstrap_literature_clustering_unit":"source_key",
                   "hist_MED_comparison":"model_relative_quadrant_only_no_raw_cross_model_difference",
                   "PED_comparison":"shared_current_fold_scale_nonnecreasing_support",
                   "n_transition_audited":int(len(changed)),
                   "n_fig5d_candidates":int(len(selected)),
                   "n_neighbor_trace_queries":int(len(neighbor_summary)),
                   "selection":"heuristic_preselection_not_mechanistic_validation"})
    (out/"audit_checks.json").write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding="utf-8")
    if not args.no_excel:
        try:
            export_excel(out/"nested_temporal_transition_audit.xlsx",{
                "Bootstrap_CI":boot,"P40_P60":ps,"Transition_Counts":class_counts,
                "Candidate_Audit":changed,"Fig5d_Candidates":selected,
                "Neighbor_Summary":neighbor_summary,"Neighbor_Records":neighbor_detail})
            print("Excel saved:",out/"nested_temporal_transition_audit.xlsx",flush=True)
        except ImportError:
            print("openpyxl unavailable; CSVs were saved.",flush=True)
    print("\n"+"="*100)
    print("FINAL NESTED TEMPORAL TRANSITION AUDIT")
    print("="*100)
    print("PAIRED/FROZEN/P40-P60/PED CHECKS: PASS")
    print("SOURCE-CLUSTER BOOTSTRAP 95% CI (FROZEN MODELS):")
    print(boot[["metric","estimate","ci95_low","ci95_high"]].to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nTRANSITION STABILITY ACROSS P40-P60:")
    print(ps.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nCHANGED SAMPLE CASE AUDIT:")
    print(f"  cases={len(changed)}, P40-P60 exact transition 5/5 = {int(changed.same_transition_all_P40_P60.sum())}")
    print("  counts per transition:")
    print(class_counts.head(20).to_string(index=False) if len(class_counts) else "  No transitions")
    print("\nFIG. 5(d) CANDIDATES (subject to scientific review):")
    cols=["fig5d_role","sample_id","source_key","fold","transition","transition_driver",
          "n_matching_transition_P40_P60","min_relative_P50_clearance_P40P60_units",
          "PED_improvement_historical_minus_current","delta_abs_error_current_minus_historical"]
    print(selected[cols].to_string(index=False,float_format=lambda x:f"{x:.6f}") if not selected.empty else "  No eligible cases")
    if not neighbor_summary.empty:
        print(f"NEIGHBOR TRACING: PASS, {len(neighbor_summary)} distinct queries, "
              f"{len(neighbor_detail)} traced reference records",flush=True)
    print("\nINTERPRETATION NOTES:")
    print("  - Conditional source-cluster bootstrap, no model refitting or hyperparameter reselection.")
    print("  - Historical/current MED are model-relative; do not subtract raw cross-model MED.")
    print("  - Selection uses evidence robustness, not best improvement in prediction accuracy.")
    print("  - Top-k counts may decrease even when the physical reference pool expands.")
    print("ALL CHECKS PASSED. Saved to:",out,flush=True)


if __name__=="__main__":
    main()
