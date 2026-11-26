# -*- coding: utf-8 -*-
"""Exploratory chronological risk--coverage / evidence-reliability analysis.

Run from the project root (the controlled working directory):

    python code/chronological_validation/analyze_chronological_error.py
    python code/chronological_validation/analyze_chronological_error.py --bootstrap-draws 5000

Reads ONLY existing evaluation outputs from the frozen 2022-cutoff run:
  runs/<run-name>/evaluation/future_predictions.csv
  runs/<run-name>/evaluation/historical_reference.csv

DOES NOT train, tune, change evidence states, adjust cutoffs, or alter the
historical model / original screening result. All selection rules are based
on pre-outcome prediction/evidence values. Source-bootstrap intervals are
EXPLORATORY / pointwise, not confirmation after inspecting future outcomes.

Risk--coverage means select a fraction of recorded future *predictions* for
reliability evaluation, NOT improved alloy performance / prospective success.
A smaller selected-subset MAE is better.

Joint_MAX is an ADDITIONAL diagnostic score, NOT the original Q1--Q4 metric:
  max(history_ECDF(MED), history_ECDF(PED_log1p)).
Lower is better; this is conservative because BOTH distances must be small.
Historical empirical percentiles are outcome-blind and avoid mixing raw units.
No future target is used in selecting records or assigning strata.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = "source_key"
STATES = ("Q1", "Q2", "Q3", "Q4")
METHODS = {
    "MED_only": "MED_ref_pct",
    "PED_only": "PED_ref_pct",
    "Joint_MAX": "Joint_MAX_ref_pct",
    "Prediction_only": "y_pred",
}


def fnum(x, digits=4):
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def parse_coverages(arg):
    try:
        vals = sorted(set(float(x.strip()) for x in arg.split(",")))
    except ValueError as e:
        raise ValueError("--coverages must contain comma-separated numeric fractions") from e
    if not vals or any(not math.isfinite(x) or not 0 < x <= 1 for x in vals):
        raise ValueError("--coverages must be in (0, 1]")
    if vals[-1] != 1.0:
        vals.append(1.0)
    return vals


def checked_csv(path, must_have, description):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}; run validate_chronological_holdout.py evaluate first.")
    df = pd.read_csv(path, dtype={"sample_id": str, SOURCE: str})
    missing = set(must_have) - set(df.columns)
    if missing:
        raise ValueError(f"{description} missing columns: {sorted(missing)}")
    if not len(df):
        raise ValueError(f"{description} is empty")
    for c in ("MED", "PED_log1p"):
        df[c] = pd.to_numeric(df[c], errors="raise")
        if not np.isfinite(df[c].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid {c} in {description}")
    df[SOURCE] = df[SOURCE].astype("string").str.strip().str.lower()
    if df[SOURCE].isna().any() or df[SOURCE].eq("").any():
        raise ValueError(f"Invalid source identifiers in {description}")
    if "sample_id" in df.columns and (df.sample_id.isna().any() or df.sample_id.duplicated().any()):
        raise ValueError(f"Missing or duplicate sample_id in {description}")
    return df


def load_data(run):
    directory = run / "evaluation"
    fut_path = directory / "future_predictions.csv"
    ref_path = directory / "historical_reference.csv"
    future = checked_csv(fut_path, ["sample_id", SOURCE, "quadrant", "MED", "PED_log1p", "y_pred", "y_true"], "future predictions")
    hist = checked_csv(ref_path, ["sample_id", SOURCE, "MED", "PED_log1p"], "historical reference")
    future["quadrant"] = future.quadrant.astype("string").str.strip()
    if future.quadrant.isna().any() or not set(future.quadrant).issubset(STATES):
        raise ValueError("Invalid Q1--Q4 labels")
    for c in ("y_pred", "y_true"):
        future[c] = pd.to_numeric(future[c], errors="raise")
        if not np.isfinite(future[c].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid {c} values")
    if set(future[SOURCE]) & set(hist[SOURCE]):
        raise ValueError("Historical and future sources overlap: chronology is not source-disjoint")
    if "availability_year" in future.columns:
        if (pd.to_numeric(future.availability_year, errors="raise") <= 2022).any():
            raise ValueError("Future predictions include a record dated <= 2022")
    if "availability_year" in hist.columns:
        if (pd.to_numeric(hist.availability_year, errors="raise") > 2022).any():
            raise ValueError("Historical reference includes a record dated > 2022")
    future["abs_error"] = (future.y_pred - future.y_true).abs()
    future["error"] = future.y_pred - future.y_true
    future["support_group"] = future.quadrant.map({
        "Q1": "model_supported", "Q2": "model_supported",
        "Q3": "model_deficient", "Q4": "model_deficient"})
    if future.support_group.isna().any():
        raise ValueError("Unmapped evidence state")
    # Historical empirical percentile uses the training-reference distribution
    # only; future outcomes and the future MED/PED distribution are NOT involved.
    for col, out in [("MED", "MED_ref_pct"), ("PED_log1p", "PED_ref_pct")]:
        sorted_ref = np.sort(hist[col].to_numpy(dtype=float))
        future[out] = np.searchsorted(sorted_ref, future[col].to_numpy(dtype=float), side="right") / len(sorted_ref)
    future["Joint_MAX_ref_pct"] = future[["MED_ref_pct", "PED_ref_pct"]].max(axis=1)
    # All comparisons share the exact same records and historical reference.
    return future.reset_index(drop=True), hist, fut_path, ref_path


class ClusterBootstrap:
    """Re-sample independent source clusters; freeze selections and strata."""

    def __init__(self, df, draws, seed):
        self.n_draws = int(draws)
        self.sources, self.codes = np.unique(df[SOURCE].astype(str), return_inverse=True)
        n_sources = len(self.sources)
        rng = np.random.default_rng(seed)
        self.counts = rng.multinomial(n_sources, np.full(n_sources, 1 / n_sources), size=draws)
        self.values = df.abs_error.to_numpy(dtype=float)
        self.frame = df
        self.all_source_count = n_sources

    def mean(self, indices):
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            return np.full(self.n_draws, np.nan)
        codes = self.codes[idx]
        sums = np.bincount(codes, weights=self.values[idx], minlength=self.all_source_count)
        ns = np.bincount(codes, minlength=self.all_source_count)
        denominator = self.counts @ ns
        numerator = self.counts @ sums
        return np.divide(numerator, denominator,
                         out=np.full(self.n_draws, np.nan, dtype=float), where=denominator > 0)

    def describe(self, samples, point, min_sources=3):
        valid = samples[np.isfinite(samples)]
        required = max(100, int(self.n_draws * 0.5))
        if len(valid) < required or min_sources < 3:
            return {"estimate": point, "CI95_low": np.nan, "CI95_high": np.nan,
                    "valid_draws": len(valid), "min_informative_sources": min_sources}
        lo, hi = np.percentile(valid, [2.5, 97.5])
        return {"estimate": point, "CI95_low": float(lo), "CI95_high": float(hi),
                "valid_draws": len(valid), "min_informative_sources": min_sources}

    def sources_in(self, indices):
        return int(len(np.unique(self.codes[np.asarray(indices, dtype=int)])))


def simple_gap(frame, boot, mask_a, mask_b, label, positive_means):
    ia = np.flatnonzero(np.asarray(mask_a, dtype=bool))
    ib = np.flatnonzero(np.asarray(mask_b, dtype=bool))
    point = (frame.abs_error.iloc[ia].mean() - frame.abs_error.iloc[ib].mean()) if len(ia) and len(ib) else np.nan
    draws = boot.mean(ia) - boot.mean(ib)
    return {
        "comparison": label,
        "positive_means": positive_means,
        "A_records": len(ia), "A_sources": boot.sources_in(ia),
        "A_MAE": float(frame.abs_error.iloc[ia].mean()) if len(ia) else np.nan,
        "B_records": len(ib), "B_sources": boot.sources_in(ib),
        "B_MAE": float(frame.abs_error.iloc[ib].mean()) if len(ib) else np.nan,
        **boot.describe(draws, float(point), min(boot.sources_in(ia), boot.sources_in(ib))),
    }


def risk_coverage(frame, boot, coverages):
    n = len(frame)
    all_curves = []
    selected = {}
    for method, col in METHODS.items():
        order = frame.sort_values([col, "sample_id"], kind="mergesort").index.to_numpy()
        for fraction in coverages:
            k = min(n, max(1, math.ceil(n * fraction - 1e-12)))
            ids = np.asarray(order[:k], dtype=int)
            selected[(method, k)] = ids
            draws = boot.mean(ids)
            all_curves.append({
                "method": method, "target_coverage": fraction, "selected_records": k,
                "actual_coverage": k / n, "selected_sources": boot.sources_in(ids),
                "MAE": float(frame.abs_error.iloc[ids].mean()),
                "full_MAE": float(frame.abs_error.mean()),
                **{k: v for k, v in boot.describe(draws, float(frame.abs_error.iloc[ids].mean()), boot.sources_in(ids)).items()
                   if k != "estimate"},
            })
    curve = pd.DataFrame(all_curves)
    rows = []
    for fraction in coverages:
        k = min(n, max(1, math.ceil(n * fraction - 1e-12)))
        for comparator in ("MED_only", "PED_only", "Prediction_only"):
            # Positive = comparator MAE higher, so Joint_MAX lower (better).
            ia, ib = selected[(comparator, k)], selected[("Joint_MAX", k)]
            point = float(frame.abs_error.iloc[ia].mean() - frame.abs_error.iloc[ib].mean())
            draws = boot.mean(ia) - boot.mean(ib)
            # Only differing sources identify the paired comparison.
            different = np.setxor1d(ia, ib)
            n_info = boot.sources_in(different) if len(different) else 0
            rows.append({
                "target_coverage": fraction, "selected_records": k,
                "comparator": comparator, "joint_method": "Joint_MAX",
                "comparator_MAE": float(frame.abs_error.iloc[ia].mean()),
                "Joint_MAX_MAE": float(frame.abs_error.iloc[ib].mean()),
                "changed_records": len(different), "changed_sources": n_info,
                "difference_definition": "comparator_MAE - Joint_MAX_MAE; positive favors joint",
                **boot.describe(draws, point, n_info),
            })
    comparison = pd.DataFrame(rows)
    aurc = []
    for method, group in curve.groupby("method", sort=False):
        group = group.sort_values("actual_coverage")
        x, y = group.actual_coverage.to_numpy(), group.MAE.to_numpy()
        area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / (x.max() - x.min())) if len(group) > 1 else np.nan
        aurc.append({"method": method, "AURC_on_tested_coverage_range": area,
                     "min_actual_coverage": x.min(), "max_actual_coverage": x.max(),
                     "note": "Descriptive; lower is better; integrate only displayed coverage range"})
    return curve, comparison, pd.DataFrame(aurc)


def make_pred_strata(frame, bins, by_family):
    out = frame.copy()
    out["pred_bin"] = ""
    if by_family and "alloy_family" in out.columns:
        group_cols = ["alloy_family"]
    else:
        group_cols = []
    groups = out.groupby(group_cols, dropna=False, sort=True) if group_cols else [("all", out)]
    for _, group in groups:
        ordered = group.sort_values(["y_pred", "sample_id"], kind="mergesort")
        # Equal-count bins. Deterministic tie-breaking uses only sample_id.
        cuts = np.floor(np.arange(len(ordered)) * bins / len(ordered)).astype(int)
        out.loc[ordered.index, "pred_bin"] = [f"B{b+1}" for b in cuts]
    if group_cols:
        out["stratum"] = out["alloy_family"].astype(str) + ":" + out.pred_bin
    else:
        out["stratum"] = out.pred_bin
    return out


def stratified_gap(frame, boot, bins, min_n, min_sources, by_family):
    work = make_pred_strata(frame, bins, by_family)
    info = []
    weights_and_ids = []
    for label, grp in work.groupby("stratum", sort=True):
        a = grp.loc[grp.support_group.eq("model_deficient")]
        b = grp.loc[grp.support_group.eq("model_supported")]
        weight = min(len(a), len(b))
        eligibility = (len(a) >= min_n and len(b) >= min_n
                       and a[SOURCE].nunique() >= min_sources
                       and b[SOURCE].nunique() >= min_sources)
        info.append({"stratification": "family+prediction" if by_family else "prediction_only",
                     "stratum": label, "records": len(grp), "sources": grp[SOURCE].nunique(),
                     "n_deficient": len(a), "sources_deficient": a[SOURCE].nunique(),
                     "n_supported": len(b), "sources_supported": b[SOURCE].nunique(),
                     "deficient_MAE": a.abs_error.mean() if len(a) else np.nan,
                     "supported_MAE": b.abs_error.mean() if len(b) else np.nan,
                     "gap_deficient_minus_supported": a.abs_error.mean() - b.abs_error.mean() if len(a) and len(b) else np.nan,
                     "balance_weight": weight if eligibility else 0, "eligible": bool(eligibility)})
        if eligibility:
            weights_and_ids.append((weight, a.index.to_numpy(dtype=int), b.index.to_numpy(dtype=int)))
    summary = {"stratification": "family+prediction" if by_family else "prediction_only",
               "requested_bins_per_group": bins, "eligible_strata": len(weights_and_ids),
               "retained_records_in_eligible_strata": sum(len(x[1]) + len(x[2]) for x in weights_and_ids),
               "definition": "min-group-count weighted within-stratum MAE(deficient)-MAE(supported)"}
    if not weights_and_ids:
        summary.update({"estimate": np.nan, "CI95_low": np.nan, "CI95_high": np.nan,
                        "valid_draws": 0, "min_informative_sources": 0})
    else:
        w = np.array([item[0] for item in weights_and_ids], dtype=float)
        pt = np.array([frame.abs_error.iloc[a].mean() - frame.abs_error.iloc[b].mean()
                       for _, a, b in weights_and_ids])
        paired = np.column_stack([boot.mean(a) - boot.mean(b) for _, a, b in weights_and_ids])
        valid = np.isfinite(paired).all(axis=1)
        combined = np.full(boot.n_draws, np.nan)
        combined[valid] = paired[valid] @ w / w.sum()
        all_a = np.concatenate([x[1] for x in weights_and_ids])
        all_b = np.concatenate([x[2] for x in weights_and_ids])
        n_info = min(boot.sources_in(all_a), boot.sources_in(all_b))
        summary.update(boot.describe(combined, float(pt @ w / w.sum()), n_info))
    return pd.DataFrame(info), pd.DataFrame([summary])


def leave_one_source_out(frame):
    rows = []
    for source in sorted(frame[SOURCE].astype(str).unique()):
        remaining = frame.loc[frame[SOURCE].ne(source)]
        q4 = remaining.loc[remaining.quadrant.eq("Q4"), "abs_error"]
        q1 = remaining.loc[remaining.quadrant.eq("Q1"), "abs_error"]
        d = remaining.loc[remaining.support_group.eq("model_deficient"), "abs_error"]
        s = remaining.loc[remaining.support_group.eq("model_supported"), "abs_error"]
        rows.append({"removed_source": source,
                     "removed_records": len(frame) - len(remaining),
                     "Q4_minus_Q1_MAE": q4.mean() - q1.mean() if len(q4) and len(q1) else np.nan,
                     "deficient_minus_supported_MAE": d.mean() - s.mean() if len(d) and len(s) else np.nan,
                     "remaining_Q1": len(q1), "remaining_Q4": len(q4),
                     "remaining_supported": len(s), "remaining_deficient": len(d)})
    return pd.DataFrame(rows)


def maybe_plot(curve, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for method in METHODS:
        g = curve.loc[curve.method.eq(method)].sort_values("actual_coverage")
        ax.plot(100 * g.actual_coverage, g.MAE, marker="o", label=method)
    ax.set(xlabel="Coverage of future experimental records (%)", ylabel="Selected-prediction MAE (lgkp)",
           title="Chronological test: prediction reliability vs coverage")
    ax.grid(True, alpha=0.20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="cutoff_2022_p50")
    p.add_argument("--bootstrap-draws", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--coverages", default="0.20,0.40,0.60,0.80,1.00")
    p.add_argument("--pred-bins", type=int, default=4)
    p.add_argument("--min-bin-records", type=int, default=5)
    p.add_argument("--min-bin-sources", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    if args.bootstrap_draws < 100 or args.pred_bins < 2 or args.min_bin_records < 1 or args.min_bin_sources < 1:
        p.error("Invalid bootstrap / prediction bin settings")
    fractions = parse_coverages(args.coverages)
    run = Path.cwd() / "chronological_runs" / args.run_name
    df, hist, fut_path, hist_path = load_data(run)
    out = args.out_dir if args.out_dir else run / "chronological_risk_coverage"
    out = out.resolve()
    if out in ((run / "evaluation").resolve(), (run / "training").resolve()):
        raise ValueError("Refusing to overwrite source evaluation/training directory")
    out.mkdir(parents=True, exist_ok=True)
    boot = ClusterBootstrap(df, args.bootstrap_draws, args.seed)
    curves, paired, auc = risk_coverage(df, boot, fractions)
    gaps = pd.DataFrame([
        simple_gap(df, boot, df.quadrant.eq("Q4"), df.quadrant.eq("Q1"),
                   "Q4_minus_Q1", "Positive = Q4 higher MAE"),
        simple_gap(df, boot, df.support_group.eq("model_deficient"),
                   df.support_group.eq("model_supported"),
                   "Q3+Q4_minus_Q1+Q2", "Positive = model-deficient higher MAE"),
    ])
    pred_bins, pred_gap = stratified_gap(df, boot, args.pred_bins,
                                         args.min_bin_records, args.min_bin_sources, False)
    if "alloy_family" in df:
        fam_bins, fam_gap = stratified_gap(df, boot, min(args.pred_bins, 3),
                                           args.min_bin_records, args.min_bin_sources, True)
        bins_df = pd.concat([pred_bins, fam_bins], ignore_index=True)
        pred_gap_df = pd.concat([pred_gap, fam_gap], ignore_index=True)
    else:
        bins_df, pred_gap_df = pred_bins, pred_gap
    loso = leave_one_source_out(df)
    # Save score-level table for exact source-specific traceback, but never change
    # model/evaluation inputs. No unrecorded scoring/threshold fitting on labels.
    files = {
        "risk_coverage.csv": curves,
        "risk_coverage_paired_comparisons.csv": paired,
        "risk_coverage_AURC.csv": auc,
        "future_state_error_gaps.csv": gaps,
        "predicted_lgkp_strata.csv": bins_df,
        "predicted_lgkp_adjusted_gaps.csv": pred_gap_df,
        "leave_one_source_out.csv": loso,
        "future_with_historical_percentiles.csv": df,
    }
    for name, table in files.items():
        table.to_csv(out / name, index=False, encoding="utf-8-sig")
    plotted = maybe_plot(curves, out / "risk_coverage.png")
    manifest = {
        "analysis_label": "Post-hoc exploratory; not independent confirmatory ranking evidence",
        "future_file": str(fut_path), "future_sha256": sha256(fut_path.read_bytes()).hexdigest(),
        "historical_file": str(hist_path), "historical_sha256": sha256(hist_path.read_bytes()).hexdigest(),
        "n_future": len(df), "n_future_sources": boot.all_source_count,
        "n_historical_reference": len(hist), "n_historical_sources": hist[SOURCE].nunique(),
        "reference_percentile": "ECDF using historical fitted-model LOO reference, side=right; low is supported",
        "joint_diagnostic_score": "max(MED_reference_percentile, PED_reference_percentile)",
        "risk_coverage_ranking": "fixed future prediction/evidence scores; rank before outcomes, outcomes for evaluation only",
        "prediction_only_caveat": "sorting lower predicted lgkp optimizes predicted oxidation kinetics, NOT forecast reliability",
        "intervals": "pointwise percentile intervals via paired source-cluster bootstrap, frozen selections; exploratory",
        "bootstrap_draws": args.bootstrap_draws, "bootstrap_seed": args.seed,
        "coverages": fractions, "prediction_bins": args.pred_bins,
        "minimum_stratum_group_records": args.min_bin_records,
        "minimum_stratum_group_sources": args.min_bin_sources,
        "plot_created": plotted,
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = []
    lines += ["=" * 100, "CHRONOLOGICAL PREDICTION RELIABILITY",
              "=" * 100,
              f"Future records: {len(df)}, independent sources: {boot.all_source_count}",
              f"Historical reference: {len(hist)} records, sources: {hist[SOURCE].nunique()}",
              f"Source bootstrap: {args.bootstrap_draws} draws, seed={args.seed}",
              "Joint_MAX = max(historical percentile(MED), historical percentile(PED)).",
              "Prediction_only means low predicted lgkp first; it is NOT an error-risk estimator.",
              "No retraining / no re-calibration on future outcomes / original Q1--Q4 unchanged.", "",
              "1) CHRONOLOGICAL EVIDENCE-STATE ERROR GAPS", gaps.to_string(index=False, float_format=lambda x: f"{x:.4f}"), "",
              "2) RISK--COVERAGE: SELECTED-SUBSET MAE (LOWER IS BETTER)",
              curves[["method", "target_coverage", "selected_records", "selected_sources", "MAE", "CI95_low", "CI95_high"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"), "",
              "3) PAIRED DIFFERENCES: COMPARATOR MAE - JOINT_MAX MAE (POSITIVE FAVORS JOINT)",
              paired[["target_coverage", "comparator", "changed_records", "changed_sources", "estimate", "CI95_low", "CI95_high"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"), "",
              "4) PREDICTION-VALUE-STRATIFIED MODEL-SUPPORT GAP (DEFICIENT - SUPPORTED)",
              pred_gap_df[["stratification", "eligible_strata", "estimate", "CI95_low", "CI95_high", "valid_draws"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"), "",
              "5) LEAVE-ONE-LITERATURE-SOURCE-OUT STABILITY"]
    for col in ("Q4_minus_Q1_MAE", "deficient_minus_supported_MAE"):
        valid = loso[col].dropna()
        lines.append(f"  {col}: {len(valid)} valid; min={fnum(valid.min())}, median={fnum(valid.median())}, max={fnum(valid.max())}; positive={(valid > 0).sum()}/{len(valid)}")
    lines += ["", "IMPORTANT: Joint_MAX is an additional exploratory ranking, not the Q-state rule.",
              "Intervals are pointwise, NOT corrected for testing many methods/coverages.",
              "Prediction-stratified comparisons do not isolate confounders or establish causality.",
              "These reliability metrics do NOT prove improvement in experimentally optimal alloy selection.",
              f"Results saved: {out}"]
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
