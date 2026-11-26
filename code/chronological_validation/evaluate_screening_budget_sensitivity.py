"""Outcome-blind Q4 -> Q1 opportunity audit and descriptive screening grid.

Run from the project root (the controlled working directory):

  python code/chronological_validation/evaluate_screening_budget_sensitivity.py
  python code/chronological_validation/evaluate_screening_budget_sensitivity.py --run-name cutoff_2022_p50

Uses ONLY the already-saved evaluation/future_predictions.csv and the original
selection algorithm from validate_chronological_holdout.py. Does NOT train models, reassign
Q states, adjust cutoffs, or modify the original run/evaluation files.

IMPORTANT: This is a *post hoc, exploratory* budget/tolerance sensitivity grid.
Showing all settings avoids choosing a winner after examining future outcomes.
Uncertainty intervals are intentionally not computed here; sparse swaps and
source dependence make inferential claims inappropriate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import math
import hashlib

import numpy as np
import pandas as pd

from validate_chronological_holdout import DEFAULT_POLICY, GROUP, SOURCE, candidate_pools, make_shortlists

HERE = Path(__file__).resolve().parent
DEFAULT_FRACTIONS = (0.10, 0.20, 0.25, 0.33, 0.40, 0.50)
DEFAULT_TOLERANCES = (0.05, 0.10, 0.20)
REQUIRED = {"sample_id", SOURCE, *GROUP, "quadrant", "y_pred", "y_true"}


def options(raw, option, limit, *, allow_zero=False):
    try:
        out = tuple(float(s.strip()) for s in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid {option}: {raw}") from exc
    if not out or not all(math.isfinite(v) and (0 <= v <= limit if allow_zero else 0 < v < limit) for v in out):
        raise ValueError(f"{option} must be comma-separated values {'[0, ' + str(limit) + ']' if allow_zero else '(0, ' + str(limit) + ')' }.")
    return tuple(sorted(set(out)))


def load_evaluation(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run validate_chronological_holdout.py evaluate first.")
    frame = pd.read_csv(path, dtype={"sample_id": str, SOURCE: str})
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"Missing columns from saved predictions: {sorted(missing)}")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Sample IDs must be complete and unique.")
    frame[SOURCE] = frame[SOURCE].str.strip().str.lower()
    if frame[SOURCE].isna().any() or frame[SOURCE].eq("").any():
        raise ValueError("Missing literature sources.")
    for col in ("y_pred", "y_true", "Temperature", "E_Time"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[["y_pred", "y_true", "Temperature", "E_Time"]].to_numpy()).all():
        raise ValueError("Non-finite predictions, outcomes or experimental conditions.")
    if not set(frame.quadrant).issubset({"Q1", "Q2", "Q3", "Q4"}):
        raise ValueError("Unexpected evidence-state label.")
    frame["abs_error"] = (frame.y_pred - frame.y_true).abs()
    frame["positive_optimism"] = (frame.y_true - frame.y_pred).clip(lower=0)
    return frame


def pool_opportunities(frame, policy):
    rows = []
    for group, pool, budget in candidate_pools(frame, policy):
        ordered = pool.sort_values(["y_pred", "sample_id"])
        selected = ordered.iloc[:budget]
        outside = ordered.iloc[budget:] if budget else ordered.iloc[:0]
        q4 = selected[selected.quadrant.eq("Q4")]
        q1 = outside[outside.quadrant.eq("Q1")]
        theoretical = min(len(q4), len(q1))
        possible_gaps = (
            q1.y_pred.to_numpy()[:, None] - q4.y_pred.to_numpy()[None, :]
            if len(q1) and len(q4) else np.empty((0, 0))
        )
        min_gap = float(possible_gaps.min()) if possible_gaps.size else np.nan
        reasons = []
        if budget == 0:
            reasons.append("ineligible_pool")
        else:
            if not len(q4):
                reasons.append("no_Q4_in_prediction_shortlist")
            if not len(q1):
                reasons.append("no_Q1_outside_shortlist")
            if theoretical and min_gap > policy["prediction_tolerance"] + 1e-12:
                reasons.append("prediction_gap_exceeds_tolerance")
        rows.append({
            "selection_fraction": policy["selection_fraction"],
            "prediction_tolerance": policy["prediction_tolerance"],
            **dict(zip(GROUP, group)),
            "pool_records": len(pool), "pool_sources": pool[SOURCE].nunique(),
            "selection_budget": budget,
            "Q1_in_pool": int((pool.quadrant == "Q1").sum()),
            "Q4_in_pool": int((pool.quadrant == "Q4").sum()),
            "Q4_in_prediction_shortlist": len(q4),
            "Q1_outside_shortlist": len(q1),
            "max_swaps_without_tolerance": theoretical,
            "min_required_prediction_gap": min_gap,
            "blocking_reason": ";".join(reasons) if reasons else "none",
        })
    return pd.DataFrame(rows)


def comparison_metric(frame, column):
    baseline = frame.loc[frame.in_prediction_only, column]
    evidence = frame.loc[frame.in_evidence_aware, column]
    return float(baseline.mean()) if len(baseline) else np.nan, float(evidence.mean()) if len(evidence) else np.nan


def evaluate_policy(frame, policy):
    # Reuse EXACTLY the original (frozen) decision algorithm, not a new one.
    shortlisted, swaps = make_shortlists(frame, policy)
    if len(shortlisted) != len(frame):
        raise RuntimeError("Original selection function returned a different candidate count.")
    if shortlisted.sample_id.duplicated().any():
        raise RuntimeError("Duplicate candidates in selection result.")
    pools = pool_opportunities(frame, policy)
    indexed = frame.set_index("sample_id")
    if not swaps.empty:
        for role in ("removed", "added"):
            for col in ("y_true", "abs_error", "positive_optimism", "quadrant"):
                swaps[f"{role}_{col}"] = swaps[f"{role}_sample_id"].map(indexed[col])
        swaps["same_literature_source"] = swaps.removed_source.eq(swaps.added_source)
        swaps["change_in_abs_error_aware_minus_baseline"] = (swaps.added_abs_error - swaps.removed_abs_error)
        swaps["change_in_measured_lgkp_aware_minus_baseline"] = (swaps.added_y_true - swaps.removed_y_true)
    else:
        swaps["same_literature_source"] = pd.Series(dtype="bool")
        swaps["change_in_abs_error_aware_minus_baseline"] = pd.Series(dtype="float64")
        swaps["change_in_measured_lgkp_aware_minus_baseline"] = pd.Series(dtype="float64")
    swaps.insert(0, "prediction_tolerance", policy["prediction_tolerance"])
    swaps.insert(0, "selection_fraction", policy["selection_fraction"])

    pool_swap_counts = (
        swaps.groupby(GROUP, dropna=False).size().rename("actual_swaps").reset_index()
        if not swaps.empty else pd.DataFrame(columns=[*GROUP, "actual_swaps"])
    )
    pools = pools.merge(pool_swap_counts, on=GROUP, how="left", validate="one_to_one")
    pools["actual_swaps"] = pd.to_numeric(pools.actual_swaps, errors="coerce").fillna(0).astype("int64")
    if int(pools.actual_swaps.sum()) != len(swaps):
        raise RuntimeError("Swap count and pool counts are inconsistent.")

    budget = int(shortlisted.in_prediction_only.sum())
    if budget != int(shortlisted.in_evidence_aware.sum()):
        raise RuntimeError("Unequal shortlist budgets.")
    selected = shortlisted.loc[shortlisted.in_prediction_only | shortlisted.in_evidence_aware]
    all_changed = selected.loc[selected.in_prediction_only.ne(selected.in_evidence_aware)]
    if len(all_changed) != 2 * len(swaps):
        raise RuntimeError("Mismatch in recorded changed selections.")

    summary = {
        "selection_fraction": policy["selection_fraction"],
        "prediction_tolerance": policy["prediction_tolerance"],
        "eligible_pools": int((pools.selection_budget > 0).sum()),
        "eligible_records": int(pools.loc[pools.selection_budget > 0, "pool_records"].sum()),
        "budget_per_method": budget,
        "Q4_in_baseline": int(pools.Q4_in_prediction_shortlist.sum()),
        "Q1_outside_baseline": int(pools.Q1_outside_shortlist.sum()),
        "max_swaps_without_tolerance": int(pools.max_swaps_without_tolerance.sum()),
        "actual_swaps": int(len(swaps)),
        "pools_with_swaps": int((pools.actual_swaps > 0).sum()),
        "changed_sources": int(all_changed[SOURCE].nunique()),
        "same_source_swaps": int(swaps.same_literature_source.sum()) if len(swaps) else 0,
        "mean_swap_prediction_gap": float(swaps.prediction_gap.mean()) if len(swaps) else np.nan,
    }
    for metric, column in (
        ("MAE", "abs_error"),
        ("measured_lgkp", "y_true"),
        ("predicted_lgkp", "y_pred"),
        ("positive_optimism", "positive_optimism"),
    ):
        base, aware = comparison_metric(shortlisted, column)
        summary[f"prediction_only_{metric}"] = base
        summary[f"evidence_aware_{metric}"] = aware
        summary[f"improvement_{metric}"] = base - aware
    return summary, pools, swaps


def verify_existing_summary(summary, existing_path):
    if not existing_path.is_file():
        return "Original screening_summary.csv not found; skipped baseline reproduction check."
    original = pd.read_csv(existing_path)
    check = {
        "MAE": "MAE", "mean_measured_lgkp": "measured_lgkp",
        "mean_positive_optimism": "positive_optimism",
        "mean_predicted_lgkp": "predicted_lgkp",
    }
    for old_name, new_name in check.items():
        row = original.loc[original.metric.eq(old_name)]
        if len(row) != 1:
            raise ValueError(f"Cannot reproduce original metric {old_name}; malformed summary.")
        row = row.iloc[0]
        for old_col, current in (("prediction_only", f"prediction_only_{new_name}"),
                                 ("evidence_aware", f"evidence_aware_{new_name}")):
            if not np.isclose(float(row[old_col]), summary[current], atol=1e-8, rtol=1e-8):
                raise ValueError(f"Original screening result mismatch for {old_name} {old_col}: "
                                 f"original={row[old_col]}, grid={summary[current]}")
        if int(row.swaps) != summary["actual_swaps"] or int(row.budget_per_method) != summary["budget_per_method"]:
            raise ValueError(f"Original screening swap/budget mismatch for {old_name}.")
    return "Original 25% / 0.10 screening results reproduced exactly."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="cutoff_2022_p50")
    parser.add_argument("--fractions", default=",".join(map(str, DEFAULT_FRACTIONS)))
    parser.add_argument("--tolerances", default=",".join(map(str, DEFAULT_TOLERANCES)))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    fractions = options(args.fractions, "fractions", 1)
    tolerances = options(args.tolerances, "tolerances", float("inf"), allow_zero=True)
    run = Path.cwd() / "chronological_runs" / args.run_name
    input_path = run / "evaluation" / "future_predictions.csv"
    data = load_evaluation(input_path)
    if len(data) != 169:
        raise ValueError(f"Expected 169 frozen future experimental records, got {len(data)}.")

    manifest_path = run / "training" / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
    base_policy = dict(DEFAULT_POLICY)
    base_policy.update(manifest.get("screening_policy", {}))
    # Only *two* policy dimensions are varied; pool min sizes remain those
    # fixed during training. Outcomes never influence any selection choice.
    min_records = int(base_policy["min_pool_records"])
    min_sources = int(base_policy["min_pool_sources"])
    if min_records < 2 or min_sources < 2:
        raise ValueError("Invalid source/record minimum in saved policy")
    out = args.out_dir if args.out_dir is not None else run / "screening_budget_sensitivity"
    out = out.resolve()
    if out == (run / "evaluation").resolve() or out == (run / "training").resolve():
        raise ValueError("Refusing to save diagnostic output into the frozen training or evaluation folder.")
    out.mkdir(parents=True, exist_ok=True)

    summary_rows, pool_tables, swap_tables = [], [], []
    for fraction in fractions:
        for tol in tolerances:
            policy = {**base_policy, "selection_fraction": fraction, "prediction_tolerance": tol}
            row, pools, swaps = evaluate_policy(data, policy)
            summary_rows.append(row)
            pool_tables.append(pools)
            swap_tables.append(swaps)
    summary = pd.DataFrame(summary_rows)
    all_pools = pd.concat(pool_tables, ignore_index=True)
    nonempty_swaps = [table for table in swap_tables if not table.empty]
    all_swaps = (pd.concat(nonempty_swaps, ignore_index=True) if nonempty_swaps
                 else swap_tables[0].iloc[:0].copy())

    original_frac = float(base_policy["selection_fraction"])
    original_tol = float(base_policy["prediction_tolerance"])
    verified = "Saved original configuration not included in grid; reproduction check skipped."
    selected = summary.loc[np.isclose(summary.selection_fraction, original_frac) &
                           np.isclose(summary.prediction_tolerance, original_tol)]
    if len(selected) == 1:
        verified = verify_existing_summary(selected.iloc[0].to_dict(), run / "evaluation" / "screening_summary.csv")
    if len(selected) > 1:
        raise RuntimeError("Duplicate original config in diagnostic grid.")

    summary_path = out / "grid_summary.csv"
    pools_path = out / "pool_opportunities.csv"
    swaps_path = out / "swap_details.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    all_pools.to_csv(pools_path, index=False, encoding="utf-8-sig")
    all_swaps.to_csv(swaps_path, index=False, encoding="utf-8-sig")
    notes = [
        "EXPLORATORY budget/tolerance sensitivity analysis — do not select a favored setting post hoc.",
        f"Input: {input_path}",
        f"Input SHA256: {hashlib.sha256(input_path.read_bytes()).hexdigest()}",
        f"Future records={len(data)}, sources={data[SOURCE].nunique()}",
        f"Fixed grouping: {GROUP}; pool minima: N>={min_records}, sources>={min_sources}.",
        f"Fractions: {fractions}; tolerances: {tolerances}.",
        "Selection rule: original validate_chronological_holdout.make_shortlists; baseline predicted lowest lgkp;",
        "evidence-aware Q4->Q1 substitutions only if predicted lgkp gap <= tolerance.",
        "Outcome values used only for evaluation, not for candidate selection.",
        "Improvement = prediction_only - evidence_aware; positive MAE or positive optimism improvement",
        "means lower error/overoptimism with evidence-aware; positive measured_lgkp improvement",
        "means more negative (smaller) measured lgkp with evidence-aware.",
        "No source-cluster confidence intervals computed for this exploratory grid.",
        f"Baseline check: {verified}",
    ]
    (out / "README_diagnostic.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("=" * 104)
    print("SCREENING BUDGET AND TOLERANCE SENSITIVITY")
    print("=" * 104)
    print(f"Records={len(data)}, sources={data[SOURCE].nunique()}; fixed pools: {GROUP}")
    print(f"Min pool records={min_records}, sources={min_sources}; Q1={int((data.quadrant=='Q1').sum())}, Q4={int((data.quadrant=='Q4').sum())}")
    print(verified)
    print("\n1) OPPORTUNITY + SCREENING GRID (all combinations, not best-of)")
    compact_cols = [
        "selection_fraction", "prediction_tolerance", "budget_per_method", "Q4_in_baseline",
        "Q1_outside_baseline", "max_swaps_without_tolerance", "actual_swaps",
        "pools_with_swaps", "changed_sources", "same_source_swaps",
        "prediction_only_MAE", "evidence_aware_MAE", "improvement_MAE",
        "prediction_only_measured_lgkp", "evidence_aware_measured_lgkp", "improvement_measured_lgkp",
    ]
    print(summary[compact_cols].round(4).to_string(index=False))
    print("\n2) WHY ARE SWAPS RARE? — ORIGINAL 25% BUDGET, 0.10 TOLERANCE")
    original_pool = all_pools.loc[np.isclose(all_pools.selection_fraction, original_frac) &
                                  np.isclose(all_pools.prediction_tolerance, original_tol)]
    if len(original_pool):
        cols = [*GROUP, "pool_records", "pool_sources", "selection_budget",
                "Q4_in_prediction_shortlist", "Q1_outside_shortlist", "max_swaps_without_tolerance",
                "min_required_prediction_gap", "actual_swaps", "blocking_reason"]
        print(original_pool.loc[original_pool.selection_budget.gt(0), cols].round(4).to_string(index=False))
    else:
        print("Original configuration was not included in the grid.")
    print("\n3) INTERPRETATION GUIDANCE")
    print("- If Q1_outside_baseline is zero, increasing tolerance cannot create Q1 candidates.")
    print("- If max_swaps_without_tolerance > actual_swaps, predicted gap blocks some potential swaps.")
    print("- >=3 changed sources does NOT alone guarantee statistically persuasive evidence.")
    print("- Treat all grid outcome comparisons as descriptive/exploratory, not an independent confirmatory test.")
    print("\nSaved:")
    for path in [summary_path, pools_path, swaps_path, out / "README_diagnostic.txt"]:
        print(" ", path)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"ERROR: {exc}")
