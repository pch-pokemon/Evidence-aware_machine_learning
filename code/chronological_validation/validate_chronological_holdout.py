"""Retrospective chronological validation and evidence-aware screening.

Run with the Python environment used for the project's nested CV:
    python code/chronological_validation/validate_chronological_holdout.py audit
    python code/chronological_validation/validate_chronological_holdout.py train
    python code/chronological_validation/validate_chronological_holdout.py evaluate

Audit reads the source-fold test CSVs and prints the split and pool eligibility.
Train runs historical-only Optuna and saves one XGBoost model. Evaluate loads
that saved model, computes MED/PED, fixes the shortlists, and joins outcomes.
Outputs are under this folder's runs/<run-name> directory. Train and
evaluate replace their existing outputs after successful execution.
A replacement model invalidates the previous evaluation for its run.
Use --run-name before the subcommand to keep a separate run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()
SPLIT_DIR = ROOT / "splits" / "source_kfold"
NESTED_CONFIG = ROOT / "models" / "xgb_source_nested_kfold" / "run_config.json"
PRIOR_FILE = ROOT / "models" / "fixed_evidence_weights" / "final_fixed_evidence_weights.json"
CUTOFF = "2022-12-31"
DEFAULT_RUN_NAME = "cutoff_2022_p50"
SOURCE = "source_key"
TARGET = "lgkp"
GROUP = ["alloy_family", "Temperature", "E_Time"]
STATES = ("Q1", "Q2", "Q3", "Q4")
ID_COLS = ["sample_id", SOURCE, "alloy_id", "Year", "availability_year"]

# The issue year is 2023 for these two DOI sources. Springer lists the
# version of record on the dates below. Other dataset years are used only
# for their side of the cutoff after checking the 79 sources from 2022 on.
# Publisher records: https://link.springer.com/article/<DOI>
ONLINE_DATE_OVERRIDES = {
    "10.1007/s11661-022-06860-6": "2022-11-08",
    "10.1007/s11661-022-06896-8": "2022-11-23",
}
DEFAULT_POLICY = {
    "prediction_tolerance": 0.10,
    "selection_fraction": 0.25,
    "min_pool_records": 5,
    "min_pool_sources": 3,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def write_csv(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def fingerprint(frame, columns):
    return sha256(frame.loc[:, columns].to_csv(index=False).encode("utf-8")).hexdigest()


def load_split(features):
    frames = [pd.read_csv(SPLIT_DIR / f"fold{i}_test.csv", dtype={SOURCE: str, "sample_id": str})
              for i in range(1, 6)]
    data = pd.concat(frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    required = (set(ID_COLS) - {"availability_year"}) | set(features) | {TARGET}
    if required.difference(data.columns):
        raise ValueError(f"Missing split columns: {sorted(required.difference(data.columns))}")
    if len(data) != 613 or data.sample_id.isna().any() or data.sample_id.duplicated().any():
        raise ValueError("Expected 613 distinct sample IDs from the five frozen outer test folds.")
    data[SOURCE] = data[SOURCE].str.strip().str.lower()
    if data[SOURCE].isna().any() or data[SOURCE].eq("").any() or data[SOURCE].nunique() != 163:
        raise ValueError("Expected 163 nonempty, distinct literature sources.")
    if not np.isfinite(data[[*features, TARGET, "Year"]].to_numpy(dtype=float)).all():
        raise ValueError("Input features, target and publication years must be finite numbers.")
    if data.groupby(SOURCE).Year.nunique().ne(1).any():
        raise ValueError("Resolve inconsistent publication years for the same source first.")
    if set(ONLINE_DATE_OVERRIDES).difference(data[SOURCE]):
        raise ValueError("The frozen source manifest is missing a date-audited DOI.")

    source_table = (data.groupby(SOURCE, as_index=False)
                    .agg(Year=("Year", "first"), records=("sample_id", "size")))
    source_table["verified_online_date_override"] = source_table[SOURCE].map(ONLINE_DATE_OVERRIDES)
    source_table["availability_year"] = [
        int(date[:4]) if pd.notna(date) else int(year)
        for year, date in zip(source_table.Year, source_table.verified_online_date_override)
    ]
    source_table["split"] = np.where(source_table.availability_year.le(2022), "historical", "future")
    data = data.merge(source_table[[SOURCE, "availability_year", "split"]], on=SOURCE,
                      validate="many_to_one", sort=False)
    historical = data.loc[data.split.eq("historical")].reset_index(drop=True)
    future = data.loc[data.split.eq("future")].reset_index(drop=True)
    if (len(historical), historical[SOURCE].nunique(), len(future), future[SOURCE].nunique()) != (
        444, 106, 169, 57
    ):
        raise ValueError("Split counts differ from the audited 2022 cutoff; check source provenance.")
    if set(historical[SOURCE]) & set(future[SOURCE]):
        raise ValueError("A literature source straddles the chronological split.")

    future["same_model_features_in_history"] = (
        pd.MultiIndex.from_frame(future[features]).isin(pd.MultiIndex.from_frame(historical[features]))
    )
    future["alloy_family"] = np.select(
        [future.x_Ni.gt(future.x_Co), future.x_Co.gt(future.x_Ni)],
        ["Ni-dominant", "Co-dominant"], default="Ni-Co-tie",
    )
    return historical, future, source_table


def candidate_pools(frame, policy):
    for group, pool in frame.groupby(GROUP, sort=True):
        eligible = (
            len(pool) >= policy["min_pool_records"]
            and pool[SOURCE].nunique() >= policy["min_pool_sources"]
        )
        budget = math.ceil(len(pool) * policy["selection_fraction"]) if eligible else 0
        if budget >= len(pool):
            budget = 0
        yield group, pool, budget


def report_split(historical, future):
    print(f"Cutoff: {CUTOFF} (inclusive) | DOI overlap: 0")
    print(f"Historical: {len(historical)} records / {historical[SOURCE].nunique()} sources")
    print(f"Future:     {len(future)} records / {future[SOURCE].nunique()} sources")
    print("\nEligible pools under the default screening policy (family, C, h | records, sources, K):")
    for group, pool, budget in candidate_pools(future, DEFAULT_POLICY):
        if budget:
            print(f"  {group} | {len(pool)}, {pool[SOURCE].nunique()}, {budget}")


@contextmanager
def run_lock(run_dir):
    """Serialize training and evaluation for the same named run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".run.lock").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is using this run: {run_dir}") from exc
        # Closing the handle releases the lock, including on failure.
        yield


def publish_artifacts(staged, destination, *, invalidate=()):
    """Publish a prepared directory and restore previous outputs on failure."""
    run_dir = destination.parent.resolve()
    targets = [destination, *invalidate]
    backups = []
    for target in targets:
        if (target.parent.resolve() != run_dir or target.is_symlink()
                or target.resolve() != run_dir / target.name
                or (target.exists() and not target.is_dir())):
            raise RuntimeError(f"Invalid artifact directory: {target}")
    try:
        for target in targets:
            if target.exists():
                backup = staged.parent / f"previous-{target.name}"
                target.rename(backup)
                backups.append((target, backup))
        staged.rename(destination)
    except BaseException:
        for target, backup in reversed(backups):
            backup.rename(target)
        raise


def train_model(args, run_dir, historical, future, source_table, features, policy):
    import optuna
    import sklearn
    import xgboost
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBRegressor
    from train_source_aware_xgboost import (
        FEATURE_COLS, SEARCH_SPACE, SEED, final_model_params, proposed_params,
    )

    if list(FEATURE_COLS) != features:
        raise ValueError("The frozen feature order differs from the nested CV training code.")
    published_model_dir = run_dir / "training"
    published_evaluation_dir = run_dir / "evaluation"
    existing = [p for p in (published_model_dir, published_evaluation_dir) if p.exists() or p.is_symlink()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Existing run artifacts: " + ", ".join(str(p) for p in existing)
            + ". Remove --no-overwrite to replace this run, or choose a new --run-name."
        )
    for p in existing:
        if p.is_symlink() or p.resolve() != run_dir.resolve() / p.name or not p.is_dir():
            raise RuntimeError(f"Invalid artifact directory: {p}")
    # Train in a staging folder. An interrupted Optuna run leaves the previous
    # successful training/evaluation untouched; publish only after model save.
    with TemporaryDirectory(prefix="training-staging-", dir=run_dir) as scratch:
        model_dir = Path(scratch) / "training"
        model_dir.mkdir()
        main_weights = read_json(PRIOR_FILE)["main_weights"]
        x = historical[features].to_numpy(dtype=float)
        y = historical[TARGET].to_numpy(dtype=float)
        groups = historical[SOURCE].to_numpy()
        splits = list(GroupKFold(n_splits=4).split(x, y, groups))
        split_rows = []
        for fold, (tr, va) in enumerate(splits, 1):
            if set(groups[tr]) & set(groups[va]):
                raise RuntimeError(f"Source overlap in historical CV fold {fold}.")
            split_rows.append(historical.iloc[va][["sample_id", SOURCE]].assign(inner_fold=fold))
        write_csv(model_dir / "historical_inner_folds.csv", pd.concat(split_rows))
        write_csv(model_dir / "source_manifest.csv", source_table)

        manifest = {
            "cutoff_inclusive": CUTOFF,
            "online_date_overrides": ONLINE_DATE_OVERRIDES,
            "features": features,
            "historical_rows": len(historical),
            "historical_sha256": fingerprint(historical, [*ID_COLS, *features, TARGET]),
            "future_rows": len(future),
            "future_predictors_sha256": fingerprint(future, [*ID_COLS, *features]),
            "inner_folds": 4,
            "inner_objective": "Unweighted mean of four source-grouped validation R2 scores",
            "requested_trials": args.trials,
            "xgboost_threads": args.n_jobs,
            "versions": {"xgboost": xgboost.__version__, "optuna": optuna.__version__,
                         "scikit_learn": sklearn.__version__},
            "seed": SEED,
            "search_space": SEARCH_SPACE,
            "screening_policy": policy,
            "main_weights": main_weights,
        }
        write_json(model_dir / "protocol.json", manifest)

        def objective(trial):
            params = proposed_params(trial, args.n_jobs)
            scores = []
            for fold, (tr, va) in enumerate(splits, 1):
                model = XGBRegressor(**params)
                model.fit(x[tr], y[tr])
                score = float(r2_score(y[va], model.predict(x[va])))
                trial.set_user_attr(f"inner{fold}_R2", score)
                scores.append(score)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        try:
            study.optimize(objective, n_trials=args.trials)
        finally:
            write_csv(model_dir / "optuna_trials.csv", study.trials_dataframe())

        model = XGBRegressor(**final_model_params(study.best_params, args.n_jobs))
        model.fit(x, y)
        model_path = model_dir / "historical_xgb.ubj"
        model.save_model(str(model_path))
        manifest.update(best_params=study.best_params, best_inner_R2=float(study.best_value),
                        model_sha256=sha256(model_path.read_bytes()).hexdigest())
        write_json(model_dir / "training_manifest.json", manifest)
        # Publish only after a successful model fit and complete manifest write.
        # Existing evaluation would refer to the OLD model, so invalidate it too.
        publish_artifacts(model_dir, published_model_dir, invalidate=(published_evaluation_dir,))
        print(f"Historical model and training metadata saved to {published_model_dir}")


def score_future(model, historical, future, features, main_weights):
    from compute_oof_evidence import CFG, model_inputs, protocol_gain
    from evidence_core import (
        assign_quadrants, build_med_from_training_reference, compute_ped_external,
        compute_ped_training_loo, derive_training_thresholds, ensure_2d_leaf_matrix,
        physical_standardization_to_dict, transform_ped_for_threshold,
    )

    tr_x = model_inputs(model, historical, features)
    qu_x = model_inputs(model, future, features)
    tr_leaf = ensure_2d_leaf_matrix(model.apply(tr_x))
    qu_leaf = ensure_2d_leaf_matrix(model.apply(qu_x))
    tr_med, qu_med, _, _, norm = build_med_from_training_reference(tr_leaf, qu_leaf, CFG)
    protocol_weights = protocol_gain(model, features)
    tr_ped_raw = compute_ped_training_loo(historical, main_weights, protocol_weights, CFG)
    qu_ped_raw, _, _, physical = compute_ped_external(historical, future, main_weights,
                                                       protocol_weights, CFG)
    tr_ped = transform_ped_for_threshold(tr_ped_raw, CFG)
    qu_ped = transform_ped_for_threshold(qu_ped_raw, CFG)
    predicted = np.asarray(model.predict(qu_x), dtype=float)
    if not all(np.isfinite(x).all() for x in (predicted, tr_med.MED, qu_med.MED, tr_ped, qu_ped)):
        raise ValueError("Nonfinite historical calibration values or future scores.")

    thresholds = derive_training_thresholds(tr_med.MED, tr_ped, (50,))
    p50 = thresholds.iloc[0]
    score_cols = [*ID_COLS, *GROUP, "same_model_features_in_history"]
    scored = pd.concat([future[score_cols].reset_index(drop=True), qu_med], axis=1)
    scored["y_pred"] = predicted
    scored["PED_raw"] = qu_ped_raw
    scored["PED_log1p"] = np.log1p(qu_ped_raw)
    scored["quadrant"] = assign_quadrants(scored.MED, qu_ped, p50.MED_threshold, p50.PED_threshold)
    reference = pd.concat([historical[ID_COLS], tr_med], axis=1)
    reference["PED_raw"] = tr_ped_raw
    reference["PED_log1p"] = np.log1p(tr_ped_raw)
    calibration = {
        "config": CFG.to_dict(),
        "P50_MED": float(p50.MED_threshold),
        "P50_log1p_PED": float(p50.PED_threshold),
        "reference_method": "Historical fitted-model training-reference LOO (same as main MED/PED)",
        "historical_records": len(historical),
        "protocol_weights": protocol_weights,
        "physical_standardization": physical_standardization_to_dict(physical),
        "MED_normalization": norm,
    }
    return scored, reference, calibration


def make_shortlists(scores, policy):
    """Pick both lists using only recorded conditions, predictions, and Q states."""
    plans = []
    swaps = []
    swap_cols = [*GROUP, "removed_sample_id", "added_sample_id", "removed_source",
                 "added_source", "removed_y_pred", "added_y_pred", "prediction_gap"]
    for group, pool, budget in candidate_pools(scores, policy):
        ordered = pool.sort_values(["y_pred", "sample_id"])
        baseline = ordered.iloc[:budget]
        chosen = set(baseline.index)
        q4 = baseline.loc[baseline.quadrant.eq("Q4")]
        q1 = ordered.iloc[budget:].loc[lambda rows: rows.quadrant.eq("Q1")]
        next_q1 = iter(q1.itertuples())
        incoming = next(next_q1, None) if budget else None
        # Both lists are in prediction order. Every outside prediction is
        # >= each baseline prediction, so the earliest feasible pair can be
        # consumed without reducing the number of one-to-one swaps.
        for outgoing in q4.itertuples():
            if incoming is None:
                break
            gap = float(incoming.y_pred - outgoing.y_pred)
            if gap > policy["prediction_tolerance"] + 1e-12:
                continue
            chosen.remove(outgoing.Index)
            chosen.add(incoming.Index)
            swaps.append((*group, outgoing.sample_id, incoming.sample_id, outgoing.source_key,
                          incoming.source_key, outgoing.y_pred, incoming.y_pred, gap))
            incoming = next(next_q1, None)
        if len(chosen) != budget:
            raise RuntimeError("The two shortlists have different budgets.")
        plans.append(ordered.assign(selection_budget=budget,
                                    in_prediction_only=ordered.index.isin(baseline.index),
                                    in_evidence_aware=ordered.index.isin(chosen)))
    return pd.concat(plans, ignore_index=True), pd.DataFrame(swaps, columns=swap_cols)


def source_draws(frame, n_draws, seed=42):
    sources, codes = np.unique(frame[SOURCE].to_numpy(dtype=str), return_inverse=True)
    count = len(sources)
    draws = np.random.default_rng(seed).multinomial(count, np.ones(count) / count, size=n_draws)
    weights = {source: draws[:, code] for code, source in enumerate(sources)}
    return weights


def bootstrap_mean(rows, column, weights, n_draws):
    if rows.empty:
        return np.full(n_draws, np.nan)
    grouped = rows.groupby(SOURCE)[column].agg(["sum", "size"])
    source_weights = np.column_stack([weights[name] for name in grouped.index])
    numerator = source_weights @ grouped["sum"].to_numpy()
    denominator = source_weights @ grouped["size"].to_numpy()
    return np.divide(numerator, denominator, out=np.full(n_draws, np.nan),
                     where=denominator > 0)


def interval(point, samples, informative_sources):
    valid = samples[np.isfinite(samples)]
    low, high = (np.percentile(valid, (2.5, 97.5))
                 if informative_sources >= 3 and len(valid) >= max(100, len(samples) / 2)
                 else (np.nan, np.nan))
    return {"estimate": point, "CI95_low": low, "CI95_high": high,
            "valid_bootstrap_draws": len(valid), "informative_sources": informative_sources}


def summarize_states(frame, weights, n_draws):
    stats = []
    for state in STATES:
        rows = frame.loc[frame.quadrant.eq(state)]
        draws = bootstrap_mean(rows, "abs_error", weights, n_draws)
        stats.append({"quadrant": state, "records": len(rows), "source_count": rows[SOURCE].nunique(),
                      **interval(rows.abs_error.mean(), draws, rows[SOURCE].nunique())})
    q1 = frame.loc[frame.quadrant.eq("Q1")]
    q4 = frame.loc[frame.quadrant.eq("Q4")]
    delta = (bootstrap_mean(q4, "abs_error", weights, n_draws)
             - bootstrap_mean(q1, "abs_error", weights, n_draws))
    gap = interval(q4.abs_error.mean() - q1.abs_error.mean(), delta,
                   min(q1[SOURCE].nunique(), q4[SOURCE].nunique()))
    return pd.DataFrame(stats).rename(columns={"estimate": "MAE"}), pd.DataFrame([gap])


def summarize_screening(frame, weights, n_draws):
    group_rows = []
    metrics = {"MAE": "abs_error", "mean_measured_lgkp": "y_true",
               "mean_positive_optimism": "positive_optimism", "mean_predicted_lgkp": "y_pred"}
    pooled_draws = {metric: np.zeros(n_draws) for metric in metrics}
    pooled_values = {metric: np.zeros(2) for metric in metrics}
    total_budget = 0
    for group, pool in frame.groupby(GROUP, sort=True):
        baseline = pool.loc[pool.in_prediction_only]
        evidence = pool.loc[pool.in_evidence_aware]
        budget = len(baseline)
        changed = pool.loc[pool.in_prediction_only != pool.in_evidence_aware]
        row = dict(zip(GROUP, group))
        row.update(records=len(pool), sources=pool[SOURCE].nunique(), budget_per_method=budget,
                   swaps=len(changed) // 2, sources_in_swaps=changed[SOURCE].nunique())
        for metric, column in metrics.items():
            a = baseline[column].mean()
            b = evidence[column].mean()
            row[f"prediction_only_{metric}"] = a
            row[f"evidence_aware_{metric}"] = b
            if budget:
                pooled_values[metric] += budget * np.array([a, b])
                if not changed.empty:
                    pooled_draws[metric] += budget * (
                        bootstrap_mean(baseline, column, weights, n_draws)
                        - bootstrap_mean(evidence, column, weights, n_draws)
                    )
        group_rows.append(row)
        total_budget += budget

    changed = frame.loc[frame.in_prediction_only != frame.in_evidence_aware]
    summary = []
    for metric in metrics:
        a, b = pooled_values[metric] / total_budget if total_budget else [np.nan, np.nan]
        n_changed_sources = changed[SOURCE].nunique()
        draws = (pooled_draws[metric] / total_budget if total_budget and n_changed_sources
                 else np.full(n_draws, np.nan))
        summary.append({"metric": metric, "budget_per_method": total_budget,
                        "eligible_groups": sum(row["budget_per_method"] > 0 for row in group_rows),
                        "swaps": len(changed) // 2, "prediction_only": a, "evidence_aware": b,
                        **interval(a - b, draws, n_changed_sources)})
    return pd.DataFrame(group_rows), pd.DataFrame(summary)


def evaluate_model(args, run_dir, historical, future, features):
    model_dir = run_dir / "training"
    model_path = model_dir / "historical_xgb.ubj"
    manifest_path = model_dir / "training_manifest.json"
    if not manifest_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Completed training artifacts are missing in {model_dir}. "
                                "Run the train stage first with the same --run-name.")
    manifest = read_json(manifest_path)
    if (manifest["cutoff_inclusive"] != CUTOFF
            or manifest["features"] != features
            or manifest["historical_sha256"] != fingerprint(historical, [*ID_COLS, *features, TARGET])
            or manifest["future_predictors_sha256"] != fingerprint(future, [*ID_COLS, *features])
            or manifest["model_sha256"] != sha256(model_path.read_bytes()).hexdigest()):
        raise RuntimeError("Historical model, saved metadata, or their data snapshot differ from training.")

    out = run_dir / "evaluation"
    if out.exists() or out.is_symlink():
        if not args.overwrite:
            raise FileExistsError(
                f"Evaluation already exists: {out}. Remove --no-overwrite to replace it."
            )
        if out.is_symlink() or out.resolve() != run_dir.resolve() / out.name or not out.is_dir():
            raise RuntimeError(f"Invalid artifact directory: {out}")

    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(str(model_path))
    predictors = future.loc[:, [*ID_COLS, *features, "alloy_family",
                                "same_model_features_in_history"]]
    scores, reference, calibration = score_future(
        model, historical, predictors, features, manifest["main_weights"]
    )
    shortlists, swaps = make_shortlists(scores, manifest["screening_policy"])

    # Outcome values are used only from this point onward.
    measured = future[["sample_id", TARGET]].rename(columns={TARGET: "y_true"})
    results = shortlists.merge(measured, on="sample_id", validate="one_to_one", sort=False)
    results["error"] = results.y_pred - results.y_true
    results["abs_error"] = results.error.abs()
    results["positive_optimism"] = (-results.error).clip(lower=0)
    n_draws = args.bootstrap_draws
    weights = source_draws(results, n_draws)
    states, state_gap = summarize_states(results, weights, n_draws)
    by_group, screening = summarize_screening(results, weights, n_draws)
    indexed = results.set_index("sample_id")
    for label in ("removed", "added"):
        ids = swaps[f"{label}_sample_id"]
        swaps[f"{label}_y_true"] = ids.map(indexed.y_true)
        swaps[f"{label}_abs_error"] = ids.map(indexed.abs_error)

    with TemporaryDirectory(prefix="evaluation-", dir=run_dir) as scratch:
        pending = Path(scratch) / "results"
        pending.mkdir()
        for name, frame in {
            "historical_reference": reference,
            "future_predictions": results,
            "state_error_summary": states,
            "Q4_minus_Q1_MAE": state_gap,
            "candidate_swaps": swaps,
            "screening_by_group": by_group,
            "screening_summary": screening,
        }.items():
            write_csv(pending / f"{name}.csv", frame)
        write_json(pending / "historical_evidence_calibration.json", calibration)
        write_json(pending / "evaluation_manifest.json", {
            "cutoff_inclusive": CUTOFF,
            "future_outcomes_sha256": fingerprint(future, ["sample_id", TARGET]),
            "screening_policy": manifest["screening_policy"],
            "swap_order": "Ascending predicted lgkp and sample_id, one-to-one",
            "bootstrap_draws": n_draws,
            "bootstrap_seed": 42,
            "screening_difference": "Prediction-only minus evidence-aware",
            "screening_pooling": "Within-group differences weighted by the observed group budget",
            "bootstrap": "Paired source resampling conditional on frozen models and selections",
            "bootstrap_CI_reporting": "At least three informative sources and half of draws valid",
        })
        # Keep any existing evaluation until this replacement is fully prepared.
        publish_artifacts(pending, out)
    print(states.to_string(index=False))
    print(screening.to_string(index=False))
    print(f"Evaluation saved to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Name of the run directory.")
    stages = parser.add_subparsers(dest="stage", required=True)
    stages.add_parser("audit", help="Check input records and chronological source separation")
    train = stages.add_parser("train", help="Tune and fit one historical-only XGBoost model")
    train.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                       help="Replace training and invalidate previous evaluation (default: yes)")
    train.add_argument("--trials", type=int, default=100)
    train.add_argument("--n-jobs", type=int, default=8)
    train.add_argument("--selection-fraction", type=float, default=DEFAULT_POLICY["selection_fraction"])
    train.add_argument("--min-pool-records", type=int, default=DEFAULT_POLICY["min_pool_records"])
    train.add_argument("--min-pool-sources", type=int, default=DEFAULT_POLICY["min_pool_sources"])
    train.add_argument("--prediction-tolerance", type=float, default=DEFAULT_POLICY["prediction_tolerance"])
    evaluate = stages.add_parser("evaluate", help="Score future records and compare selection rules")
    evaluate.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True,
                          help="Replace existing evaluation files (default: yes)")
    evaluate.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name):
        parser.error("--run-name must contain only letters, digits, underscores and hyphens.")
    if args.stage == "train" and (
        args.trials < 1 or args.n_jobs < 1 or args.min_pool_records < 2 or args.min_pool_sources < 2
        or not 0 < args.selection_fraction < 1
        or not math.isfinite(args.prediction_tolerance) or args.prediction_tolerance < 0
    ):
        parser.error("Invalid model or screening policy settings.")
    if args.stage == "evaluate" and args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100.")

    sys.path[:0] = [str(HERE.parent), str(HERE.parent / "nested_cv"), str(ROOT)]
    features = read_json(NESTED_CONFIG)["feature_cols"]
    historical, future, source_table = load_split(features)
    report_split(historical, future)
    run_dir = Path.cwd() / "chronological_runs" / args.run_name
    if args.stage == "train":
        print(f"Training run: {run_dir}", flush=True)
        policy = {key: getattr(args, key) for key in DEFAULT_POLICY}
        with run_lock(run_dir):
            train_model(args, run_dir, historical, future, source_table, features, policy)
    elif args.stage == "evaluate":
        print(f"Evaluation run: {run_dir}", flush=True)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run not found: {run_dir}. Run train first.")
        with run_lock(run_dir):
            evaluate_model(args, run_dir, historical, future, features)


if __name__ == "__main__":
    main()

