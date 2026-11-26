"""Run the research scripts in a separate author-data workspace."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parent
PRIORS = str(PACKAGE / "configs/physical_priors.json")

STAGES = {
    "nested-source": [("nested_cv/train_source_aware_xgboost.py",)],
    "nested-random": [("nested_cv/train_random_xgboost.py",)],
    "benchmark": [("nested_cv/benchmark_regression_models.py", "--split", "both")],
    "nested-evidence": [
        ("nested_cv/compute_oof_evidence.py", "--main-weights-json", PRIORS,
         "--skip-diagnostic-check", "--no-excel"),
    ],
    "nested-figures": [("figures/plot_evidence_error_distributions.py",)],
    "sensitivity": [
        ("sensitivity_and_ablation/evaluate_evidence_weights.py", "--main-weights-json", PRIORS, "--no-excel"),
        ("sensitivity_and_ablation/evaluate_neighborhood_ablation.py", "--main-weights-json", PRIORS, "--no-plot"),
        ("sensitivity_and_ablation/fit_cross_fitted_med_reference.py",),
        ("sensitivity_and_ablation/evaluate_med_calibration.py", "--no-excel"),
    ],
    "q4-coverage": [("sensitivity_and_ablation/summarize_q4_coverage.py",)],
    "fitting-quality": [("sensitivity_and_ablation/evaluate_fitting_quality.py",)],
    "temporal": [
        ("temporal_evolution/fit_historical_fold_models.py", "--no-excel"),
        ("temporal_evolution/compute_temporal_evidence.py", "--main-weights-json", PRIORS, "--no-excel"),
        ("temporal_evolution/summarize_temporal_transitions.py", "--no-excel", "--trace-sample-ids", "s279", "s125"),
    ],
    "temporal-figures": [
        ("temporal_evolution/plot_spatial_and_temporal_evidence.py",),
        ("temporal_evolution/plot_temporal_cases.py", "--no-excel"),
    ],
    "chronological-validation": [
        ("chronological_validation/validate_chronological_holdout.py", "train", "--no-overwrite"),
        ("chronological_validation/validate_chronological_holdout.py", "evaluate", "--no-overwrite"),
        ("chronological_validation/analyze_chronological_error.py",),
        ("chronological_validation/evaluate_screening_budget_sensitivity.py",),
    ],
    "deployment": [
        ("prospective_screening/fit_deployment_model.py", "--reference", "full_reference.csv", "--model", "deployment.joblib"),
        ("prospective_screening/compute_candidate_evidence.py", "--reference", "full_reference.csv", "--model", "deployment.joblib",
         "--output-dir", "prospective_outputs"),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("stages", nargs="+", choices=STAGES)
    args = parser.parse_args()
    work = args.work_dir.resolve(strict=True)
    if work.is_relative_to(PACKAGE):
        raise ValueError("Use a separate controlled working directory.")
    if not (work / "full_reference.csv").is_file():
        raise FileNotFoundError("Run prepare_validation_splits.py on the author-supplied data first.")
    env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE="1")
    for stage in args.stages:
        for program, *options in STAGES[stage]:
            print(f"Running {stage}: {program}", flush=True)
            subprocess.run([sys.executable, str(CODE / program), *options], cwd=work, env=env, check=True)


if __name__ == "__main__":
    main()

