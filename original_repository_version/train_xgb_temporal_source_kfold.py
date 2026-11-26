# -*- coding: utf-8 -*-
"""
train_xgb_temporal_source_kfold.py
==================================
Train historical (time-cut) source-aware XGBoost models for temporal knowledge-state analysis.

Design
------
- Reuse the EXISTING five source-aware folds.
- Keep each fold's held-out test set unchanged.
- Truncate only the corresponding training fold to Year <= YEAR_CUTOFF.
- Reuse the SAME globally optimized hyperparameters as the current full-data source-aware models.
- Do NOT rerun Optuna on the historical subset. This isolates the effect of evidence accumulation
  rather than conflating it with a second hyperparameter search.

Prerequisites
-------------
1) split_dataset.py
2) train_xgb_source_kfold_optuna.py

Outputs
-------
models/xgb_source_kfold_temporal_2022/
    xgb_fold1.joblib ... xgb_fold5.joblib
    xgb_fold1_meta.json ... xgb_fold5_meta.json
    fold1_train_2022.csv ... fold5_train_2022.csv
    temporal_fold_metrics.csv
    run_config.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from evidence_core import calc_regression_metrics, identify_source_column

# =============================================================================
# Configuration
# =============================================================================

YEAR_CUTOFF = 2022
FOLDS: Sequence[int] = (1, 2, 3, 4, 5)

DATA_DIR = Path("splits/source_kfold")
CURRENT_MODEL_DIR = Path("models/xgb_source_kfold")
OUTPUT_DIR = Path(f"models/xgb_source_kfold_temporal_{YEAR_CUTOFF}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utilities
# =============================================================================

def load_current_fold_artifacts(fold: int):
    train_path = DATA_DIR / f"fold{fold}_train.csv"
    test_path = DATA_DIR / f"fold{fold}_test.csv"
    meta_path = CURRENT_MODEL_DIR / f"xgb_fold{fold}_meta.json"

    required = [train_path, test_path, meta_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return train_df, test_df, meta


def merged_model_params(meta: dict) -> Dict:
    if "params_fixed" not in meta or "params_best" not in meta:
        raise ValueError(
            "Current model metadata must contain params_fixed and params_best. "
            "Re-run train_xgb_source_kfold_optuna.py if needed."
        )
    params = dict(meta["params_fixed"])
    params.update(meta["params_best"])
    # Make runtime reproducibility explicit even if older metadata omitted it.
    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", 8)
    return params


def assert_shared_hyperparameters(all_params: List[Dict]) -> None:
    reference = all_params[0]
    for i, params in enumerate(all_params[1:], start=2):
        # n_jobs is runtime-only; ignore it in the equality check.
        a = {k: v for k, v in reference.items() if k != "n_jobs"}
        b = {k: v for k, v in params.items() if k != "n_jobs"}
        if a != b:
            raise RuntimeError(
                f"Fold {i} metadata does not share the same optimized hyperparameters. "
                "The final source-aware training pipeline is expected to use one global parameter set."
            )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    fold_payloads = []
    all_params = []

    # First pass: load and verify current artifacts.
    for fold in FOLDS:
        full_train, test_df, meta = load_current_fold_artifacts(fold)
        params = merged_model_params(meta)
        all_params.append(params)
        fold_payloads.append((fold, full_train, test_df, meta, params))

    assert_shared_hyperparameters(all_params)

    results = []

    for fold, full_train, test_df, meta, params in fold_payloads:
        print(f"\n{'=' * 90}\nHistorical temporal model | Fold {fold}\n{'=' * 90}")

        feature_cols = list(meta["feature_cols"])
        target_col = str(meta["target_col"])

        if "Year" not in full_train.columns:
            raise ValueError(f"Fold {fold} training data does not contain Year")

        hist_train = full_train.loc[
            pd.to_numeric(full_train["Year"], errors="coerce") <= YEAR_CUTOFF
        ].copy()

        if len(hist_train) == 0:
            raise ValueError(f"Fold {fold}: historical training pool is empty")
        if len(hist_train) <= 10:
            raise ValueError(f"Fold {fold}: historical training pool is too small ({len(hist_train)})")

        for name, df in (("historical train", hist_train), ("fixed test", test_df)):
            missing = [c for c in feature_cols + [target_col] if c not in df.columns]
            if missing:
                raise ValueError(f"Fold {fold} {name} missing columns: {missing}")

        # Source-aware split remains valid because historical pool is a subset of full fold training.
        source_col = identify_source_column(full_train)
        if source_col and source_col in test_df.columns:
            overlap = set(hist_train[source_col].dropna()) & set(test_df[source_col].dropna())
            if overlap:
                raise RuntimeError(
                    f"Fold {fold}: historical source leakage detected ({len(overlap)} sources)"
                )
            print("Source overlap check: PASS (0 overlapping sources)")

        X_train = hist_train[feature_cols].to_numpy(dtype=float)
        y_train = hist_train[target_col].to_numpy(dtype=float)
        X_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=float)
        metrics = calc_regression_metrics(y_test, y_pred)

        print(
            f"Year <= {YEAR_CUTOFF} | train={len(hist_train)} | "
            f"R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}"
        )

        model_path = OUTPUT_DIR / f"xgb_fold{fold}.joblib"
        joblib.dump(model, model_path)

        hist_train_path = OUTPUT_DIR / f"fold{fold}_train_{YEAR_CUTOFF}.csv"
        hist_train.to_csv(hist_train_path, index=False, encoding="utf-8-sig")

        fold_meta = {
            "fold": int(fold),
            "year_cutoff": int(YEAR_CUTOFF),
            "derived_from_current_source_fold": int(fold),
            "feature_cols": feature_cols,
            "target_col": target_col,
            "model_params": params,
            "n_train_full": int(len(full_train)),
            "n_train_historical": int(len(hist_train)),
            "n_test_fixed": int(len(test_df)),
            "n_train_sources_full": int(full_train[source_col].nunique()) if source_col else None,
            "n_train_sources_historical": int(hist_train[source_col].nunique()) if source_col else None,
            "n_test_sources": int(test_df[source_col].nunique()) if source_col else None,
            "train_year_min": int(pd.to_numeric(hist_train["Year"], errors="coerce").min()),
            "train_year_max": int(pd.to_numeric(hist_train["Year"], errors="coerce").max()),
            "metrics": metrics,
            "model_path": str(model_path),
            "historical_train_path": str(hist_train_path),
            "fixed_test_path": str(DATA_DIR / f"fold{fold}_test.csv"),
        }
        with (OUTPUT_DIR / f"xgb_fold{fold}_meta.json").open("w", encoding="utf-8") as f:
            json.dump(fold_meta, f, indent=2, ensure_ascii=False)

        results.append({
            "fold": int(fold),
            "year_cutoff": int(YEAR_CUTOFF),
            "n_train_full": int(len(full_train)),
            "n_train_historical": int(len(hist_train)),
            "n_test": int(len(test_df)),
            **metrics,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        OUTPUT_DIR / "temporal_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_config = {
        "year_cutoff": int(YEAR_CUTOFF),
        "folds": list(FOLDS),
        "data_dir": str(DATA_DIR),
        "current_model_dir": str(CURRENT_MODEL_DIR),
        "output_dir": str(OUTPUT_DIR),
        "hyperparameter_rule": (
            "Reuse the globally optimized current source-aware XGBoost hyperparameters; "
            "do not rerun Optuna on the historical subset."
        ),
        "test_set_rule": (
            "Use the same held-out source-aware test set in each fold; only the training "
            f"pool is truncated to Year <= {YEAR_CUTOFF}."
        ),
    }
    with (OUTPUT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("\nTemporal historical model training finished.")
    print(result_df.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
