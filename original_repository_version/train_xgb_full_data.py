# -*- coding: utf-8 -*-
"""
train_xgb_full_data.py
======================
Train the final deployment XGBoost model on the complete 613-record dataset.

Design principles
-----------------
1) Reconstruct the complete preprocessed dataset from the five source-aware
   held-out folds. Under GroupKFold, each record appears exactly once across
   the five test folds.
2) Reuse the SAME globally optimized hyperparameters selected by the final
   source-aware five-fold Optuna run.
3) Do NOT rerun Optuna on the full dataset. Cross-validation is the model-
   selection stage; this script is the final refit/deployment stage.
4) In-sample metrics are exported only as diagnostics and MUST NOT replace
   the pooled source-aware OOF metrics in the manuscript.

Prerequisites
-------------
- splits/source_kfold/fold1_test.csv ... fold5_test.csv
- models/xgb_source_kfold/xgb_fold1_meta.json ... fold5_meta.json
- build_pooled_oof_evidence.py should already have been run for the frozen
  retrospective evaluation, although it is not required for model fitting.

Outputs
-------
models/xgb_full_data/
    xgb_full_data.joblib
    xgb_full_data_meta.json
    full_dataset.csv
    full_data_training_predictions.csv
    source_fold_membership.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from evidence_core import calc_regression_metrics, identify_source_column


# =============================================================================
# Configuration
# =============================================================================

FOLDS: Sequence[int] = (1, 2, 3, 4, 5)

DATA_DIR = Path("splits/source_kfold")
REFERENCE_MODEL_DIR = Path("models/xgb_source_kfold")
POOLED_OOF_DIR = REFERENCE_MODEL_DIR / "pooled_oof_evidence"

OUTPUT_DIR = Path("models/xgb_full_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "xgb_full_data.joblib"
META_PATH = OUTPUT_DIR / "xgb_full_data_meta.json"
FULL_DATA_PATH = OUTPUT_DIR / "full_dataset.csv"


# =============================================================================
# Utilities
# =============================================================================

def load_fold_meta(fold: int) -> dict:
    path = REFERENCE_MODEL_DIR / f"xgb_fold{fold}_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing source-aware fold metadata: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merged_model_params(meta: dict) -> Dict:
    if "params_fixed" not in meta or "params_best" not in meta:
        raise ValueError(
            "Fold metadata must contain 'params_fixed' and 'params_best'. "
            "Please use the final train_xgb_source_kfold_optuna.py outputs."
        )
    params = dict(meta["params_fixed"])
    params.update(meta["params_best"])
    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", 8)
    return params


def assert_same_model_definition(
    metas: List[dict],
    params_list: List[Dict],
) -> Tuple[List[str], str, Dict]:
    ref_features = list(metas[0]["feature_cols"])
    ref_target = str(metas[0]["target_col"])
    ref_params = dict(params_list[0])

    for fold_idx, (meta, params) in enumerate(zip(metas[1:], params_list[1:]), start=2):
        features = list(meta["feature_cols"])
        target = str(meta["target_col"])
        if features != ref_features:
            raise RuntimeError(f"Feature-order mismatch in fold {fold_idx}")
        if target != ref_target:
            raise RuntimeError(f"Target mismatch in fold {fold_idx}")

        # n_jobs is a runtime choice rather than a fitted model definition.
        a = {k: v for k, v in ref_params.items() if k != "n_jobs"}
        b = {k: v for k, v in params.items() if k != "n_jobs"}
        if a != b:
            raise RuntimeError(
                f"Hyperparameter mismatch in fold {fold_idx}. "
                "The final source-aware pipeline should use one global best parameter set."
            )

    return ref_features, ref_target, ref_params


def reconstruct_full_dataset() -> pd.DataFrame:
    """
    Concatenate the five source-aware held-out folds.

    GroupKFold guarantees each original row is held out exactly once. This is
    preferable to selecting one arbitrary fold's train+test pair because it
    explicitly verifies the pooled OOF membership used in the manuscript.
    """
    frames = []

    for fold in FOLDS:
        path = DATA_DIR / f"fold{fold}_test.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold test data: {path}")
        df = pd.read_csv(path).copy()
        df.insert(0, "source_oof_fold", int(fold))
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)

    if "sample_id" not in full.columns:
        raise ValueError(
            "The final full-data pipeline requires sample_id for row-identity checks."
        )

    n_unique = int(full["sample_id"].nunique(dropna=False))
    if n_unique != len(full):
        dup = full.loc[
            full["sample_id"].duplicated(keep=False), "sample_id"
        ].astype(str).unique()[:10].tolist()
        raise RuntimeError(
            f"Full-data reconstruction contains duplicate sample_id values: {dup}"
        )

    source_col = identify_source_column(full)
    if source_col is not None:
        source_fold_counts = full.groupby(source_col)["source_oof_fold"].nunique()
        if (source_fold_counts > 1).any():
            bad = source_fold_counts[source_fold_counts > 1]
            raise RuntimeError(
                "Source-aware GroupKFold violation during full-data reconstruction: "
                f"{bad.index.astype(str).tolist()[:10]}"
            )

    return full


def load_optional_oof_metrics() -> dict | None:
    path = POOLED_OOF_DIR / "pooled_oof_metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    metas = [load_fold_meta(fold) for fold in FOLDS]
    params_list = [merged_model_params(meta) for meta in metas]
    feature_cols, target_col, model_params = assert_same_model_definition(
        metas, params_list
    )

    full_df = reconstruct_full_dataset()

    required = feature_cols + [target_col]
    missing = [c for c in required if c not in full_df.columns]
    if missing:
        raise ValueError(f"Full dataset missing required model columns: {missing}")

    print("=" * 90)
    print("Final full-data deployment model")
    print("=" * 90)
    print(f"Full records: {len(full_df)}")
    print(f"Unique sample_id: {full_df['sample_id'].nunique()}")

    source_col = identify_source_column(full_df)
    if source_col is not None:
        print(f"Distinct sources: {full_df[source_col].nunique()}")

    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target_col}")
    print("Hyperparameters: reused from the global source-aware 5-fold Optuna optimum")

    X = full_df[feature_cols].to_numpy(dtype=float)
    y = full_df[target_col].to_numpy(dtype=float)

    model = xgb.XGBRegressor(**model_params)
    model.fit(X, y)

    # Diagnostic only: the model was trained on these same records.
    y_fit = np.asarray(model.predict(X), dtype=float)
    fit_metrics = calc_regression_metrics(y, y_fit)

    print(
        "\nIn-sample diagnostic only: "
        f"R2={fit_metrics['R2']:.4f}, "
        f"MAE={fit_metrics['MAE']:.4f}, "
        f"RMSE={fit_metrics['RMSE']:.4f}"
    )
    print("NOTE: manuscript predictive performance must remain the pooled source-aware OOF result.")

    joblib.dump(model, MODEL_PATH)

    # Full evidence pool used by later calibration and prospective screening.
    full_df.to_csv(FULL_DATA_PATH, index=False, encoding="utf-8-sig")

    pred = full_df[
        [c for c in (
            "source_oof_fold", "sample_id", "source_key", "Ref_source_num",
            "Year", "Temperature", "E_Time"
        ) if c in full_df.columns]
    ].copy()
    pred["y_true"] = y
    pred["y_fit_full_model"] = y_fit
    pred["fit_error"] = y_fit - y
    pred["fit_abs_error"] = np.abs(pred["fit_error"])
    pred.to_csv(
        OUTPUT_DIR / "full_data_training_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    membership_cols = [
        c for c in ("sample_id", "source_key", "Ref_source_num", "source_oof_fold")
        if c in full_df.columns
    ]
    full_df[membership_cols].to_csv(
        OUTPUT_DIR / "source_fold_membership.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oof_metrics = load_optional_oof_metrics()

    meta = {
        "purpose": "final full-data deployment model for prospective screening",
        "n_records": int(len(full_df)),
        "n_sources": int(full_df[source_col].nunique()) if source_col else None,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "model_params": model_params,
        "hyperparameter_source": (
            "globally optimized source-aware five-fold Optuna parameter set; "
            "no new full-data optimization performed"
        ),
        "dataset_reconstruction": (
            "concatenation of the five mutually exclusive source-aware held-out folds"
        ),
        "in_sample_diagnostic_metrics": fit_metrics,
        "pooled_source_aware_oof_metrics": oof_metrics,
        "interpretation_note": (
            "In-sample metrics are diagnostics only. Generalization performance is "
            "reported from pooled source-aware five-fold OOF evaluation."
        ),
        "model_path": str(MODEL_PATH),
        "full_dataset_path": str(FULL_DATA_PATH),
    }
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved full evidence pool: {FULL_DATA_PATH}")
    print(f"Saved metadata: {META_PATH}")


if __name__ == "__main__":
    main()
