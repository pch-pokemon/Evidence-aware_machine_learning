# -*- coding: utf-8 -*-
"""Sensitivity of MED mixing and the PED protocol contribution.

Uses only FROZEN Nested source-aware OOF evidence, saved training-reference
LOO components, physical priors, and fold-specific protocol feature gain.
NO XGBoost fitting, model loading, Optuna, or changes to official results.

One-factor-at-a-time sensitivity:
  MED_alpha = alpha*D_LEAF + (1-alpha)*D_SUPPORT
      alpha = 0.25,0.40,0.50,0.60,0.75 (baseline 0.50)
  PED_lambda = mean_top10 sqrt((1-lambda)*d2_main + lambda*d2_protocol)
      lambda = 0.05,0.10,0.15,0.20,0.25,0.30 (baseline 0.15)

For EVERY scenario: compute its own per-fold TRAINING-reference P50 thresholds.
The other axis retains the frozen baseline metric and its per-fold threshold.
Sample-level labels are compared with the frozen official Nested P50 labels.

Run from the project root, alongside evidence_core.py:
  python evaluate_evidence_weights.py
  # or: !python evaluate_evidence_weights.py

Optional:
  --bootstrap 0    # skip source-cluster bootstrap CI for faster diagnosis
  --no-excel
  --project-root 'C:/path/to/project'
  --expected-records 613  # default, set 0 for artificial tests

Outputs: models/nested_evidence_weight_sensitivity/
  MED_alpha_sensitivity.csv, PED_lambda_sensitivity.csv,
  sensitivity_all_scenarios.csv, fold_scenario_thresholds.csv,
  scenario_oof_states.csv, sensitivity_summary.xlsx, run_config.json

CI: a conditional-on-frozen-predictions DOI/source-cluster bootstrap. It does
not include retraining or parameter selection uncertainty. This experiment
assesses evidence partition stability, NOT the predictive R2 of XGBoost.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from evidence_core import (
    EvidenceMetricConfig, MAIN_PHYSICAL_FEATURES, PROTOCOL_NUM_FEATURES,
    build_physical_arrays, physical_distance_components_for_query,
    load_main_weights, topk_indices, assign_quadrants,
)

FOLDS = range(1, 6)
ALPHAS = (0.25, 0.40, 0.50, 0.60, 0.75)
LAMBDAS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
ALPHA0 = 0.50
LAMBDA0 = 0.15
STATES = ("Q1", "Q2", "Q3", "Q4")
BASE_CFG = EvidenceMetricConfig(topk=10, lambda_protocol=LAMBDA0,
                                ht_route_mismatch_penalty=1.0,
                                ped_threshold_scale="log1p",
                                clip_training_reference_normalization=True)


def parse_args():
    p = argparse.ArgumentParser(description="Nested evidence-weight sensitivity")
    p.add_argument("--project-root", default=None)
    p.add_argument("--data-dir", default="splits/source_kfold")
    p.add_argument("--evidence-dir", default="models/nested_evidence_final")
    p.add_argument("--main-weights-json", default="models/fixed_evidence_weights/final_fixed_evidence_weights.json")
    p.add_argument("--output-dir", default="models/nested_evidence_weight_sensitivity")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Source-cluster bootstrap draws per scenario, 0 to disable")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected-records", type=int, default=613)
    p.add_argument("--no-excel", action="store_true")
    a = p.parse_args()
    if a.bootstrap < 0 or a.expected_records < 0:
        p.error("bootstrap and expected-records must be nonnegative")
    return a


def path(root: Path, s: str) -> Path:
    p = Path(s).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def save_csv(frame: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(p, index=False, encoding="utf-8-sig")


def validate_oof(df: pd.DataFrame, expected: int):
    needed = {"fold", "sample_id", "source_key", "abs_error", "MED", "PED_log1p",
              "D_LEAF", "D_SUPPORT", "quadrant", "y_true", "y_pred"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError("Official Nested OOF is missing " + str(sorted(missing)))
    if expected and len(df) != expected:
        raise ValueError(f"Expected {expected} frozen OOF records; found {len(df)}")
    if df.sample_id.isna().any() or df.sample_id.astype(str).duplicated().any():
        raise ValueError("Missing/duplicate sample_id in frozen OOF")
    if set(df.fold.unique()) != set(FOLDS):
        raise ValueError("Frozen OOF must cover folds 1..5")
    if df.groupby("source_key").fold.nunique().max() != 1:
        raise ValueError("Source_key appears in multiple outer held-out folds")
    if not np.allclose(df.MED, ALPHA0 * df.D_LEAF + (1-ALPHA0)*df.D_SUPPORT, atol=1e-11):
        raise ValueError("Frozen MED is not 0.5*D_LEAF + 0.5*D_SUPPORT")
    if not np.allclose(np.abs(df.y_pred-df.y_true), df.abs_error, atol=1e-8):
        raise ValueError("Frozen OOF absolute prediction error inconsistent")
    if not set(df.quadrant.unique()).issubset(set(STATES)):
        raise ValueError("Unexpected frozen Q state")


def load_fold_reference(ev_dir: Path, fold: int, train: pd.DataFrame) -> pd.DataFrame:
    p = ev_dir / f"fold{fold}_training_reference_evidence.csv"
    if not p.is_file():
        raise FileNotFoundError(p)
    d = pd.read_csv(p, low_memory=False)
    for col in ("sample_id", "D_LEAF", "D_SUPPORT", "PED_log1p", "MED"):
        if col not in d:
            raise ValueError(f"Missing {col} in {p}")
    if len(train) != len(d) or train.sample_id.astype(str).tolist() != d.sample_id.astype(str).tolist():
        raise ValueError(f"Fold {fold} training reference rows not exactly aligned to source fold train CSV")
    if not np.allclose(d.MED, (d.D_LEAF+d.D_SUPPORT)*0.5, atol=1e-10):
        raise ValueError(f"Fold {fold} training MED did not reproduce 0.5/0.5")
    return d


def protocol_weights(df: pd.DataFrame, fold: int) -> dict:
    part = df[df.fold == fold]
    if len(part) != len(PROTOCOL_NUM_FEATURES) or part.feature.duplicated().any():
        raise ValueError(f"Fold {fold} protocol gain count/duplicates mismatch")
    w = dict(zip(part.feature.astype(str), part.weight.astype(float)))
    if set(w) != set(PROTOCOL_NUM_FEATURES) or abs(sum(w.values())-1) > 1e-6:
        raise ValueError(f"Fold {fold} protocol weights missing/unnormalized")
    return w


def distance_component_matrices(train: pd.DataFrame, test: pd.DataFrame, main_w: dict, prot_w: dict):
    """Compute pairwise main/protocol squared distances only once per outer fold.

    Uses the SAME evidence_core definitions, route gating and training scaling
    as the official Nested OOF, but postpones mixing by lambda.
    """
    train_arrays = build_physical_arrays(train, train, main_w, prot_w)
    test_arrays = build_physical_arrays(train, test, main_w, prot_w)
    tr_main = np.empty((len(train), len(train)), dtype=float)
    tr_proto = np.empty_like(tr_main)
    te_main = np.empty((len(test), len(train)), dtype=float)
    te_proto = np.empty_like(te_main)
    for i in range(len(train)):
        _, ma, pr = physical_distance_components_for_query(train_arrays, i, BASE_CFG)
        tr_main[i] = ma
        tr_proto[i] = pr
    for i in range(len(test)):
        _, ma, pr = physical_distance_components_for_query(test_arrays, i, BASE_CFG)
        te_main[i] = ma
        te_proto[i] = pr
    return (tr_main, tr_proto), (te_main, te_proto)


def ped_for_lambda(components: tuple[np.ndarray, np.ndarray], lam: float, *, loo: bool):
    main, proto = components
    d = np.sqrt(np.maximum((1-lam)*main + lam*proto, 0.0))
    n_query, n_ref = d.shape
    if loo and n_query != n_ref:
        raise ValueError("Training LOO must use a square pairwise matrix")
    if loo:
        np.fill_diagonal(d, np.inf)
    k = int(BASE_CFG.topk)
    if n_ref <= k:
        raise ValueError("Number of training references must exceed topk")
    # Reuse exact evidence_core top-k selection semantics, preserving ties.
    ped_raw = np.empty(n_query, dtype=float)
    for i in range(n_query):
        ix = topk_indices(d[i], k)
        ped_raw[i] = float(np.mean(d[i, ix]))
    return np.log1p(ped_raw)


def cluster_bootstrap_ci(df: pd.DataFrame, q_col: str, draws: np.ndarray) -> tuple[float, float, int]:
    """Conditional source-cluster bootstrap of Q4-Q1 MAE gap.

    draws are source multiplicity matrices (n_boot, n_sources), ensuring
    the *same resampling* for every tested coefficient value.
    """
    source_codes = df["_source_code"].to_numpy(dtype=int)
    n_source = draws.shape[1]
    mask1 = (df[q_col].to_numpy() == "Q1")
    mask4 = (df[q_col].to_numpy() == "Q4")
    errors = df.abs_error.to_numpy(dtype=float)
    num1 = np.bincount(source_codes, weights=errors*mask1, minlength=n_source)
    den1 = np.bincount(source_codes, weights=mask1.astype(float), minlength=n_source)
    num4 = np.bincount(source_codes, weights=errors*mask4, minlength=n_source)
    den4 = np.bincount(source_codes, weights=mask4.astype(float), minlength=n_source)
    count1 = draws @ den1
    count4 = draws @ den4
    good = (count1 > 0) & (count4 > 0)
    delta = (draws[good] @ num4)/(count4[good]) - (draws[good] @ num1)/(count1[good])
    if delta.size < 20:
        return float('nan'), float('nan'), int(delta.size)
    lo, hi = np.percentile(delta, [2.5,97.5])
    return float(lo), float(hi), int(delta.size)


def summary_row(df: pd.DataFrame, scenario: str, param_name: str, param: float,
                base_labels: np.ndarray, base_value: np.ndarray,
                score_col: str, bootstrap_draws: np.ndarray | None):
    vals = df[score_col].to_numpy(dtype=float)
    q = df.quadrant_scenario.to_numpy(dtype=str)
    errors = df.abs_error.to_numpy(dtype=float)
    counts = {s: int(np.sum(q == s)) for s in STATES}
    q1 = q == "Q1"
    q4 = q == "Q4"
    mae1 = float(errors[q1].mean()) if q1.any() else float('nan')
    mae4 = float(errors[q4].mean()) if q4.any() else float('nan')
    if bootstrap_draws is not None:
        ci_lo, ci_hi, n_valid = cluster_bootstrap_ci(df, "quadrant_scenario", bootstrap_draws)
    else:
        ci_lo, ci_hi, n_valid = float('nan'), float('nan'), 0
    # Across folds, raw metric scales need not coincide; rank comparison is descriptive only.
    rho = pd.Series(vals).corr(pd.Series(base_value), method='spearman')
    change = int(np.sum(q != base_labels))
    return {
        "analysis": scenario,
        "parameter": param_name,
        "value": float(param),
        "baseline_value": ALPHA0 if param_name == "alpha" else LAMBDA0,
        "n": len(df),
        **{f"{s}_count": counts[s] for s in STATES},
        **{f"{s}_fraction": counts[s]/len(df) for s in STATES},
        "Q1_MAE": mae1,
        "Q4_MAE": mae4,
        "Q4_minus_Q1_MAE": mae4-mae1,
        "Q4_minus_Q1_CI95_low": ci_lo,
        "Q4_minus_Q1_CI95_high": ci_hi,
        "bootstrap_valid": n_valid,
        "n_changed_vs_base": change,
        "fraction_changed_vs_base": change/len(df),
        "Q_agreement_vs_base": 1-change/len(df),
        "spearman_changed_axis_vs_base": float(rho),
        "frozen_predictions": True,
        "fold_specific_train_P50_recalibrated": True,
    }


def main():
    args = parse_args()
    BASE_CFG.validate()
    root = Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd()
    ev_dir = path(root, args.evidence_dir)
    data_dir = path(root, args.data_dir)
    prior_file = path(root, args.main_weights_json)
    out_dir = path(root, args.output_dir)
    if out_dir in (ev_dir, data_dir, prior_file.parent) or out_dir in ev_dir.parents or ev_dir in out_dir.parents:
        raise ValueError("Output directory overlaps protected input path")
    for p in (ev_dir / "pooled_source_oof_evidence.csv",
              ev_dir / "weights/protocol_fold_weights.csv", prior_file):
        if not p.is_file():
            raise FileNotFoundError(f"Required frozen input not found: {p}")
    frozen = pd.read_csv(ev_dir / "pooled_source_oof_evidence.csv", low_memory=False)
    validate_oof(frozen, args.expected_records)
    frozen["sample_id"] = frozen.sample_id.astype(str)
    frozen = frozen.sort_values(["fold", "fold_test_idx"], kind="stable").reset_index(drop=True)
    base_labels = frozen.quadrant.astype(str).to_numpy()
    main_w = load_main_weights(prior_file)
    if set(main_w) != set(MAIN_PHYSICAL_FEATURES):
        raise ValueError("Main physical weight features do not match core")
    protocol_csv = pd.read_csv(ev_dir / "weights/protocol_fold_weights.csv")

    all_alpha, all_lambda = [], []
    print("="*96)
    print("NESTED MED AND PED WEIGHT SENSITIVITY")
    print("Project:", root)
    print("Frozen OOF:", ev_dir)
    print("Output:", out_dir)
    print("alpha grid:", ALPHAS, "baseline", ALPHA0)
    print("lambda grid:", LAMBDAS, "baseline", LAMBDA0)
    print("Source bootstrap:", args.bootstrap, "draws (conditional on frozen models)")
    print("For every variant, training reference P50 is recomputed per fold.")
    print("="*96, flush=True)

    for fold in FOLDS:
        trp = data_dir / f"fold{fold}_train.csv"
        tep = data_dir / f"fold{fold}_test.csv"
        if not trp.is_file() or not tep.is_file():
            raise FileNotFoundError(f"Missing fold {fold} train/test source CSV")
        tr = pd.read_csv(trp, low_memory=False)
        te = pd.read_csv(tep, low_memory=False)
        subset = frozen[frozen.fold == fold].reset_index(drop=True)
        if (len(subset) != len(te) or
            subset.sample_id.tolist() != te.sample_id.astype(str).tolist()):
            raise ValueError(f"Fold {fold} official OOF/test sample_id mismatch")
        if set(tr.source_key.astype(str)) & set(te.source_key.astype(str)):
            raise ValueError(f"Fold {fold} contains train/test source overlap")
        ref = load_fold_reference(ev_dir, fold, tr)
        w = protocol_weights(protocol_csv, fold)
        print(f"Fold {fold}: train={len(tr)}, test={len(te)}; alpha variants...", flush=True)
        for alpha in ALPHAS:
            tr_med = alpha*ref.D_LEAF.to_numpy(dtype=float) + (1-alpha)*ref.D_SUPPORT.to_numpy(dtype=float)
            test_med = alpha*subset.D_LEAF.to_numpy(dtype=float)+(1-alpha)*subset.D_SUPPORT.to_numpy(dtype=float)
            med_threshold = float(np.percentile(tr_med, 50))
            ped_threshold = float(subset.PED_threshold_P50_fold.iloc[0])
            q = assign_quadrants(test_med, subset.PED_log1p.to_numpy(dtype=float),
                                 med_threshold, ped_threshold)
            d = subset[["fold", "sample_id", "source_key", "abs_error", "quadrant"]].copy()
            d = d.rename(columns={"quadrant": "quadrant_baseline"})
            d["quadrant_scenario"] = q
            d["scenario"] = "MED_alpha"
            d["parameter"] = "alpha"
            d["value"] = alpha
            d["MED_variant"] = test_med
            d["PED_variant"] = subset.PED_log1p.to_numpy(dtype=float)
            d["MED_threshold"] = med_threshold
            d["PED_threshold"] = ped_threshold
            all_alpha.append(d)
            if alpha == ALPHA0:
                if not np.allclose(test_med, subset.MED, atol=1e-11):
                    raise RuntimeError(f"Fold {fold}: alpha baseline MED failed reproduction")
                if not np.array_equal(q, subset.quadrant.to_numpy(dtype=str)):
                    raise RuntimeError(f"Fold {fold}: alpha baseline Q failed reproduction")
                if not np.isclose(med_threshold, float(subset.MED_threshold_P50_fold.iloc[0]), atol=1e-10):
                    raise RuntimeError(f"Fold {fold}: alpha baseline MED P50 failed reproduction")

        print(f"Fold {fold}: computing fixed d2_main/d2_protocol (once) for lambda grid...", flush=True)
        training_pair, external_pair = distance_component_matrices(tr, te, main_w, w)
        for lam in LAMBDAS:
            training_ped = ped_for_lambda(training_pair, lam, loo=True)
            external_ped = ped_for_lambda(external_pair, lam, loo=False)
            ped_threshold = float(np.percentile(training_ped, 50))
            med_threshold = float(subset.MED_threshold_P50_fold.iloc[0])
            q = assign_quadrants(subset.MED.to_numpy(dtype=float), external_ped,
                                 med_threshold, ped_threshold)
            d = subset[["fold", "sample_id", "source_key", "abs_error", "quadrant"]].copy()
            d = d.rename(columns={"quadrant": "quadrant_baseline"})
            d["quadrant_scenario"] = q
            d["scenario"] = "PED_lambda"
            d["parameter"] = "lambda"
            d["value"] = lam
            d["MED_variant"] = subset.MED.to_numpy(dtype=float)
            d["PED_variant"] = external_ped
            d["MED_threshold"] = med_threshold
            d["PED_threshold"] = ped_threshold
            all_lambda.append(d)
            if lam == LAMBDA0:
                maxdiff = float(np.max(np.abs(external_ped - subset.PED_log1p.to_numpy(dtype=float))))
                if maxdiff > 2e-9:
                    raise RuntimeError(f"Fold {fold}: lambda baseline PED differs from frozen OOF, max {maxdiff}")
                if not np.isclose(ped_threshold, float(subset.PED_threshold_P50_fold.iloc[0]), atol=2e-9):
                    raise RuntimeError(f"Fold {fold}: lambda baseline PED P50 failed reproduction")
                if not np.array_equal(q, subset.quadrant.to_numpy(dtype=str)):
                    raise RuntimeError(f"Fold {fold}: lambda baseline Q failed reproduction")
        print(f"Fold {fold}: all baseline checks passed", flush=True)

    details = pd.concat(all_alpha + all_lambda, ignore_index=True)
    source_ids, source_codes = np.unique(frozen.source_key.astype(str).to_numpy(), return_inverse=True)
    src_map = dict(zip(source_ids, np.arange(len(source_ids))))
    details["_source_code"] = details.source_key.astype(str).map(src_map).astype(int)
    rng = np.random.default_rng(args.seed)
    if args.bootstrap:
        counts = rng.multinomial(len(source_ids), np.full(len(source_ids), 1/len(source_ids)), size=args.bootstrap)
    else:
        counts = None
    base_score = {"MED_alpha": frozen.MED.to_numpy(dtype=float), "PED_lambda": frozen.PED_log1p.to_numpy(dtype=float)}
    rows=[]
    for (analysis, par, val), sub in details.groupby(["scenario", "parameter", "value"], sort=True):
        # Regroup in identical sample_id order before baseline label/rank comparison.
        sub = sub.set_index("sample_id").loc[frozen.sample_id].reset_index()
        sub["_source_code"] = sub.source_key.astype(str).map(src_map).astype(int)
        rows.append(summary_row(sub, analysis, par, val, base_labels, base_score[analysis],
                                "MED_variant" if analysis=="MED_alpha" else "PED_variant", counts))
    summary = pd.DataFrame(rows).sort_values(["analysis", "value"]).reset_index(drop=True)
    alpha_table = summary.loc[summary.analysis == "MED_alpha"].copy()
    lambda_table = summary.loc[summary.analysis == "PED_lambda"].copy()
    # Mandatory end-to-end baseline consistency.
    for a, v in (("MED_alpha", ALPHA0), ("PED_lambda", LAMBDA0)):
        row = summary.loc[(summary.analysis==a) & np.isclose(summary.value, v)].iloc[0]
        if (int(row.n_changed_vs_base) != 0 or
            int(row.Q1_count) != int((frozen.quadrant=="Q1").sum()) or
            int(row.Q4_count) != int((frozen.quadrant=="Q4").sum())):
            raise RuntimeError(f"Baseline scenario {a} did not reproduce frozen Q states")

    # compact threshold audit of each fold and scenario
    fold_thresh = details.groupby(["scenario", "parameter", "value", "fold"], as_index=False).agg(
        MED_threshold=("MED_threshold", "first"),
        PED_threshold=("PED_threshold", "first"),
        n_test=("sample_id", "size")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(alpha_table, out_dir / "MED_alpha_sensitivity.csv")
    save_csv(lambda_table, out_dir / "PED_lambda_sensitivity.csv")
    save_csv(summary, out_dir / "sensitivity_all_scenarios.csv")
    save_csv(fold_thresh, out_dir / "fold_scenario_thresholds.csv")
    save_csv(details.drop(columns=["_source_code"]), out_dir / "scenario_oof_states.csv")
    config={"alpha_grid": ALPHAS, "lambda_grid": LAMBDAS, "alpha_baseline": ALPHA0,
            "lambda_baseline": LAMBDA0, "topk": BASE_CFG.topk,
            "ht_route_mismatch_penalty": BASE_CFG.ht_route_mismatch_penalty,
            "reference_threshold": "each outer training fold's own P50, recomputed per variation",
            "other_axis": "frozen baseline metric and threshold",
            "number_of_oof_records": len(frozen), "n_literature_sources": len(source_ids),
            "bootstrap_draws": args.bootstrap, "seed": args.seed,
            "bootstrap_scope": "conditional on frozen models and predictions",
            "note": "Sensitivity is one-factor-at-a-time. Does not retune models or optimize Q errors."}
    (out_dir/"run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_excel:
        with pd.ExcelWriter(out_dir/"sensitivity_summary.xlsx", engine="openpyxl") as writer:
            alpha_table.to_excel(writer, sheet_name="MED_alpha", index=False)
            lambda_table.to_excel(writer, sheet_name="PED_lambda", index=False)
            fold_thresh.to_excel(writer, sheet_name="Fold_Thresholds", index=False)
            pd.DataFrame([config]).to_excel(writer, sheet_name="Protocol", index=False)
            for sheet in writer.sheets.values():
                sheet.freeze_panes="A2"
                sheet.auto_filter.ref=sheet.dimensions
    cols = ["value", "Q1_count", "Q2_count", "Q3_count", "Q4_count", "Q_agreement_vs_base",
            "Q1_MAE", "Q4_MAE", "Q4_minus_Q1_MAE", "Q4_minus_Q1_CI95_low", "Q4_minus_Q1_CI95_high"]
    print("\n"+"="*96)
    print("MED MIXING WEIGHT SENSITIVITY")
    print(alpha_table[cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nPED PROTOCOL WEIGHT SENSITIVITY")
    print(lambda_table[cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nBoth frozen baseline scenario reproductions: STRICT PASS")
    print("No XGBoost training. Files saved to:", out_dir, flush=True)


if __name__ == "__main__":
    main()
