# -*- coding: utf-8 -*-
"""evaluate_med_calibration.py

MED calibration sensitivity for FROZEN Nested source-aware OOF.

GOAL: contrast three different calibration REFERENCES, not retrain XGBoost:
  A. Official fitted-tree training-reference LOO median, per outer fold.
  B. Source-held-out crossfit reference, transported by percentile rank.
  C. Random-held-out crossfit reference, transported by percentile rank (control).

Critically, auxiliary models have DIFFERENT trees/MED scales. This script NEVER
pools their raw MED scores or transfers an absolute auxiliary MED threshold to
an outer model. Instead, for each outer training fold:
  - Each auxiliary external score was already converted into a percentile rank
    against THAT auxiliary model's fitted-tree LOO training reference.
  - Take the median such rank across all source-disjoint held-out records from
    that outer fold, defining a comparative external-reference percentile p_i.
  - On the SAVED FULL outer model, set a transported threshold at percentile
    p_i of THAT MODEL's own LOO reference MED distribution.
  - Leave the current official PED and its P50 untouched, recompute Q1-Q4,
    and compare error separation / state changes, with source-cluster bootstrap.

This is a CALIBRATION DEFINITION STRESS TEST, not a statistical proof that the
transported threshold is optimal, unbiased, or uniquely correct.
It deliberately answers a DIFFERENT question than the official training P50:
"relative to the held-out reference percentile" vs "typical training case".
The auxiliary theta_i was selected using the entire outer training set, so
crossfit held-out samples participated INDIRECTLY in hyperparameter choice.

Inputs (read only):
    models/nested_evidence_final/pooled_source_oof_evidence.csv
    models/nested_evidence_final/fold{i}_training_reference_evidence.csv
    models/nested_evidence_final/fold_training_reference_thresholds.csv
    models/nested_med_crossfit_diagnostic/crossfit_heldout_MED.csv
    models/nested_med_crossfit_diagnostic/random_control_crossfit_heldout_MED.csv
    splits/source_kfold/fold{i}_train.csv

Outputs (in a new directory):
    fold_calibration_thresholds.csv
    calibration_scenario_summary.csv
    bootstrap_source_cluster_CI.csv
    oof_calibration_comparison.csv
    source_crossfit_rank_summary.csv
    calibration_sensitivity.xlsx
    run_config.json

Run: !python evaluate_med_calibration.py
Optional: --bootstrap 3000 --seed 42 --no-excel
          --project-root "C:/path/to/project"
          --expected-records 0     # for smoke tests only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = (1, 2, 3, 4, 5)
Q_STATES = ("Q1", "Q2", "Q3", "Q4")
SCENARIOS = ("training_LOO_P50", "source_crossfit_rank", "random_crossfit_rank")


def get_args():
    p = argparse.ArgumentParser(description="Nested MED crossfit calibration sensitivity; NO TRAINING")
    p.add_argument("--project-root", default=None)
    p.add_argument("--formal-dir", default="models/nested_evidence_final")
    p.add_argument("--crossfit-dir", default="models/nested_med_crossfit_diagnostic")
    p.add_argument("--data-dir", default="splits/source_kfold")
    p.add_argument("--output-dir", default="models/nested_med_calibration_sensitivity")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected-records", type=int, default=613)
    p.add_argument("--no-excel", action="store_true")
    args = p.parse_args()
    if args.bootstrap < 100 or args.expected_records < 0:
        p.error("bootstrap >= 100, expected-records >= 0")
    return args


def located(root: Path, s: str) -> Path:
    p = Path(s).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def require_columns(df: pd.DataFrame, cols, label: str):
    absent = sorted(set(cols) - set(df.columns))
    if absent:
        raise RuntimeError(f"{label}: missing columns: {absent}")


def numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    a = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(a).all():
        raise RuntimeError(f"Missing/nonfinite numeric values: {col}")
    return a


def csv_write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def calculate_q(med, ped, med_threshold, ped_threshold) -> np.ndarray:
    # Same <= boundary semantics as evidence_core.assign_quadrants.
    m = np.asarray(med, float) <= np.asarray(med_threshold, float)
    p = np.asarray(ped, float) <= np.asarray(ped_threshold, float)
    return np.select([m & p, m & ~p, ~m & p], ["Q1", "Q2", "Q3"], default="Q4")


def validate_official(formal_dir: Path, expected: int):
    official = pd.read_csv(formal_dir / "pooled_source_oof_evidence.csv", low_memory=False)
    need = ("sample_id", "source_key", "fold", "y_true", "y_pred", "abs_error", "MED",
            "PED_log1p", "MED_threshold_P50_fold", "PED_threshold_P50_fold", "quadrant")
    require_columns(official, need, "Frozen official OOF")
    if expected and len(official) != expected:
        raise RuntimeError(f"Expected {expected} OOF samples, found {len(official)}")
    if official.sample_id.isna().any() or official.sample_id.astype(str).duplicated().any():
        raise RuntimeError("Official sample_id absent/duplicated")
    if official.source_key.isna().any() or set(official.fold.unique()) != set(FOLDS):
        raise RuntimeError("Official source/fold coverage incorrect")
    if official.groupby("source_key").fold.nunique().max() != 1:
        raise RuntimeError("Official held-out source leaks across folds")
    for c in ("MED", "PED_log1p", "MED_threshold_P50_fold", "PED_threshold_P50_fold",
              "y_true", "y_pred", "abs_error"):
        numeric(official, c)
    if np.max(np.abs(np.abs(official.y_true - official.y_pred) - official.abs_error)) > 2e-6:
        raise RuntimeError("Frozen absolute errors do not match predictions")
    base = calculate_q(official.MED, official.PED_log1p,
                       official.MED_threshold_P50_fold, official.PED_threshold_P50_fold)
    if not np.array_equal(base, official.quadrant.astype(str).to_numpy()):
        raise RuntimeError("Formal Q states fail self-reproduction")
    return official


def load_crossfit(path: Path, scheme: str, data_dir: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {scheme} crossfit input: {path}")
    cf = pd.read_csv(path, low_memory=False)
    require_columns(cf, ("outer_fold", "inner_fold", "scheme", "sample_id", "source_key",
                         "MED_percentile_vs_train_LOO", "MED", "training_ref_MED_P50"), str(path))
    if not cf.scheme.astype(str).eq(scheme).all():
        raise RuntimeError(f"Crossfit CSV is not purely {scheme}: {path}")
    if set(cf.outer_fold.unique()) != set(FOLDS):
        raise RuntimeError(f"{scheme}: expected outer folds 1..5")
    if cf.duplicated(["outer_fold", "sample_id"]).any():
        raise RuntimeError(f"{scheme}: repeated outer-train heldout sample")
    ranks = numeric(cf, "MED_percentile_vs_train_LOO")
    if ((ranks < 0) | (ranks > 100)).any():
        raise RuntimeError(f"{scheme}: invalid percentile ranks")
    for outer in FOLDS:
        train = pd.read_csv(data_dir / f"fold{outer}_train.csv", low_memory=False)
        require_columns(train, ("sample_id", "source_key"), f"outer train {outer}")
        sub = cf.loc[cf.outer_fold.eq(outer)]
        if len(sub) != len(train):
            raise RuntimeError(f"{scheme} outer {outer}: crossfit size {len(sub)} != train {len(train)}")
        merged = train[["sample_id", "source_key"]].astype(str).merge(
            sub[["sample_id", "source_key"]].astype(str), on="sample_id",
            how="outer", validate="one_to_one", indicator=True, suffixes=("_train", "_crossfit"))
        if len(merged) != len(train) or not merged._merge.eq("both").all():
            raise RuntimeError(f"{scheme} outer {outer}: sample IDs differ from outer train")
        if not merged.source_key_train.eq(merged.source_key_crossfit).all():
            raise RuntimeError(f"{scheme} outer {outer}: source identities mismatch")
        if sub.inner_fold.nunique() != 4:
            raise RuntimeError(f"{scheme} outer {outer}: expected four auxiliary CV folds")
    return cf


def source_cluster_draws(source_codes: np.ndarray, n_sources: int, bootstrap: int, seed: int):
    rng = np.random.default_rng(seed)
    draws = np.zeros((bootstrap, n_sources), dtype=np.int32)
    for i in range(bootstrap):
        sampled = rng.integers(0, n_sources, size=n_sources)
        draws[i] = np.bincount(sampled, minlength=n_sources)
    return draws


def stat_arrays(df: pd.DataFrame, q: np.ndarray, n_sources: int):
    idx = df._source_code.to_numpy(dtype=int)
    err = df.abs_error.to_numpy(dtype=float)
    sums = np.zeros((4, n_sources), dtype=float)
    cnts = np.zeros((4, n_sources), dtype=float)
    for j, state in enumerate(Q_STATES):
        mask = (q == state)
        sums[j] = np.bincount(idx, weights=err*mask, minlength=n_sources)
        cnts[j] = np.bincount(idx, weights=mask.astype(float), minlength=n_sources)
    return sums, cnts


def bootstrap_stat(df: pd.DataFrame, labels: dict[str,np.ndarray], nboot: int, seed: int):
    source_list = np.sort(df.source_key.astype(str).unique())
    src_to_idx = {s: i for i, s in enumerate(source_list)}
    df = df.copy()
    df["_source_code"] = df.source_key.astype(str).map(src_to_idx).astype(int)
    nsrc = len(source_list)
    draws = source_cluster_draws(df._source_code.to_numpy(), nsrc, nboot, seed)
    out_rows=[]
    arrays = {}
    baseline = labels["training_LOO_P50"]
    base_q4_minus_q1 = None
    for scenario in labels:
        sums, cnts = stat_arrays(df, labels[scenario], nsrc)
        sum_boot = draws @ sums.T
        n_boot = draws @ cnts.T
        total = n_boot.sum(axis=1)
        mae_boot = np.full_like(sum_boot, np.nan, dtype=float)
        np.divide(sum_boot, n_boot, out=mae_boot, where=n_boot>0)
        share_boot = n_boot / total[:, None]
        d = mae_boot[:,3] - mae_boot[:,0]
        arrays[scenario] = {"mae": mae_boot, "share": share_boot, "gap": d}
        q1mask = labels[scenario] == "Q1"
        q4mask = labels[scenario] == "Q4"
        err = df.abs_error.to_numpy(float)
        stats = {"Q1_MAE": float(err[q1mask].mean()) if q1mask.any() else float("nan"),
                 "Q4_MAE": float(err[q4mask].mean()) if q4mask.any() else float("nan"),
                 "Q4_minus_Q1_MAE": float(err[q4mask].mean()-err[q1mask].mean()) if q1mask.any() and q4mask.any() else float("nan")}
        for j, st in enumerate(Q_STATES):
            stats[f"fraction_{st}"] = float((labels[scenario] == st).mean())
            stats[f"MAE_{st}"] = float(err[labels[scenario] == st].mean()) if np.any(labels[scenario] == st) else float("nan")
        stats["fraction_Q_changed_vs_LOO"] = float(np.mean(labels[scenario] != baseline))
        nchange_per_source = np.bincount(df._source_code.to_numpy(int),
                                          weights=(labels[scenario]!=baseline).astype(float),
                                          minlength=nsrc)
        vals = (draws @ nchange_per_source) / total
        for key, estimate in stats.items():
            if key == "Q1_MAE": sample=mae_boot[:,0]
            elif key == "Q4_MAE": sample=mae_boot[:,3]
            elif key == "Q4_minus_Q1_MAE": sample=d
            elif key.startswith("fraction_Q") and key != "fraction_Q_changed_vs_LOO":
                st=key.replace("fraction_", ""); sample=share_boot[:,Q_STATES.index(st)]
            elif key.startswith("MAE_Q"):
                st=key.replace("MAE_", ""); sample=mae_boot[:,Q_STATES.index(st)]
            elif key == "fraction_Q_changed_vs_LOO": sample=vals
            else: continue
            valid = sample[np.isfinite(sample)]
            ci_lo,ci_hi = np.percentile(valid, [2.5,97.5]) if len(valid) >= 20 else (np.nan,np.nan)
            out_rows.append({"scenario":scenario,"metric":key,"estimate":estimate,
                             "ci95_low":float(ci_lo),"ci95_high":float(ci_hi),
                             "n_bootstrap_valid":len(valid)})
    ref = arrays["training_LOO_P50"]["gap"]
    for scenario in ("source_crossfit_rank", "random_crossfit_rank"):
        if scenario not in arrays: continue
        alt = arrays[scenario]["gap"]
        diff = alt - ref
        valid = diff[np.isfinite(diff)]
        m = df.abs_error.to_numpy(float)
        q0,qa = baseline,labels[scenario]
        original=float(m[q0=="Q4"].mean()-m[q0=="Q1"].mean())
        alternate=float(m[qa=="Q4"].mean()-m[qa=="Q1"].mean())
        ci_lo,ci_hi=np.percentile(valid,[2.5,97.5]) if len(valid)>=20 else (np.nan,np.nan)
        out_rows.append({"scenario":scenario,"metric":"delta_Q4_minus_Q1_gap_vs_LOO",
                         "estimate":alternate-original,"ci95_low":float(ci_lo),
                         "ci95_high":float(ci_hi),"n_bootstrap_valid":len(valid)})
    return pd.DataFrame(out_rows)


def summary_table(df: pd.DataFrame, labels: dict[str,np.ndarray]) -> pd.DataFrame:
    base=labels["training_LOO_P50"]
    err=numeric(df,"abs_error")
    records=[]
    for sc,q in labels.items():
        row={"scenario":sc,"n":len(df),"n_sources":df.source_key.nunique(),
             "Q_agreement_vs_training_LOO":float(np.mean(q==base)),
             "n_changed_vs_training_LOO":int(np.sum(q!=base)),
             "MED_low_fraction":float(np.mean(np.isin(q, ["Q1","Q2"]))),
             "PED_low_fraction":float(np.mean(np.isin(q, ["Q1","Q3"])))}
        for state in Q_STATES:
            mask=(q==state)
            row[f"{state}_count"]=int(mask.sum())
            row[f"{state}_MAE"]=float(np.mean(err[mask])) if np.any(mask) else float('nan')
            row[f"{state}_fraction"]=float(np.mean(mask))
        row["Q4_minus_Q1_MAE"] = row["Q4_MAE"] - row["Q1_MAE"]
        records.append(row)
    return pd.DataFrame(records)


def save_excel(path: Path, sheets: dict[str, pd.DataFrame]):
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("openpyxl not available: CSV/JSON outputs remain complete", flush=True)
        return
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws=writer.sheets[name[:31]]
            ws.freeze_panes="A2"
            ws.auto_filter.ref=ws.dimensions
            for c in ws[1]:
                c.fill=PatternFill("solid",fgColor="17365D")
                c.font=Font(bold=True,color="FFFFFF")
                c.alignment=Alignment(vertical="center")
            for i,c in enumerate(df.columns,1):
                lengths=[len(str(x)) for x in df[c].head(30).dropna()]
                ws.column_dimensions[get_column_letter(i)].width=min(42,max(12,len(str(c))+2,max(lengths,default=0)+1))
            for r in ws.iter_rows(min_row=2):
                for c in r:
                    if isinstance(c.value,float): c.number_format="0.000000"
    print("Excel saved:",path,flush=True)


def main():
    args=get_args()
    root=Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd()
    formal=located(root,args.formal_dir)
    crossfit=located(root,args.crossfit_dir)
    data=located(root,args.data_dir)
    out=located(root,args.output_dir)
    for protected in (formal,crossfit,data):
        if out == protected or protected in out.parents or out in protected.parents:
            raise RuntimeError("Output directory must be disjoint from input directories")
    if out == root:
        raise RuntimeError("Output cannot be project root")
    official=validate_official(formal,args.expected_records)
    cf=load_crossfit(crossfit/"crossfit_heldout_MED.csv", "source", data)
    random_path=crossfit/"random_control_crossfit_heldout_MED.csv"
    random_cf=load_crossfit(random_path,"random_control",data) if random_path.is_file() else None
    if random_cf is None:
        print("NOTICE: Random control CSV absent; comparison will use source crossfit only",flush=True)

    thr_csv=pd.read_csv(formal/"fold_training_reference_thresholds.csv")
    require_columns(thr_csv,("fold","percentile","MED_threshold","PED_threshold"),"official thresholds")
    print("="*96,flush=True)
    print("NESTED MED CROSS-FITTED CALIBRATION SENSITIVITY (NO MODEL FITTING)",flush=True)
    print("Project:",root,"| OOF:",formal,"| Crossfit:",crossfit,flush=True)
    print("Reference regimes: fitted training-LOO P50 / source-crossfit rank / random-crossfit rank",flush=True)
    print(f"Frozen OOF={len(official)} | sources={official.source_key.nunique()} | bootstrap={args.bootstrap}",flush=True)
    print("Auxiliary percentile ranks are mapped to each outer model's OWN training LOO distribution.",flush=True)
    print("WARNING: Rank transport is a sensitivity proxy; not an unbiased calibration guarantee.",flush=True)
    print("="*96,flush=True)

    fold_rows=[]
    rank_rows=[]
    oof_parts=[]
    for fold in FOLDS:
        test=official.loc[official.fold.eq(fold)].copy()
        ref=pd.read_csv(formal/f"fold{fold}_training_reference_evidence.csv",low_memory=False)
        require_columns(ref,("sample_id","MED"),f"frozen Fold {fold} reference")
        train=pd.read_csv(data/f"fold{fold}_train.csv",low_memory=False)
        if len(ref)!=len(train) or not ref.sample_id.astype(str).tolist()==train.sample_id.astype(str).tolist():
            raise RuntimeError(f"Fold {fold}: frozen training ref does not match original train order")
        if not np.isfinite(ref.MED.to_numpy(float)).all():
            raise RuntimeError(f"Fold {fold}: invalid training MED")
        tbl=thr_csv.loc[(thr_csv.fold==fold)&(thr_csv.percentile==50)]
        if len(tbl)!=1:
            raise RuntimeError(f"Fold {fold}: missing/duplicate official MED and PED P50")
        old_med=float(tbl.MED_threshold.iloc[0]); old_ped=float(tbl.PED_threshold.iloc[0])
        derived=float(np.percentile(ref.MED.to_numpy(float),50))
        if abs(derived-old_med)>2e-9:
            raise RuntimeError(f"Fold {fold}: baseline P50 mismatch with reference")
        if (np.max(np.abs(test.MED_threshold_P50_fold-old_med))>2e-9 or
            np.max(np.abs(test.PED_threshold_P50_fold-old_ped))>2e-9):
            raise RuntimeError(f"Fold {fold}: per-record official thresholds inconsistent")
        ranks={"training_LOO_P50":50.0}
        for scheme,frame in (("source_crossfit_rank",cf),("random_crossfit_rank",random_cf)):
            if frame is None: continue
            t=frame.loc[frame.outer_fold.eq(fold)].copy()
            rank=numeric(t,"MED_percentile_vs_train_LOO")
            ranks[scheme]=float(np.median(rank))
            source_med=t.groupby("source_key")["MED_percentile_vs_train_LOO"].median()
            rank_rows.append({"fold":fold,"scheme":scheme,"n":len(t),
                              "n_sources":t.source_key.nunique(),
                              "crossfit_external_MED_rank_P50":float(np.median(rank)),
                              "crossfit_external_MED_rank_mean":float(np.mean(rank)),
                              "source_equal_weight_MED_rank_P50":float(np.median(source_med)),
                              "crossfit_external_above_own_train_P50":float(np.mean(t.MED>t.training_ref_MED_P50)),
                              "rank_iqr_low":float(np.percentile(rank,25)),
                              "rank_iqr_high":float(np.percentile(rank,75))})
        row={"fold":fold,"n_outer_train":len(train),"n_outer_test":len(test),
             "n_test_sources":test.source_key.nunique(),
             "official_train_LOO_P50":old_med,"official_PED_P50":old_ped}
        for scheme,p_rank in ranks.items():
            threshold=float(np.percentile(ref.MED.to_numpy(float),p_rank))
            row[f"{scheme}_rank_percentile"]=p_rank
            row[f"{scheme}_MED_threshold"]=threshold
            row[f"{scheme}_delta_threshold_vs_LOO"]=threshold-old_med
            row[f"{scheme}_MED_low_fraction_in_outer_test"]=float(np.mean(test.MED<=threshold))
            test[f"MED_threshold_{scheme}"]=threshold
            test[f"MED_low_{scheme}"]=test.MED.to_numpy(float)<=threshold
            test[f"Q_{scheme}"]=calculate_q(test.MED,test.PED_log1p,threshold,old_ped)
        if not np.array_equal(test["Q_training_LOO_P50"].to_numpy(dtype=str),
                              test.quadrant.to_numpy(dtype=str)):
            raise RuntimeError(f"Fold {fold}: official Q labels fail strict reproduction")
        fold_rows.append(row)
        oof_parts.append(test)
        info=" ".join(f"{s} P{p:.1f} MED={row[f'{s}_MED_threshold']:.5f}" for s,p in ranks.items())
        print(f"Fold {fold}: n_test={len(test)} | {info}",flush=True)
    oof=pd.concat(oof_parts,ignore_index=True)
    if len(oof)!=len(official) or oof.sample_id.astype(str).duplicated().any():
        raise RuntimeError("OOF sample coverage error")
    q_labels={s:oof[f"Q_{s}"].to_numpy(dtype=str) for s in SCENARIOS if f"Q_{s}" in oof}
    summary=summary_table(oof,q_labels)
    boot=bootstrap_stat(oof,q_labels,args.bootstrap,args.seed)
    folds=pd.DataFrame(fold_rows)
    ranks=pd.DataFrame(rank_rows)

    print("\n"+"="*96,flush=True)
    print("FINAL NESTED CROSSFIT CALIBRATION SENSITIVITY",flush=True)
    cols=["scenario","Q1_count","Q2_count","Q3_count","Q4_count","Q_agreement_vs_training_LOO",
          "Q1_MAE","Q4_MAE","Q4_minus_Q1_MAE"]
    print(summary[cols].to_string(index=False,float_format=lambda x:f"{x:.5f}"),flush=True)
    print("\nSELECTED SOURCE-CLUSTER BOOTSTRAP 95% CI",flush=True)
    selected=("Q4_minus_Q1_MAE","Q1_MAE","Q4_MAE","fraction_Q_changed_vs_LOO","delta_Q4_minus_Q1_gap_vs_LOO")
    print(boot.loc[boot.metric.isin(selected)].to_string(index=False,float_format=lambda x:f"{x:.5f}"),flush=True)
    print("\nFOLD REFERENCE RANK TRANSPORT",flush=True)
    print(folds[[c for c in folds.columns if c in ("fold","official_train_LOO_P50") or c.endswith("_rank_percentile") or c.endswith("_MED_threshold")]].to_string(index=False,float_format=lambda x:f"{x:.5f}"),flush=True)
    print("IMPORTANT: rank-transport reference changes the interpretation of 'low MED'.",flush=True)
    print("IMPORTANT: crossfit auxiliary theta_i was chosen on full outer train.",flush=True)
    print("IMPORTANT: bootstrap is conditional on FROZEN models and threshold choices.",flush=True)

    out.mkdir(parents=True,exist_ok=True)
    csv_write(folds,out/"fold_calibration_thresholds.csv")
    csv_write(ranks,out/"source_crossfit_rank_summary.csv")
    csv_write(summary,out/"calibration_scenario_summary.csv")
    csv_write(boot,out/"bootstrap_source_cluster_CI.csv")
    selected_columns=["fold","sample_id","source_key","y_true","y_pred","abs_error","MED","PED_log1p","quadrant"]
    selected_columns += [c for c in oof.columns if c.startswith(("MED_threshold_training_LOO_P50","MED_threshold_source_crossfit_rank","MED_threshold_random_crossfit_rank","Q_training_LOO_P50","Q_source_crossfit_rank","Q_random_crossfit_rank","MED_low_training_LOO_P50","MED_low_source_crossfit_rank","MED_low_random_crossfit_rank"))]
    csv_write(oof[selected_columns],out/"oof_calibration_comparison.csv")
    config={"project_root":str(root),"formal_dir":str(formal),"crossfit_dir":str(crossfit),
            "output_dir":str(out),"bootstrap":args.bootstrap,"seed":args.seed,
            "scenarios":list(q_labels),"n_oof_records":len(oof),"n_sources":oof.source_key.nunique(),
            "mapping":"Median aux held-out MED percentile rank (relative to its own LOO training reference) transported to matching outer frozen model LOO training percentile",
            "limitations":["not unbiased calibration","different reference semantics","theta_i selected on entire outer train","no model refitting in bootstrap","crossfit outer train observations recur across separate outer calibrations"]}
    (out/"run_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    if not args.no_excel:
        save_excel(out/"calibration_sensitivity.xlsx",{
            "Scenario_Summary":summary,"Fold_Thresholds":folds,"Rank_References":ranks,
            "Bootstrap_CIs":boot,"OOF_States":oof[selected_columns]})
    print("\nALL CHECKS PASSED. Files saved to:",out,flush=True)


if __name__=="__main__":
    main()
