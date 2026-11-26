#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prediction sensitivity to kinetic fitting quality.

Run from the EXISTING project root:
    python evaluate_fitting_quality.py
    # Jupyter: !python evaluate_fitting_quality.py

Input (no need to re-read data.xlsx or change splits/models):
    splits/source_kfold/fold1_test.csv ... fold5_test.csv
    models/nested_evidence_final/pooled_source_oof_evidence.csv
    (or ./pooled_source_oof_evidence.csv)

Output (new directory only):
    models/kinetic_fitting_quality_sensitivity/evaluation/
      summary.txt, subset_overview.csv, state_specific_mae.csv,
      source_cluster_bootstrap.csv, per_sample_with_fit_metadata.csv,
      fit_metadata_vs_abs_error.csv, analysis_design.txt

All XGBoost predictions, held-out outer folds, MED/PED, P50 thresholds,
Q1-Q4 states and hyperparameters are FROZEN: NO re-fitting, no new split.
The sensitivity is restricted to different EVALUATION subsets, NOT training
on only the restricted records. Selection thresholds are declared up front.

CAVEAT: R2 in sheet1 is assumed to be the PARABOLIC-FIT R2, and n_points from
sheet2 is assumed to count SELECTED fitting points, not all raw TGA readings.
Confirm these from the original records. The script does NOT claim that
high R2 and high n prove physical parabolic regime or absence of spallation.

The (optional) time-window sensitivity must be explicitly enabled ONLY after
confirming that t_min / t_max describe selected parabolic-fitting points in
known time units. Default analysis EXCLUDES time-window selection.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STATES = ('Q1', 'Q2', 'Q3', 'Q4')


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='Project root containing splits/ and models/')
    ap.add_argument('--oof', default='', help='Optional explicit OOF path')
    ap.add_argument('--bootstrap', type=int, default=3000, help='Source-cluster resamples; 0 disables')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--expected-n', type=int, default=613, help='0 bypasses record-count check')
    ap.add_argument('--include-time-window', action='store_true', help='Only after verifying t_min/t_max represent FIT window')
    ap.add_argument('--time-unit', default='', help='Required when --include-time-window; e.g. h')
    args = ap.parse_args()
    if args.bootstrap < 0 or args.expected_n < 0:
        ap.error('bootstrap and expected-n must be >=0')
    if args.include_time_window and not args.time_unit.strip():
        ap.error('--time-unit must be set if using --include-time-window')
    return args


class Reporter:
    def __init__(self): self.lines = []
    def say(self, s=''):
        s = str(s)
        print(s)
        self.lines.append(s)
    def table(self, title, df, cols=None):
        self.say('\n' + title)
        if cols is not None: df = df.loc[:, cols]
        with pd.option_context('display.max_columns', 30, 'display.width', 230,
                               'display.max_rows', 100, 'display.max_colwidth', 50):
            self.say(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


def find_oof(root, explicit):
    options = ([Path(explicit)] if explicit else []) + [
        Path('models/nested_evidence_final/pooled_source_oof_evidence.csv'),
        Path('pooled_source_oof_evidence.csv'),
    ]
    for opt in options:
        p = opt if opt.is_absolute() else root / opt
        if p.is_file(): return p
    raise FileNotFoundError('Official pooled OOF CSV not found; searched: ' + ', '.join(map(str, options)))


def load_verified_data(root, oof_path, expected_n):
    frames=[]
    for fold in range(1, 6):
        p = root / f'splits/source_kfold/fold{fold}_test.csv'
        if not p.is_file(): raise FileNotFoundError(f'Missing held-out split: {p}')
        d = pd.read_csv(p)
        d.columns = d.columns.str.strip()
        need = {'sample_id','source_key','lgkp','n_points','R2'}
        if not need.issubset(d): raise ValueError(f'{p}: missing {sorted(need - set(d))}')
        d['sample_id'] = d['sample_id'].astype('string').str.strip()
        d['source_key'] = d['source_key'].astype('string').str.strip()
        if d['sample_id'].isna().any() or d['sample_id'].duplicated().any():
            raise ValueError(f'{p}: missing or repeated sample_id')
        d['_split_fold'] = fold
        frames.append(d)
    meta = pd.concat(frames, ignore_index=True)
    if meta['sample_id'].isna().any() or meta['sample_id'].duplicated().any():
        raise ValueError('A sample appears in multiple held-out folds')
    if meta['source_key'].isna().any() or (meta.groupby('source_key')['_split_fold'].nunique() > 1).any():
        raise ValueError('A source appears in multiple held-out folds')
    if expected_n and len(meta) != expected_n:
        raise ValueError(f'Expected {expected_n} records in folds, found {len(meta)}')

    oof = pd.read_csv(oof_path)
    oof.columns = oof.columns.str.strip()
    required = {'sample_id','source_key','fold','quadrant','y_true','y_pred'}
    if not required.issubset(oof): raise ValueError(f'OOF missing: {sorted(required - set(oof))}')
    oof['sample_id'] = oof['sample_id'].astype('string').str.strip()
    oof['source_key'] = oof['source_key'].astype('string').str.strip()
    if oof['sample_id'].isna().any() or oof['sample_id'].duplicated().any():
        raise ValueError('OOF has missing/duplicated sample IDs')
    if len(oof) != len(meta): raise ValueError(f'OOF N={len(oof)}, held-out split N={len(meta)}')
    # Join by sample ID, never relying on file row order.
    keep = ['sample_id','source_key','fold','quadrant','y_true','y_pred']
    if 'abs_error' in oof: keep.append('abs_error')
    merged = meta.merge(oof[keep].rename(columns={'source_key':'_oof_source_key'}),
                        on='sample_id', how='outer', indicator=True, validate='one_to_one')
    if not merged['_merge'].eq('both').all(): raise ValueError('Split / OOF sample_id mismatch')
    merged.drop(columns=['_merge'], inplace=True)
    if not merged['source_key'].eq(merged['_oof_source_key']).all():
        raise ValueError('Split / OOF literature source mismatch')
    if not (merged['_split_fold'].astype(int) == merged['fold'].astype(int)).all():
        raise ValueError('Split / OOF fold mismatch')

    for col in ['lgkp','R2','n_points','y_true','y_pred']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
        if not np.isfinite(merged[col].to_numpy(float)).all():
            raise ValueError(f'{col}: missing/nonfinite values detected')
    if not np.allclose(merged['lgkp'], merged['y_true'], rtol=0, atol=1e-7):
        raise ValueError('OOF y_true differs from held-out split lgkp')
    if (merged['R2'] < -1e-10).any() or (merged['R2'] > 1.0000001).any():
        raise ValueError('The R2 metadata does not look like 0..1 goodness of fit')
    if (merged['n_points'] < 4).any():
        raise ValueError('Some records contain fewer than 4 recorded fitting points')
    merged['quadrant'] = merged['quadrant'].astype(str).str.strip().str.upper()
    if not merged['quadrant'].isin(STATES).all(): raise ValueError('Unrecognized Q-state labels')
    merged['abs_error'] = np.abs(merged['y_pred'] - merged['y_true'])
    if 'abs_error' in oof:
        official_abs = oof.set_index('sample_id')['abs_error'].astype(float)
        aligned_abs = merged['sample_id'].map(official_abs).to_numpy(float)
        if not np.allclose(merged['abs_error'], aligned_abs, rtol=0, atol=1e-8):
            raise ValueError('Official abs_error differs from computed y_true/y_pred')
    return merged


def scenarios(data, include_span=False, time_unit=''):
    n = data['n_points'].to_numpy(float)
    r = data['R2'].to_numpy(float)
    cases = [
        ('Baseline','All expert-reviewed records', np.ones(len(data), dtype=bool)),
        ('Fit R2','R2_fit >= 0.90', r >= 0.90),
        ('Fit R2','R2_fit >= 0.95', r >= 0.95),
        ('Fitting points','n_fit >= 6', n >= 6),
        ('Fitting points','n_fit >= 8', n >= 8),
        ('Joint moderate','n_fit >= 5 and R2_fit >= 0.90', (n >= 5) & (r >= .90)),
        ('Joint strict','n_fit >= 5 and R2_fit >= 0.95', (n >= 5) & (r >= .95)),
        ('Joint exploratory','n_fit >= 8 and R2_fit >= 0.95', (n >= 8) & (r >= .95)),
    ]
    if include_span:
        if not {'t_min','t_max'}.issubset(data):
            raise ValueError('t_min/t_max columns missing')
        tmin = pd.to_numeric(data['t_min'], errors='coerce').to_numpy(float)
        tmax = pd.to_numeric(data['t_max'], errors='coerce').to_numpy(float)
        if not np.isfinite(tmin).all() or not np.isfinite(tmax).all() or (tmax<=tmin).any():
            raise ValueError('Invalid t_min/t_max')
        span = tmax-tmin
        for p in (0.25, 0.50):
            cutoff = float(np.quantile(span,p))
            cases.append(('Time window [exploratory]', f'span >= p{int(p*100)} ({cutoff:.3f} {time_unit})', span >= cutoff))
        cases.append(('Joint time [exploratory]',
                      f'n_fit >= 5 and R2_fit >= 0.95 and span >= p25 {time_unit}',
                      (n>=5)&(r>=.95)&(span>=np.quantile(span,.25))))
    return cases


def metrics(y, pred):
    y=np.asarray(y, dtype=float); pred=np.asarray(pred,dtype=float)
    err = pred-y
    if len(y)<2 or np.sum((y-y.mean())**2)==0: r2=np.nan
    else: r2=1-np.sum(err*err)/np.sum((y-y.mean())**2)
    return float(r2),float(np.mean(np.abs(err))),float(np.sqrt(np.mean(err**2)))


def source_bootstrap(sub, draws, seed):
    """Resample entire publication clusters; retain fixed subset membership and states."""
    cols = ['MAE_all', *[f'MAE_{q}' for q in STATES], 'Q4_minus_Q1_MAE']
    if not draws: return {c:(np.nan,np.nan,0) for c in cols}
    groups = list(sub.groupby('source_key', sort=True, observed=True))
    ns=len(groups)
    if ns < 2: return {c:(np.nan,np.nan,0) for c in cols}
    sums={}; cnts={}
    sums['all'] = np.array([g['abs_error'].sum() for _,g in groups], dtype=float)
    cnts['all'] = np.array([len(g) for _,g in groups], dtype=float)
    for q in STATES:
        sums[q] = np.array([g.loc[g.quadrant.eq(q),'abs_error'].sum() for _,g in groups], dtype=float)
        cnts[q] = np.array([int(g.quadrant.eq(q).sum()) for _,g in groups], dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, ns, size=(draws, ns), dtype=np.int32)
    arr={}
    for key in ('all',*STATES):
        bsum = sums[key][picks].sum(axis=1)
        bcnt = cnts[key][picks].sum(axis=1)
        arr[key] = np.divide(bsum, bcnt, out=np.full(draws,np.nan), where=bcnt>0)
    arr['diff'] = arr['Q4']-arr['Q1']
    d={}
    for col, vals in [('MAE_all',arr['all']),*[(f'MAE_{q}',arr[q]) for q in STATES],
                      ('Q4_minus_Q1_MAE',arr['diff'])]:
        v = vals[np.isfinite(vals)]
        d[col] = (float(np.quantile(v,.025)), float(np.quantile(v,.975)), len(v)) if len(v) else (np.nan,np.nan,0)
    return d


def make_bin_report(data):
    d = data.copy()
    d['fit_R2_bin'] = pd.cut(d.R2, bins=[.8,.9,.95,.98,1.000001], right=False,
                             labels=['[0.80,0.90)','[0.90,0.95)','[0.95,0.98)','[0.98,1.00]'])
    d['n_points_bin'] = pd.cut(d.n_points, bins=[4,6,8,10,100], right=False,
                                labels=['4-5','6-7','8-9','10+'])
    out=[]
    for factor in ('fit_R2_bin','n_points_bin'):
        for key, grp in d.groupby(factor, observed=True):
            r2,mae,rmse=metrics(grp.y_true,grp.y_pred)
            out.append(dict(factor=factor, bin=str(key), N=len(grp), sources=grp.source_key.nunique(),
                            MAE=mae,RMSE=rmse, pooled_R2=r2,
                            Q1=int(grp.quadrant.eq('Q1').sum()),Q4=int(grp.quadrant.eq('Q4').sum())))
    return pd.DataFrame(out)


def main():
    args=cli(); root=Path(args.root).expanduser().resolve()
    out = root/'models/kinetic_fitting_quality_sensitivity/evaluation'
    out.mkdir(parents=True, exist_ok=True)
    rep=Reporter()
    rep.say('='*104)
    rep.say('KINETIC FITTING QUALITY SENSITIVITY')
    rep.say('='*104)
    oof_path=find_oof(root,args.oof)
    d=load_verified_data(root,oof_path,args.expected_n)
    rep.say(f'Source-aware held-out folds: N={len(d)}, sources={d.source_key.nunique()}')
    rep.say(f'Official frozen OOF: {oof_path}')
    rep.say(f'All fold/sample/true-label/source alignments PASS; models and Q states unchanged')
    rep.say(f'FIT metadata: n min={d.n_points.min():.0f}, median={d.n_points.median():.1f}; R2 min={d.R2.min():.5f}, median={d.R2.median():.5f}')
    rep.say(f'Source-cluster bootstrap draws={args.bootstrap}, seed={args.seed}; fixed OOF selections')
    rep.say('WARNING: R2/n_points are fit-quality PROXIES, not proof of parabolic oxidation or absence of spallation.')
    rep.say('WARNING: Results are EVALUATION SUBSETS of the same 613-record-trained fold models, NOT subset-trained models.')
    if args.include_time_window:
        rep.say(f'OPTIONAL fitted time-window analysis ENABLED after user confirmation; time unit={args.time_unit}')
    else:
        rep.say('Fitting time-window analysis DISABLED pending confirmation that Time rows represent fitted intervals.')
    overview=[]; states=[]; cis=[]
    for i,(family,label,mask) in enumerate(scenarios(d,args.include_time_window,args.time_unit)):
        sub=d.loc[mask].copy()
        if sub.empty: continue
        r2,mae,rmse=metrics(sub.y_true,sub.y_pred)
        counts=sub.quadrant.value_counts().reindex(STATES,fill_value=0)
        q1=sub.loc[sub.quadrant.eq('Q1')]; q4=sub.loc[sub.quadrant.eq('Q4')]
        q1mae=float(q1.abs_error.mean()) if len(q1) else np.nan
        q4mae=float(q4.abs_error.mean()) if len(q4) else np.nan
        delta=q4mae-q1mae if len(q1) and len(q4) else np.nan
        enough_sources=q1.source_key.nunique()>=10 and q4.source_key.nunique()>=20
        boot=source_bootstrap(sub,args.bootstrap,args.seed+i*7919)
        lo,hi,valid=boot['Q4_minus_Q1_MAE']
        overview.append(dict(category=family, subset=label, N=len(sub), pct_of_all=100*len(sub)/len(d),
                             N_sources=sub.source_key.nunique(), R2_pooled=r2, MAE=mae,RMSE=rmse,
                             Q1=int(counts.Q1),Q2=int(counts.Q2),Q3=int(counts.Q3),Q4=int(counts.Q4),
                             Q1_pct=100*counts.Q1/len(sub),Q2_pct=100*counts.Q2/len(sub),
                             Q3_pct=100*counts.Q3/len(sub),Q4_pct=100*counts.Q4/len(sub),
                             Q1_sources=q1.source_key.nunique(),Q4_sources=q4.source_key.nunique(),
                             Q1_MAE=q1mae,Q4_MAE=q4mae,delta_MAE_Q4_minus_Q1=delta,
                             delta_CI95_low=lo,delta_CI95_high=hi, valid_boot_draws=valid,
                             CI_interpretation_caution=(not enough_sources)))
        for q in STATES:
            qsub=sub.loc[sub.quadrant.eq(q)]
            sq=float(qsub.abs_error.mean()) if len(qsub) else np.nan
            x,y,nvalid=boot[f'MAE_{q}']
            states.append(dict(category=family,subset=label,state=q,N=len(qsub),
                               N_sources=qsub.source_key.nunique(), MAE=sq, CI95_low=x,CI95_high=y,
                               valid_boot_draws=nvalid))
        for name,(x,y,nvalid) in boot.items():
            cis.append(dict(category=family,subset=label,metric=name,CI95_low=x,CI95_high=y,
                            valid_boot_draws=nvalid))
    summary=pd.DataFrame(overview)
    state_df=pd.DataFrame(states)
    boot_df=pd.DataFrame(cis)
    bins=make_bin_report(d)
    summary.to_csv(out/'subset_overview.csv',index=False,encoding='utf-8-sig')
    state_df.to_csv(out/'state_specific_mae.csv',index=False,encoding='utf-8-sig')
    boot_df.to_csv(out/'source_cluster_bootstrap.csv',index=False,encoding='utf-8-sig')
    bins.to_csv(out/'fit_metadata_vs_abs_error.csv',index=False,encoding='utf-8-sig')
    d.to_csv(out/'per_sample_with_fit_metadata.csv',index=False,encoding='utf-8-sig')
    rep.table('1) EVALUATION-SUBSET PERFORMANCE + Q1/Q4 ERROR SEPARATION',summary,
              ['subset','N','N_sources','R2_pooled','MAE','RMSE','Q1','Q2','Q3','Q4',
               'Q1_sources','Q4_sources','Q1_MAE','Q4_MAE','delta_MAE_Q4_minus_Q1',
               'delta_CI95_low','delta_CI95_high','CI_interpretation_caution'])
    rep.table('2) STATE-SPECIFIC MAE & SOURCE-CLUSTER 95% CI',state_df,
              ['subset','state','N','N_sources','MAE','CI95_low','CI95_high'])
    rep.table('3) DESCRIPTIVE FIT-METADATA BINS (NOT A CAUSAL TEST)',bins)
    design = '''Analysis design and limitations\n\n'''+\
    ' - The original five literature-source-disjoint outer test folds are preserved.\n'+\
    ' - The same frozen nested XGBoost predictions and MED/PED Q states are reused.\n'+\
    ' - Each subset is formed only for evaluation, NOT for re-training or re-tuning.\n'+\
    ' - n_points and R2_fit are treated as continuous fitting-characteristic metadata.\n'+\
    ' - Larger n or R2_fit does not necessarily establish a physically valid parabolic regime.\n'+\
    ' - Sources may differ across subsets; subset performance changes may reflect composition,\n'+\
    '   temperature, exposure, paper, or label distributions, not just fitting metadata.\n'+\
    ' - Bootstrap resamples publication clusters within the chosen fixed subset; intervals\n'+\
    '   condition on models, original test folds, evidence states, and subset selections.\n'+\
    ' - Testing metadata-selected evaluation subsets cannot isolate the effect of lower-quality\n'+\
    '   labels retained in the original training folds.\n'+\
    ' - Data owner must verify R2 is parabolic-fit R2, n_points count selected fitting points,\n'+\
    '   and any optional time window represents the fitted parabolic interval.\n'
    (out/'analysis_design.txt').write_text(design, encoding='utf-8')
    rep.say('\n[SAVED] '+str(out))
    (out/'summary.txt').write_text('\n'.join(rep.lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
