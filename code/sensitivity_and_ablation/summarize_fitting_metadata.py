#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect kinetic-fitting metadata WITHOUT filtering or re-training any model.

Run from the existing project root:
    python summarize_fitting_metadata.py

Inputs (unchanged):
    splits/source_kfold/fold1_test.csv ... fold5_test.csv
    models/nested_evidence_final/pooled_source_oof_evidence.csv  [optional]

Outputs:
    models/kinetic_fitting_quality_sensitivity/diagnostic/
        report.txt
        fitting_metadata_distribution.csv
        n_points_frequency.csv
        subset_feasibility.csv
        sample_fitting_metadata.csv

NOTE: split_dataset.py calculates n_points from the count of `Time` rows in sheet2,
not necessarily the number of points in the actual selected parabolic-fit interval.
Likewise t_max - t_min is the time span in the saved sheet2, NOT automatically
verified to be a parabolic fitting window. Verify both with the original data.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_DIR = Path('splits/source_kfold')
OOF_CANDIDATES = [
    Path('models/nested_evidence_final/pooled_source_oof_evidence.csv'),
    Path('pooled_source_oof_evidence.csv'),
]
OUTPUT_DIR = Path('models/kinetic_fitting_quality_sensitivity/diagnostic')
N_THRESHOLDS = [4, 5, 6, 8, 10, 12, 15, 20, 30]
R2_THRESHOLDS = [0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995]
N_MIN_Q1_FOR_DIAGNOSTIC = 10
N_MIN_Q4_FOR_DIAGNOSTIC = 20


class Reporter:
    def __init__(self):
        self.lines = []

    def say(self, s=''):
        s = str(s)
        print(s)
        self.lines.append(s)

    def table(self, title, data, max_rows=40):
        self.say('\n' + title)
        if data is None or data.empty:
            self.say('(none)')
            return
        with pd.option_context('display.max_rows', max_rows,
                               'display.max_columns', 30,
                               'display.width', 220,
                               'display.max_colwidth', 60):
            self.say(data.head(max_rows).to_string(index=False))
        if len(data) > max_rows:
            self.say(f'... {len(data) - max_rows} further rows in saved CSV')


def find_column(frame, options):
    norm = {str(c).strip().lower().replace(' ', '').replace('_',''): c for c in frame.columns}
    for option in options:
        k = option.strip().lower().replace(' ', '').replace('_', '')
        if k in norm:
            return norm[k]
    return None


def load_source_disjoint_tests(rep):
    paths = [SPLIT_DIR / f'fold{i}_test.csv' for i in range(1, 6)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing source-aware fold test files:\n' + '\n'.join(missing)
                                + '\nPlace this script in your original project root.')
    frames = []
    for fold, path in enumerate(paths, start=1):
        a = pd.read_csv(path)
        a.columns = a.columns.str.strip()
        if 'sample_id' not in a.columns or 'source_key' not in a.columns:
            raise ValueError(f'{path}: missing sample_id or source_key. Columns={list(a.columns)}')
        a['sample_id'] = a['sample_id'].astype('string').str.strip()
        if a['sample_id'].isna().any() or a['sample_id'].duplicated().any():
            raise ValueError(f'{path}: null/duplicated sample_id')
        a['_split_fold'] = fold
        frames.append(a)
    df = pd.concat(frames, ignore_index=True, sort=False)
    if df.sample_id.duplicated().any():
        raise ValueError('The same sample_id occurs in more than one held-out fold.')
    for i, a in enumerate(frames):
        for j, b in enumerate(frames):
            if i < j:
                overlap = set(a.source_key.dropna()) & set(b.source_key.dropna())
                if overlap:
                    raise ValueError(f'Source overlap between held-out folds {i+1} and {j+1}: {len(overlap)}')
    rep.say(f'Input: 5 held-out source-aware fold test CSVs, N={len(df)}, sources={df.source_key.nunique()}')
    rep.say('Original source-aware fold assignments are PRESERVED. No split is regenerated.')
    return df


def load_optional_oof(df, rep):
    path = next((p for p in OOF_CANDIDATES if p.exists()), None)
    if path is None:
        rep.say('\n[INFO] Formal pooled OOF CSV not found; diagnostic is limited to metadata distributions.')
        return df, False
    oof = pd.read_csv(path)
    oof.columns = oof.columns.str.strip()
    if 'sample_id' not in oof.columns:
        rep.say(f'[WARN] OOF exists but sample_id missing: {path}')
        return df, False
    oof['sample_id'] = oof['sample_id'].astype('string').str.strip()
    if oof.sample_id.duplicated().any():
        raise ValueError('OOF has duplicated sample_id')
    state = find_column(oof, ['quadrant', 'evidence_state', 'state', 'Q_state', 'quadrant_P50'])
    if not state:
        rep.say(f'[WARN] OOF has no recognized Q-state column. Available columns: {list(oof.columns)}')
        return df, False
    extra = ['sample_id', state]
    if 'fold' in oof.columns:
        extra.append('fold')
    m = df.merge(oof[extra].rename(columns={state: '_Q_state', 'fold':'_oof_fold'}),
                 on='sample_id', how='left', validate='one_to_one', indicator=True)
    if (m['_merge'] != 'both').any() or len(m) != len(oof):
        raise ValueError(f'Sample coverage mismatch split vs OOF: split={len(df)}, OOF={len(oof)}, '
                         f'unmatched_split={int((m._merge!="both").sum())}')
    if '_oof_fold' in m.columns and (m['_oof_fold'].astype(int) != m['_split_fold'].astype(int)).any():
        raise ValueError('OOF and fold-test split disagree on fold assignment.')
    m = m.drop(columns=['_merge'])
    rep.say(f'OOF: {path}; Q-state column={state}; sample/fold alignment passed')
    return m, True


def numeric_col(df, possibilities, rep, caption, required=False):
    col = find_column(df, possibilities)
    if col is None:
        msg = f'{caption}: NOT FOUND (tried: {possibilities})'
        if required:
            raise KeyError(msg + '\nAvailable columns: ' + str(list(df.columns)))
        rep.say('[WARN] ' + msg)
        return None
    df[caption] = pd.to_numeric(df[col], errors='coerce')
    rep.say(f'{caption} <= column `{col}`; missing/non-numeric: {df[caption].isna().sum()}')
    return caption


def describe_numeric(df, name):
    v = df[name].dropna()
    if len(v) == 0:
        return dict(metric=name, non_missing=0, missing=len(df))
    qs = v.quantile([0, .10, .25, .5, .75, .90, .95, .99, 1.0])
    return dict(metric=name, non_missing=len(v), missing=len(df)-len(v),
                min=qs.loc[0], p10=qs.loc[.10], p25=qs.loc[.25], median=qs.loc[.5],
                p75=qs.loc[.75], p90=qs.loc[.90], p95=qs.loc[.95], p99=qs.loc[.99], max=qs.loc[1.0])


def feasibility(df, selection, label, has_oof):
    sub = df.loc[selection].copy()
    result = dict(subset=label, N=len(sub), pct_of_613=round(len(sub)/len(df)*100, 1),
                  sources=sub.source_key.nunique(),
                  median_n=round(sub.n_points_diag.median(),2) if len(sub) else np.nan,
                  median_fit_R2=round(sub.fit_R2_diag.median(),5) if len(sub) else np.nan)
    if has_oof:
        for state in ('Q1','Q2','Q3','Q4'):
            g = sub.loc[sub['_Q_state'].astype(str).str.upper() == state]
            result[f'{state}_N'] = len(g)
            result[f'{state}_sources'] = g.source_key.nunique()
        result['sufficient_Q1_Q4_for_exploration'] = bool(
            result['Q1_N'] >= N_MIN_Q1_FOR_DIAGNOSTIC and
            result['Q4_N'] >= N_MIN_Q4_FOR_DIAGNOSTIC and
            result['Q1_sources'] >= 5 and result['Q4_sources'] >= 10
        )
    return result


def main():
    rep = Reporter()
    rep.say('=' * 106)
    rep.say('KINETIC FITTING METADATA | DIAGNOSTIC ONLY (NO QUALITY LABELS, NO MODEL REFITTING)')
    rep.say('=' * 106)
    df = load_source_disjoint_tests(rep)
    df, has_oof = load_optional_oof(df, rep)
    rep.say('\nColumns in fold test CSV:')
    rep.say(', '.join(map(str, df.columns)))

    numeric_col(df, ['n_points', 'n_fit', 'fit_n', 'nfit'], rep, 'n_points_diag', True)
    numeric_col(df, ['R2_fit', 'fit_R2', 'R2', 'r_squared', 'fit_r2'], rep, 'fit_R2_diag', True)
    for name, possibilities in [
        ('t_min_diag', ['t_min', 'fit_t_start', 'fit_start_time']),
        ('t_max_diag', ['t_max', 'fit_t_end', 'fit_end_time']),
        ('E_Time_diag', ['E_Time', 'oxidation_time', 'exposure_time']),
        ('Temperature_diag', ['Temperature', 'oxidation_temperature']),
    ]:
        numeric_col(df, possibilities, rep, name)
    if {'t_min_diag','t_max_diag'}.issubset(df.columns):
        df['recorded_time_span_diag'] = df.t_max_diag - df.t_min_diag
        rep.say('recorded_time_span_diag = t_max - t_min (only a FIT window if sheet2 contains fitted points).')

    desc_cols = [c for c in ['n_points_diag','fit_R2_diag','t_min_diag','t_max_diag',
                             'recorded_time_span_diag','E_Time_diag','Temperature_diag'] if c in df.columns]
    summary = pd.DataFrame([describe_numeric(df, c) for c in desc_cols]).round(5)
    rep.table('1) METADATA DISTRIBUTION (quantiles)', summary)
    rep.say('\n[Validity check]')
    rep.say(f'  n_points < 4: {(df.n_points_diag < 4).sum()} | n_points NaN: {df.n_points_diag.isna().sum()}')
    rep.say(f'  fit R2 < 0.80: {(df.fit_R2_diag < .8).sum()} | fit R2 > 1: {(df.fit_R2_diag > 1).sum()} | fit R2 NaN: {df.fit_R2_diag.isna().sum()}')
    if 'recorded_time_span_diag' in df.columns:
        rep.say(f'  recorded time span <= 0: {(df.recorded_time_span_diag <= 0).sum()}')
    rep.say('  NOTE: Current split_dataset.py uses n_points >= 0 (not >= 4) as its programmatic filter.')
    rep.say('  Verify that n_points counts points SELECTED for parabolic fitting rather than every TGA time point.')

    nfreq = (df['n_points_diag'].value_counts(dropna=False).sort_index()
             .rename_axis('n_points').reset_index(name='count'))
    nfreq['fraction_pct'] = (nfreq['count'] / len(df) * 100).round(2)
    rep.table('2) EXACT FREQUENCY OF n_points', nfreq, 45)

    r2_bins = [-np.inf, .8, .9, .95, .97, .98, .99, .995, 1.00000000001, np.inf]
    r2_labels = ['<0.80', '[0.80,0.90)', '[0.90,0.95)', '[0.95,0.97)',
                 '[0.97,0.98)', '[0.98,0.99)', '[0.99,0.995)', '[0.995,1.00]', '>1.00']
    rb = pd.cut(df.fit_R2_diag, bins=r2_bins, labels=r2_labels, right=False)
    rfreq = (rb.value_counts(sort=False).rename_axis('fit_R2_bin').reset_index(name='count'))
    rfreq['fraction_pct'] = (rfreq['count'] / len(df) * 100).round(2)
    rep.table('3) FIT R2 FREQUENCY BY RANGE', rfreq)

    allmask = df.n_points_diag.notna() & df.fit_R2_diag.notna()
    out = [feasibility(df, allmask, 'Both fit metadata available', has_oof)]
    for n in N_THRESHOLDS:
        out.append(feasibility(df, allmask & (df.n_points_diag >= n), f'n_points >= {n}', has_oof))
    for r in R2_THRESHOLDS:
        out.append(feasibility(df, allmask & (df.fit_R2_diag >= r), f'fit R2 >= {r:g}', has_oof))
    for n in [5, 8, 10, 15]:
        for r in [.90, .95, .98, .99]:
            out.append(feasibility(df, allmask & (df.n_points_diag >= n) & (df.fit_R2_diag >= r),
                                   f'n_points >= {n} & fit R2 >= {r:g}', has_oof))
    feasibility_df = pd.DataFrame(out)
    rep.table('4) SUBSET COVERAGE / Q-STATE FEASIBILITY (not prediction accuracy)', feasibility_df, 55)

    if 'recorded_time_span_diag' in df.columns:
        vals = df.recorded_time_span_diag.dropna()
        cuts = sorted(set(float(vals.quantile(p)) for p in [.25,.5,.75]))
        spanrows=[]
        for v in cuts:
            spanrows.append(feasibility(df, allmask & (df.recorded_time_span_diag >= v),
                                        f'recorded_time_span >= {v:.4g} [UNVERIFIED units/window]', has_oof))
        rep.table('5) RECORDED TIME SPAN (exploratory ONLY; verify time unit and fitting window)',
                  pd.DataFrame(spanrows))

    rep.say('\nNEXT STEPS:')
    rep.say('  1. Confirm R2 is the parabolic kinetic FIT R2 (not the ML model R2).')
    rep.say('  2. Confirm sheet2 Time rows are exactly those selected for the parabolic regime; verify units.')
    rep.say('  3. Choose 2-3 stricter thresholds using fit metadata distribution and sample/source coverage.')
    rep.say('  4. Fix original 5 held-out folds and original OOF predictions/evidence states; compare subsets.')
    rep.say('  5. Do NOT label lower-n or lower-R2 retained records as low quality.')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR/'fitting_metadata_distribution.csv', index=False)
    nfreq.to_csv(OUTPUT_DIR/'n_points_frequency.csv', index=False)
    rfreq.to_csv(OUTPUT_DIR/'fit_R2_frequency.csv', index=False)
    feasibility_df.to_csv(OUTPUT_DIR/'subset_feasibility.csv', index=False)
    cols = ['sample_id','source_key','_split_fold','n_points_diag','fit_R2_diag']
    cols += [c for c in ['t_min_diag','t_max_diag','recorded_time_span_diag',
                         'E_Time_diag','Temperature_diag','_Q_state'] if c in df.columns]
    df[cols].to_csv(OUTPUT_DIR/'sample_fitting_metadata.csv', index=False)
    (OUTPUT_DIR/'report.txt').write_text('\n'.join(rep.lines)+'\n', encoding='utf-8')
    rep.say(f'\n[SAVED] Reports and CSV files -> {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\n[ERROR] {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
