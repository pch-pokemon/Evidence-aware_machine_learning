#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 5(d) and Table S10 for the FROZEN Nested source-aware temporal experiment.

This is the corrected v2 script. In particular, it uses the ACTUAL columns
written by compute_temporal_evidence.py, rather than recomputing
margins using nonexistent threshold field names.

Coordinates (historical and current, each relative to its own reference):
  x = PED_minus_threshold_P50_<era>  = ln(1+PED) - PED common P50
  y = MED_minus_threshold_P50_<era>  = MED - that model's MED P50

Historical/current MED magnitudes are MODEL-RELATIVE. The arrows visualize
Q-state changes across relative thresholds, not changes on one absolute MED
scale. PED is on a common physical reference scale.

Inputs, relative to project root:
  models/nested_temporal_evidence_2022_vs_current/temporal_comparison.csv
  models/nested_temporal_transition_audit/traced_topk_neighbor_records.csv
  splits/source_kfold/fold{1..5}_train.csv

Outputs, without overwriting old figures:
  figures/Fig5d_nested_final/
    Fig5d_representative_cases.png/pdf/svg
    fig5d_case_summary.csv
    Table_S10.csv
    Table_S10_neighborhood_summary.csv
    fig5d_case_outputs.xlsx (if an Excel engine is available)

Example:
  !python plot_temporal_cases.py
  !python plot_temporal_cases.py --sample-ids s099 s106
  !python plot_temporal_cases.py --project-root "C:/your/project"
  !python plot_temporal_cases.py --expected-records 0  # mock testing

No model fitting, no recalibration, no change to any official Nested result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

Q_STATES = ('Q1', 'Q2', 'Q3', 'Q4')
POINT_COLORS = {'historical_2022': '#326CA5', 'current': '#D77B25'}


def args_parse():
    ap = argparse.ArgumentParser(description='Draw Fig. 5(d) and export Table S10 for final Nested temporal OOF')
    ap.add_argument('--project-root', '--project_root', dest='project_root', default=None,
                    help='Default: directory containing this .py file')
    ap.add_argument('--sample-ids', '--sample_ids', dest='sample_ids', nargs='+', default=['s279','s125'])
    ap.add_argument('--output-dir', '--output_dir', dest='output_dir', default=None)
    ap.add_argument('--expected-records', type=int, default=613,
                    help='Total temporal OOF records. Use 0 only for synthetic smoke testing.')
    ap.add_argument('--no-excel', action='store_true', help='Skip optional XLSX output')
    return ap.parse_args()


def required(df: pd.DataFrame, cols, label: str):
    missing = sorted(set(cols).difference(df.columns))
    if missing:
        raise ValueError(f'{label} missing columns {missing}; available columns: {list(df.columns)}')


def numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    x = pd.to_numeric(df[col], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f'Non-finite values in required numeric column {col}')
    return x


def validate_temporal(df: pd.DataFrame, expected: int):
    core = ['sample_id','source_key','fold','y_true',
            'quadrant_2022','quadrant_current',
            'MED_2022','MED_current',
            'MED_threshold_P50_2022','MED_threshold_P50_current',
            'PED_for_threshold_2022','PED_for_threshold_current',
            'PED_threshold_P50_common',
            'MED_minus_threshold_P50_2022','MED_minus_threshold_P50_current',
            'PED_minus_threshold_P50_2022','PED_minus_threshold_P50_current',
            'PED_raw_2022','PED_raw_current',
            'n_sources_2022','n_sources_current',
            'lgkp_std_2022','lgkp_std_current',
            'abs_error_2022','abs_error_current']
    required(df,core,'Official temporal_comparison.csv')
    if expected and len(df)!=expected:
        raise ValueError(f'Expected {expected} temporal OOF records; found {len(df)}')
    if df.sample_id.isna().any() or df.sample_id.astype(str).duplicated().any():
        raise ValueError('Missing/duplicate sample_id in temporal OOF')
    if df.source_key.isna().any() or df.groupby('source_key').fold.nunique().max()!=1:
        raise ValueError('source_key missing or appears in multiple outer folds')
    for era in ('2022','current'):
        med = numeric(df,f'MED_{era}')
        m_thr = numeric(df,f'MED_threshold_P50_{era}')
        ped = numeric(df,f'PED_for_threshold_{era}')
        p_thr = numeric(df,'PED_threshold_P50_common')
        m_margin = numeric(df,f'MED_minus_threshold_P50_{era}')
        p_margin = numeric(df,f'PED_minus_threshold_P50_{era}')
        if not np.allclose(m_margin,med-m_thr,rtol=0,atol=3e-9):
            raise ValueError(f'{era} MED margin does not match official model-specific threshold')
        if not np.allclose(p_margin,ped-p_thr,rtol=0,atol=3e-9):
            raise ValueError(f'{era} PED margin does not match official common threshold')
        # Because evidence states use <= on each axis, the zero lines exactly partition Q1-Q4.
        qcalc = np.select([ (m_margin<=0)&(p_margin<=0),
                            (m_margin<=0)&(p_margin>0),
                            (m_margin>0)&(p_margin<=0) ], ['Q1','Q2','Q3'], default='Q4')
        if not np.array_equal(qcalc,df[f'quadrant_{era}'].astype(str).to_numpy()):
            raise ValueError(f'{era} quadrant labels disagree with official P50 margins')
        if not set(df[f'quadrant_{era}'].astype(str).unique()).issubset(Q_STATES):
            raise ValueError(f'{era} has unexpected Q labels')
    if np.any(numeric(df,'PED_raw_current')>numeric(df,'PED_raw_2022')+1e-9):
        raise ValueError('Current PED unexpectedly exceeds historical PED')


def load_cases(root:Path, ids:list[str], expected:int):
    path=root/'models/nested_temporal_evidence_2022_vs_current/temporal_comparison.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing formal temporal comparison file: {path}')
    frame=pd.read_csv(path,low_memory=False)
    validate_temporal(frame,expected)
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate requested sample IDs')
    selected=frame.assign(sample_id=frame.sample_id.astype(str)).set_index('sample_id')
    absent=[s for s in ids if s not in selected.index]
    if absent:
        raise KeyError(f'Cannot find query sample IDs in formal temporal data: {absent}')
    selected=selected.loc[ids].reset_index()
    # This is a state transition, NOT a raw MED difference.
    selected['transition']=selected.quadrant_2022.astype(str)+'->'+selected.quadrant_current.astype(str)
    if 'n_matching_transition_P40_P60' not in selected.columns:
        cols=[f'quadrant_P{p}_{era}' for p in [40,45,50,55,60] for era in ('2022','current')]
        if set(cols).issubset(frame.columns):
            count=[]
            for r in selected.itertuples(index=False):
                count.append(sum(getattr(r,f'quadrant_P{p}_2022')+'->'+getattr(r,f'quadrant_P{p}_current') == r.transition for p in [40,45,50,55,60]))
            selected['n_matching_transition_P40_P60']=count
    return selected,path


def load_table_s10(root:Path, selected:pd.DataFrame):
    audit=root/'models/nested_temporal_transition_audit'
    record_path=audit/'traced_topk_neighbor_records.csv'
    if not record_path.is_file():
        warnings.warn(f'Table S10 neighbor trace unavailable: {record_path}\n'
                      'Run: !python summarize_temporal_transitions.py  (do not use --skip-neighbor-trace).')
        return None,None
    raw=pd.read_csv(record_path,low_memory=False)
    needed=['query_sample_id','fold','reference_period','neighbor_rank','neighbor_sample_id',
            'neighbor_source_key','neighbor_Year','neighbor_lgkp','physical_distance']
    required(raw,needed,'Audit traced_topk_neighbor_records.csv')
    raw['query_sample_id']=raw.query_sample_id.astype(str)
    raw['neighbor_sample_id']=raw.neighbor_sample_id.astype(str)
    ids=selected.sample_id.astype(str).tolist()
    raw=raw.loc[raw.query_sample_id.isin(ids)].copy()
    if raw.empty:
        warnings.warn('Audit neighbor records contain none of the selected samples.\n'
                      'Re-run: !python summarize_temporal_transitions.py')
        return None,None

    config=Path(__file__).resolve().parents[2]/'configs/deployment/full_data_xgboost_parameters.json'
    features=json.loads(config.read_text(encoding='utf-8'))['feature_cols']
    out_frames=[]
    for r in selected.itertuples(index=False):
        sid=str(r.sample_id); fold=int(r.fold)
        subset=raw[raw.query_sample_id.eq(sid)].copy()
        if len(subset)!=20:
            raise ValueError(f'{sid}: found {len(subset)} traced neighbors; expected 20 (10 historical + 10 current). '
                             'Re-run summarize_temporal_transitions.py with neighbor trace enabled.')
        if not subset.fold.astype(int).eq(fold).all():
            raise ValueError(f'{sid}: neighbor trace outer fold does not match formal temporal data')
        if set(subset.reference_period.astype(str))!={'historical_2022','current'}:
            raise ValueError(f'{sid}: neighbor reference_period values do not match expected eras')
        for period in ['historical_2022','current']:
            q=subset[subset.reference_period.eq(period)]
            if (len(q)!=10 or sorted(q.neighbor_rank.astype(int).tolist())!=list(range(1,11))):
                raise ValueError(f'{sid} {period}: neighbor rank 1..10 incomplete/duplicated')
        path=root/'splits/source_kfold'/f'fold{fold}_train.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing training metadata for trace: {path}')
        tr=pd.read_csv(path,low_memory=False)
        metadata=['sample_id','source_key','Year','lgkp',*features]
        required(tr,metadata,f'fold{fold} training data')
        if tr.sample_id.isna().any() or tr.sample_id.astype(str).duplicated().any():
            raise ValueError(f'fold{fold} train has missing/duplicate sample_id')
        tr=tr[metadata].copy()
        tr['sample_id']=tr.sample_id.astype(str)
        tr=tr.rename(columns={'sample_id':'neighbor_sample_id',
                              'source_key':'source_from_train','Year':'year_from_train',
                              'lgkp':'lgkp_from_train','Temperature':'Temperature',
                              'x_Cr':'x_Cr','x_Al':'x_Al'})
        merged=subset.merge(tr,on='neighbor_sample_id',how='left',validate='many_to_one',indicator=True)
        if not merged._merge.eq('both').all():
            bad=merged.loc[~merged._merge.eq('both'),'neighbor_sample_id'].tolist()
            raise ValueError(f'Cannot match neighbor sample IDs in fold{fold} training file: {bad}')
        if not merged.neighbor_source_key.astype(str).eq(merged.source_from_train.astype(str)).all():
            raise ValueError(f'{sid}: source identities in audit vs fold training disagree')
        for a,b in [('neighbor_Year','year_from_train'),('neighbor_lgkp','lgkp_from_train')]:
            if not np.allclose(pd.to_numeric(merged[a]),pd.to_numeric(merged[b]),rtol=0,atol=2e-7):
                raise ValueError(f'{sid}: audit and fold train disagree on {a}')
        # Verify the exact Top-10 neighborhood statistics used in the manuscript.
        for period,era in [('historical_2022','2022'),('current','current')]:
            part=merged.loc[merged.reference_period.eq(period)].copy()
            source_count=part.neighbor_source_key.nunique()
            dispersion=float(part.neighbor_lgkp.astype(float).std(ddof=0))
            mean_ped=float(part.physical_distance.astype(float).mean())
            if source_count != int(getattr(r,f'n_sources_{era}')):
                raise ValueError(f'{sid} {period}: source counts {source_count} != official {getattr(r,f"n_sources_{era}")}')
            if not np.isclose(dispersion,float(getattr(r,f'lgkp_std_{era}')),atol=2e-7,rtol=0):
                raise ValueError(f'{sid} {period}: lgkp_std does not reproduce official value')
            if not np.isclose(mean_ped,float(getattr(r,f'PED_raw_{era}')),atol=2e-7,rtol=0):
                raise ValueError(f'{sid} {period}: averaged physical distance does not reproduce official PED')
            if period=='historical_2022' and (pd.to_numeric(part.neighbor_Year)>2022).any():
                raise ValueError(f'{sid}: historical references include a post-2022 record')
        out_frames.append(merged)

    merged_all=pd.concat(out_frames,ignore_index=True)
    merged_all['case']=merged_all.query_sample_id.map({sid:f'Case {chr(65+i)}' for i,sid in enumerate(ids)})
    merged_all['period']=merged_all.reference_period.map({'historical_2022':'Historical (<=2022)','current':'Current'})
    table=pd.DataFrame({
        'Case':merged_all['case'],
        'Query sample':merged_all.query_sample_id,
        'Period':merged_all.period,
        'Rank':merged_all.neighbor_rank.astype(int),
        'Neighbor sample':merged_all.neighbor_sample_id,
        'Literature source':merged_all.neighbor_source_key,
        'Ref_source_num':merged_all['neighbor_Ref_source_num'] if 'neighbor_Ref_source_num' in merged_all else '',
        'Year':merged_all.neighbor_Year,
        'lgkp':merged_all.neighbor_lgkp,
        'Temperature (C)':merged_all.Temperature,
        'Cr (at.%)':merged_all.x_Cr,
        'Al (at.%)':merged_all.x_Al,
        'Physical distance':merged_all.physical_distance,
    })
    table=table.join(merged_all[features])
    table['_qord']=pd.Categorical(table['Query sample'],categories=ids,ordered=True)
    table['_pord']=pd.Categorical(table['Period'],categories=['Historical (<=2022)','Current'],ordered=True)
    table=table.sort_values(['_qord','_pord','Rank']).drop(columns=['_qord','_pord']).reset_index(drop=True)
    stats=[]
    for row in selected.itertuples(index=False):
        for period,era in [('Historical (<=2022)','2022'),('Current','current')]:
            part=table[(table['Query sample']==row.sample_id)&(table['Period']==period)]
            stats.append({
                'Case':f'Case {chr(65+ids.index(str(row.sample_id)))}',
                'Query sample':str(row.sample_id),
                'Period':period,
                'N_neighbor':len(part),
                'n_sources':part['Literature source'].nunique(),
                'lgkp_std (ddof=0)':float(part.lgkp.std(ddof=0)),
                'mean physical distance (PED)':float(part['Physical distance'].mean()),
                'post-2022 neighbors':int((pd.to_numeric(part.Year)>2022).sum()),
            })
    return table,pd.DataFrame(stats)


def figure_limits(vals: np.ndarray, lower_floor: float=0.12, pad_ratio: float=0.15):
    m=float(np.nanmax(np.abs(vals))) if vals.size else 0.0
    m=max(m*(1+pad_ratio),lower_floor)
    return -m,m


def plot_cases(cases:pd.DataFrame,outpath:Path):
    n=len(cases)
    xs=np.concatenate((cases.PED_minus_threshold_P50_2022.to_numpy(float),
                       cases.PED_minus_threshold_P50_current.to_numpy(float)))
    ys=np.concatenate((cases.MED_minus_threshold_P50_2022.to_numpy(float),
                       cases.MED_minus_threshold_P50_current.to_numpy(float)))
    xlim=figure_limits(xs,lower_floor=0.08)
    ylim=figure_limits(ys,lower_floor=0.08)
    # All panels use the SAME ranges: their quadrant positions can be compared.
    fig,axes=plt.subplots(1,n,figsize=(5.3*n,5.05),squeeze=False)
    axes=axes.ravel()
    for idx,(ax,row) in enumerate(zip(axes,cases.itertuples(index=False))):
        ax.set_facecolor('white')
        ax.axhspan(ylim[0],0,xmin=0,xmax=0.5,color='#DCEFE0',alpha=.46,zorder=0)   # Q1
        ax.axhspan(ylim[0],0,xmin=0.5,xmax=1,color='#E4EBF6',alpha=.46,zorder=0)   # Q2
        ax.axhspan(0,ylim[1],xmin=0,xmax=0.5,color='#F9E8CC',alpha=.46,zorder=0)  # Q3
        ax.axhspan(0,ylim[1],xmin=0.5,xmax=1,color='#F3DCDC',alpha=.46,zorder=0)  # Q4
        ax.axvline(0,color='0.45',linestyle='--',lw=1.15,zorder=1)
        ax.axhline(0,color='0.45',linestyle='--',lw=1.15,zorder=1)
        hx=float(row.PED_minus_threshold_P50_2022); hy=float(row.MED_minus_threshold_P50_2022)
        cx=float(row.PED_minus_threshold_P50_current); cy=float(row.MED_minus_threshold_P50_current)
        ax.annotate('',xy=(cx,cy),xytext=(hx,hy),
                    arrowprops=dict(arrowstyle='-|>',lw=1.8,color='#454545',mutation_scale=16),zorder=4)
        ax.scatter(hx,hy,s=120,marker='o',facecolor='white',edgecolor=POINT_COLORS['historical_2022'],lw=2.0,zorder=5)
        ax.scatter(cx,cy,s=132,marker='s',facecolor=POINT_COLORS['current'],edgecolor='white',lw=0.85,zorder=6)
        # Short position labels with chosen offsets to minimize overlap.
        ax.annotate('2022',(hx,hy),xytext=(7,8),textcoords='offset points',fontsize=9.5,weight='bold',color=POINT_COLORS['historical_2022'])
        ax.annotate('Current',(cx,cy),xytext=(7,-14),textcoords='offset points',fontsize=9.5,weight='bold',color=POINT_COLORS['current'])
        for q,x,y,ha,va in [('Q3',.025,.975,'left','top'),('Q4',.975,.975,'right','top'),
                            ('Q1',.025,.035,'left','bottom'),('Q2',.975,.035,'right','bottom')]:
            ax.text(x,y,q,transform=ax.transAxes,ha=ha,va=va,size=12,weight='bold',color='0.3')
        ax.set_xlim(*xlim);ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(direction='in',labelsize=10.5)
        ax.set_xlabel(r'$\ln(1+PED)-\mathrm{P50}_{PED}$',fontsize=12,weight='bold')
        if idx==0:
            ax.set_ylabel(r'$MED-\mathrm{P50}_{MED}$',fontsize=12,weight='bold')
        ax.set_title(f'{chr(97+idx)})  Case {chr(65+idx)}: {row.quadrant_2022} $\\rightarrow$ {row.quadrant_current}\n{row.sample_id}',fontsize=12,pad=10,weight='bold')
        for spine in ax.spines.values():spine.set_linewidth(1.15)
    legend=[Line2D([],[],marker='o',ls='None',markerfacecolor='white',markeredgecolor=POINT_COLORS['historical_2022'],markeredgewidth=2,ms=9,label='Historical (<=2022)'),
            Line2D([],[],marker='s',ls='None',markerfacecolor=POINT_COLORS['current'],markeredgecolor='white',ms=9,label='Current')]
    fig.legend(handles=legend,loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.01),frameon=False,fontsize=10.5)
    fig.subplots_adjust(wspace=.19,top=.82,bottom=.17,left=.075,right=.975)
    for ext in ['png','pdf','svg']:
        p=outpath.with_suffix('.'+ext)
        if ext=='png':fig.savefig(p,dpi=600,bbox_inches='tight')
        else:fig.savefig(p,bbox_inches='tight')
        print('Saved',p)
    plt.close(fig)


def optional_excel(outdir:Path,frames:dict[str,pd.DataFrame]):
    xlsx=outdir/'fig5d_case_outputs.xlsx'
    try:
        import openpyxl  # optional for user-side tabular export
        engine='openpyxl'
    except ImportError:
        print('Excel skipped: install openpyxl if an .xlsx copy is desired (CSVs already saved).')
        return
    with pd.ExcelWriter(xlsx,engine=engine) as writer:
        for sheet,df in frames.items():
            df.to_excel(writer,sheet_name=sheet,index=False)
            ws=writer.sheets[sheet]
            ws.freeze_panes='A2'
            ws.auto_filter.ref=ws.dimensions
            for cell in ws[1]:
                cell.font=openpyxl.styles.Font(bold=True,color='FFFFFF')
                cell.fill=openpyxl.styles.PatternFill(fill_type='solid',fgColor='274768')
            for j,col in enumerate(df.columns,1):
                from openpyxl.utils import get_column_letter
                ws.column_dimensions[get_column_letter(j)].width=min(33,max(11,len(str(col))+2))
    print('Saved',xlsx)


def main():
    args=args_parse()
    root=Path(args.project_root).expanduser().resolve() if args.project_root else Path.cwd()
    cases,origin=load_cases(root,[str(x) for x in args.sample_ids],args.expected_records)
    outdir=Path(args.output_dir).expanduser().resolve() if args.output_dir else root/'figures/Fig5d_nested_final'
    outdir.mkdir(parents=True,exist_ok=True)
    print('Temporal input:',origin)
    print('Figure coordinate columns: PED_minus_threshold_P50_{era} / MED_minus_threshold_P50_{era}')
    print('Note: MED coordinates use separate model-specific thresholds in 2022 and current.')
    print('Cases:',cases[['sample_id','fold','quadrant_2022','quadrant_current']].to_string(index=False))
    # Numeric case summary from canonical temporal comparison; no raw MED difference reported.
    wanted=['sample_id','source_key','fold','transition','n_matching_transition_P40_P60',
            'quadrant_2022','quadrant_current','MED_2022','MED_current',
            'MED_threshold_P50_2022','MED_threshold_P50_current',
            'MED_minus_threshold_P50_2022','MED_minus_threshold_P50_current',
            'PED_log1p_2022','PED_log1p_current','PED_threshold_P50_common',
            'PED_minus_threshold_P50_2022','PED_minus_threshold_P50_current',
            'PED_raw_2022','PED_raw_current','n_sources_2022','n_sources_current',
            'lgkp_std_2022','lgkp_std_current','y_true','y_pred_2022','y_pred_current',
            'abs_error_2022','abs_error_current','PED_improvement_historical_minus_current']
    case_summary=cases[[x for x in wanted if x in cases.columns]].copy()
    case_path=outdir/'fig5d_case_summary.csv'
    case_summary.to_csv(case_path,index=False,encoding='utf-8-sig')
    plot_cases(cases,outdir/'Fig5d_representative_cases')
    neighbors,neighbor_stats=load_table_s10(root,cases)
    tables={'Case_summary':case_summary}
    if neighbors is not None:
        p=outdir/'Table_S10.csv'
        neighbors.to_csv(p,index=False,encoding='utf-8-sig')
        print('Saved',p,f'({len(neighbors)} rows, 20 per selected sample)')
        stats_path=outdir/'Table_S10_neighborhood_summary.csv'
        neighbor_stats.to_csv(stats_path,index=False,encoding='utf-8-sig')
        tables['Table_S10_neighbors']=neighbors
        tables['Neighborhood_summary']=neighbor_stats
        print('Saved',stats_path)
        print('\nVerified neighborhood statistics:')
        print(neighbor_stats.to_string(index=False))
    print('\nCase summary:')
    print(case_summary.to_string(index=False))
    print('Saved',case_path)
    if not args.no_excel:
        optional_excel(outdir,tables)
    print('DONE. Output directory:',outdir)


if __name__=='__main__':
    main()
