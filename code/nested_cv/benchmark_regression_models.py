# -*- coding: utf-8 -*-
"""Nested 5x4 random/source-aware CV for classical regression benchmarks.

Matches the previously trained nested XGBoost framework:
  - EXACT saved outer splits: splits/{random,source}_kfold/fold{i}_{train,test}.csv
  - source: GroupKFold(4) INSIDE each outer train; random: shuffled KFold(4)
  - Optuna TPE, 100 trials PER (outer fold, tunable model), objective mean inner R2
  - untouched outer held-out testing, pooled OOF metrics and fold-level metrics
  - no re-use of any held-out outer test responses for hyperparameter selection

Based on the six-model original benchmark scripts. Table S3 originally includes
LR/KNN/SVR/RF/MLP plus existing XGBoost. GBR is OPTIONAL (--include-gbr).
This file does NOT train XGBoost, whose nested results are read from existing CSVs.

Recommended:
  python benchmark_regression_models.py --split source
  python benchmark_regression_models.py --split random

Quick validation (use an isolated --out-dir for smoke tests):
  python benchmark_regression_models.py --split source --models LR KNN --trials 2 --inner-folds 2 --out-dir models/nested_benchmark_smoke

Outputs: models/nested_benchmark_6models/{source,random}/<model>/...,
         models/nested_benchmark_6models/Table_S3_nested.csv and .xlsx.
Requirements: numpy pandas scikit-learn optuna joblib openpyxl (xlsx optional).
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings('ignore', category=ConvergenceWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
FOLDS = (1, 2, 3, 4, 5)
FEATURE_COLS = [
    'solu_temp','solu_time','middle_temp','middle_time','aging_temp','aging_time',
    'x_Co','x_Al','x_W','x_Ni','x_Cr','x_Mo','x_Fe','x_Nb','x_C','x_Hf',
    'x_Si','x_Ta','x_Ti','x_Y','x_V','x_B','x_Zr','x_Mn','x_Sc','x_La','x_Re',
    'Temperature','E_Time','has_solu','has_middle','has_aging',
]
TARGET = 'lgkp'
SOURCE = 'source_key'
IDENTITY = 'sample_id'
STANDARD_MODELS = ('LR','KNN','SVR','RF','MLP')
ALL_MODELS = STANDARD_MODELS + ('GBR',)
HIDDEN = {
    '32': (32,), '64': (64,), '128': (128,),
    '64_32': (64, 32), '128_64': (128, 64), '128_64_32': (128, 64, 32),
}


def cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', choices=('source','random','both'), default='source')
    p.add_argument('--data-root', default='splits')
    p.add_argument('--out-dir', default='models/nested_benchmark_6models')
    p.add_argument('--models', nargs='+', choices=ALL_MODELS, default=list(STANDARD_MODELS))
    p.add_argument('--include-gbr', action='store_true', help='Append GBR to the five original Table S3 baselines')
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--inner-folds', type=int, default=4)
    p.add_argument('--rf-jobs', type=int, default=8)
    p.add_argument('--report-every', type=int, default=10)
    p.add_argument('--source-xgb-summary', default='models/xgb_source_nested_kfold/nested_summary.csv')
    p.add_argument('--random-xgb-summary', default='models/xgb_random_nested_kfold/nested_summary.csv')
    p.add_argument('--source-xgb-oof', default='models/xgb_source_nested_kfold/nested_oof_predictions.csv')
    p.add_argument('--random-xgb-oof', default='models/xgb_random_nested_kfold/nested_oof_predictions.csv')
    a = p.parse_args()
    if a.include_gbr and 'GBR' not in a.models:
        a.models.append('GBR')
    a.models = list(dict.fromkeys(a.models))
    if a.trials < 1 or a.inner_folds < 2 or a.rf_jobs < 1 or a.report_every < 1:
        p.error('trials>=1, inner-folds>=2, rf-jobs>=1, report-every>=1 required')
    return a


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def read_fold(data_root, scheme, fold):
    base = data_root / ('source_kfold' if scheme == 'source' else 'random_kfold')
    a = base / f'fold{fold}_train.csv'
    b = base / f'fold{fold}_test.csv'
    if not (a.is_file() and b.is_file()):
        raise FileNotFoundError(f'Fold {fold} missing: {a}, {b}')
    tr = pd.read_csv(a, low_memory=False)
    te = pd.read_csv(b, low_memory=False)
    required = FEATURE_COLS + [TARGET, SOURCE, IDENTITY]
    for label, df in [('train',tr),('test',te)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'{scheme} fold {fold} {label} missing {missing}')
        if df[IDENTITY].isna().any() or df[IDENTITY].astype(str).duplicated().any():
            raise ValueError(f'{scheme} fold {fold} {label}: invalid sample_id')
        if df[SOURCE].isna().any() or df[TARGET].isna().any():
            raise ValueError(f'{scheme} fold {fold} {label}: missing source/target')
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        y = df[TARGET].to_numpy(dtype=float)
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError(f'{scheme} fold {fold} {label}: non-finite values')
    ids_tr = set(tr[IDENTITY].astype(str))
    ids_te = set(te[IDENTITY].astype(str))
    if ids_tr & ids_te:
        raise ValueError(f'{scheme} fold {fold}: outer train/test record overlap')
    s_tr = set(tr[SOURCE].astype(str))
    s_te = set(te[SOURCE].astype(str))
    if scheme == 'source' and s_tr & s_te:
        raise ValueError(f'Source leakage in fold {fold}: {len(s_tr & s_te)} shared sources')
    return tr, te


def hyperparameters(trial, name):
    # Same search ranges as the uploaded original six-model benchmark scripts.
    if name == 'KNN':
        return dict(n_neighbors=trial.suggest_int('n_neighbors',2,40),
                    weights=trial.suggest_categorical('weights',['uniform','distance']),
                    p=trial.suggest_int('p',1,2), leaf_size=trial.suggest_int('leaf_size',10,80))
    if name == 'SVR':
        kernel = trial.suggest_categorical('kernel',['rbf','poly','sigmoid'])
        p = dict(kernel=kernel, C=trial.suggest_float('C',1e-2,1e3,log=True),
                 epsilon=trial.suggest_float('epsilon',1e-3,1.,log=True),
                 gamma=trial.suggest_float('gamma',1e-4,1.,log=True))
        if kernel == 'poly':
            p['degree'] = trial.suggest_int('degree',2,4)
            p['coef0'] = trial.suggest_float('coef0',0.,2.)
        elif kernel == 'sigmoid':
            p['coef0'] = trial.suggest_float('coef0',0.,2.)
        return p
    if name == 'RF':
        return dict(n_estimators=trial.suggest_int('n_estimators',200,1200),
                    max_depth=trial.suggest_int('max_depth',3,40),
                    min_samples_split=trial.suggest_int('min_samples_split',2,20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
                    max_features=trial.suggest_categorical('max_features',['sqrt','log2',0.5,0.8,1.0]),
                    bootstrap=trial.suggest_categorical('bootstrap',[True,False]))
    if name == 'GBR':
        return dict(n_estimators=trial.suggest_int('n_estimators',100,1000),
                    learning_rate=trial.suggest_float('learning_rate',1e-3,0.2,log=True),
                    max_depth=trial.suggest_int('max_depth',1,6),
                    min_samples_split=trial.suggest_int('min_samples_split',2,20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
                    subsample=trial.suggest_float('subsample',0.5,1.0),
                    max_features=trial.suggest_categorical('max_features',[None,'sqrt','log2',0.5,0.8,1.0]),
                    loss=trial.suggest_categorical('loss',['squared_error','huber','absolute_error']))
    if name == 'MLP':
        return dict(hidden_architecture=trial.suggest_categorical('hidden_architecture',list(HIDDEN)),
                    activation=trial.suggest_categorical('activation',['relu','tanh']),
                    alpha=trial.suggest_float('alpha',1e-6,1e-1,log=True),
                    learning_rate_init=trial.suggest_float('learning_rate_init',1e-4,5e-2,log=True),
                    batch_size=trial.suggest_categorical('batch_size',[16,32,64,128]))
    raise ValueError(f'Cannot tune {name}')


def construct(name, params, rf_jobs):
    p = dict(params)
    if name == 'LR':
        return Pipeline([('scaler',StandardScaler()),('model',LinearRegression())])
    if name == 'KNN':
        return Pipeline([('scaler',StandardScaler()),('model',KNeighborsRegressor(**p))])
    if name == 'SVR':
        return Pipeline([('scaler',StandardScaler()),('model',SVR(**p))])
    if name == 'RF':
        return RandomForestRegressor(random_state=SEED,n_jobs=rf_jobs,**p)
    if name == 'GBR':
        return GradientBoostingRegressor(random_state=SEED,**p)
    if name == 'MLP':
        hidden_name = p.pop('hidden_architecture', None)
        if hidden_name is not None:
            p['hidden_layer_sizes'] = HIDDEN[hidden_name]
        elif 'hidden_layer_sizes' in p:
            p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
        return Pipeline([('scaler',StandardScaler()),('model',MLPRegressor(
            solver='adam', random_state=SEED, max_iter=2500,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=50, **p))])
    raise ValueError(f'Unknown model {name}')


def scores(yt, yp):
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    return dict(R2=float(r2_score(yt, yp)),
                MAE=float(mean_absolute_error(yt, yp)),
                RMSE=float(math.sqrt(mean_squared_error(yt, yp))))


def inner_indices(scheme, tr, fold, inner_folds):
    X = tr[FEATURE_COLS].to_numpy(dtype=float)
    if scheme == 'source':
        groups = tr[SOURCE].astype(str).to_numpy()
        if len(set(groups)) < inner_folds:
            raise ValueError(f'Fold {fold}: insufficient source groups for inner CV')
        cv = GroupKFold(n_splits=inner_folds)
        ix = list(cv.split(X, groups=groups))
        for t, v in ix:
            if set(groups[t]) & set(groups[v]):
                raise RuntimeError(f'Inner source leakage: fold {fold}')
        return ix
    cv = KFold(n_splits=inner_folds, shuffle=True, random_state=SEED+fold)
    return list(cv.split(X))


def validate_config(root, args):
    root.mkdir(parents=True, exist_ok=True)
    config = dict(version='nested_benchmark_v1',
                  inner_folds=args.inner_folds, rf_jobs=args.rf_jobs,
                  features=FEATURE_COLS, target=TARGET,
                  data_root=str(Path(args.data_root).resolve()),
                  inner_source='GroupKFold',inner_random='KFold shuffle seed=42+outer',
                  search_space='Same as old uploaded 6-model benchmark',
                  tuning='Independent 100-trial Optuna per outer fold; inner mean R2',
                  note='Changing trial count between runs is allowed only by increasing the target.')
    cp = root / 'run_config.json'
    if cp.exists():
        old = json.loads(cp.read_text(encoding='utf-8'))
        for key in ('version','inner_folds','rf_jobs','features','target','data_root','inner_source','inner_random'):
            if old.get(key)!=config[key]:
                raise RuntimeError(f'Output directory has different {key}; use a fresh --out-dir')
    dump_json(config,cp)


def saved_fold_or_none(fold_dir, trials):
    meta_path=fold_dir/'completed_fold.json'
    mpath=fold_dir/'fold_metrics.csv'
    ppath=fold_dir/'fold_predictions.csv'
    if not (meta_path.exists() and mpath.exists() and ppath.exists()):
        return None
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    if int(meta.get('trials',-1)) != trials:
        return None
    return pd.read_csv(mpath),pd.read_csv(ppath),meta


def fit_fold(args, scheme, name, fold, tr, te, root):
    folder=root/scheme/name/f'fold{fold}'
    folder.mkdir(parents=True,exist_ok=True)
    if name == 'LR':
        trials=0
        params={}
        inner_best=None
    else:
        trials=args.trials
        ix=inner_indices(scheme,tr,fold,args.inner_folds)
        X=tr[FEATURE_COLS].to_numpy(dtype=float)
        y=tr[TARGET].to_numpy(dtype=float)
        db=(folder/'optuna.db').resolve().as_posix()
        st=optuna.create_study(direction='maximize',
            study_name=f'{scheme}_{name}_outer_{fold}',
            sampler=optuna.samplers.TPESampler(seed=SEED+fold),
            storage=f'sqlite:///{db}',load_if_exists=True)
        n_complete=sum(t.state==optuna.trial.TrialState.COMPLETE for t in st.trials)
        if n_complete > trials:
            raise RuntimeError(f'{scheme}/{name}/fold{fold}: {n_complete} completed trials > requested {trials}')
        pending=trials-n_complete
        print(f' {scheme} {name} fold{fold}: Optuna complete={n_complete}, new={pending}',flush=True)
        if pending:
            def objective(t):
                p=hyperparameters(t,name)
                inner_r2=[]
                for train_ix,val_ix in ix:
                    mod=construct(name,p,args.rf_jobs)
                    mod.fit(X[train_ix],y[train_ix])
                    pred=np.asarray(mod.predict(X[val_ix]),dtype=float)
                    sc=float(r2_score(y[val_ix],pred))
                    inner_r2.append(sc if np.isfinite(sc) else -1e9)
                for j,sc in enumerate(inner_r2,1):
                    t.set_user_attr(f'inner{j}_R2',sc)
                return float(np.mean(inner_r2))
            def report(study, trial):
                completed=sum(z.state==optuna.trial.TrialState.COMPLETE for z in study.trials)
                if completed==1 or completed%args.report_every==0 or completed>=trials:
                    print(f' {scheme} {name} fold{fold}: {completed}/{trials}, inner best R2={study.best_value:.5f}',flush=True)
            st.optimize(objective,n_trials=pending,callbacks=[report])
        params=dict(st.best_params)
        inner_best=float(st.best_value)
        # Save trial-level record, including inner R2 diagnostics.
        st.trials_dataframe(attrs=('number','value','params','state','user_attrs')).to_csv(
            folder/'optuna_trials.csv',index=False,encoding='utf-8-sig')
    cached=saved_fold_or_none(folder,trials)
    if cached is not None and cached[2].get('params')==params:
        print(f' {scheme} {name} fold{fold}: REUSE saved outer prediction',flush=True)
        return cached[0],cached[1],dict(cached[2])
    model=construct(name,params,args.rf_jobs)
    Xtr=tr[FEATURE_COLS].to_numpy(dtype=float)
    ytr=tr[TARGET].to_numpy(dtype=float)
    Xte=te[FEATURE_COLS].to_numpy(dtype=float)
    yte=te[TARGET].to_numpy(dtype=float)
    model.fit(Xtr,ytr)
    pred=np.asarray(model.predict(Xte),dtype=float)
    m=scores(yte,pred)
    src_overlap=len(set(tr[SOURCE].astype(str)) & set(te[SOURCE].astype(str)))
    fold_m=pd.DataFrame([dict(split=scheme,model=name,fold=fold,
        n_train=len(tr),n_test=len(te),n_source_overlap=src_overlap,
        inner_best_R2=inner_best,**m)])
    selected=['sample_id','source_key','Ref_source_num','Year','Temperature','E_Time']
    used=[k for k in selected if k in te.columns]
    fold_p=te[used].copy().reset_index(drop=True)
    fold_p.insert(0,'split',scheme)
    fold_p.insert(1,'model',name)
    fold_p.insert(2,'fold',fold)
    fold_p['y_true']=yte
    fold_p['y_pred']=pred
    fold_p['abs_error']=np.abs(pred-yte)
    fold_m.to_csv(folder/'fold_metrics.csv',index=False,encoding='utf-8-sig')
    fold_p.to_csv(folder/'fold_predictions.csv',index=False,encoding='utf-8-sig')
    meta=dict(split=scheme,model=name,fold=fold,trials=trials,
              best_inner_mean_R2=inner_best,params=params,
              n_train=len(tr),n_test=len(te),metrics=m)
    dump_json(meta,folder/'completed_fold.json')
    print(f' {scheme} {name} fold{fold}: outer R2={m["R2"]:.5f} MAE={m["MAE"]:.5f} RMSE={m["RMSE"]:.5f}',flush=True)
    return fold_m,fold_p,meta


def validate_oof(pred, universe):
    if pred['sample_id'].astype(str).duplicated().any():
        raise RuntimeError('Duplicate OOF sample_id')
    if set(pred['sample_id'].astype(str)) != universe:
        raise RuntimeError('OOF sample universe does not match saved outer splits')


def evaluate_split(args,scheme,root):
    data_root=Path(args.data_root).expanduser().resolve()
    folds={}
    universe=None
    all_test=[]
    for fold in FOLDS:
        tr,te=read_fold(data_root,scheme,fold)
        ids=set(tr[IDENTITY].astype(str)) | set(te[IDENTITY].astype(str))
        if universe is None: universe=ids
        elif universe!=ids: raise RuntimeError('Different sample universes across outer folds')
        all_test.extend(te[IDENTITY].astype(str).tolist())
        folds[fold]=(tr,te)
    if len(all_test)!=len(set(all_test)) or set(all_test)!=universe:
        raise RuntimeError('Outer test sets are not a unique complete OOF partition')
    print(f'\n{scheme.upper()} nested benchmark: {len(universe)} OOF records, models={args.models}',flush=True)
    for model_name in args.models:
        fold_m_list=[]; fold_p_list=[]; param_rows=[]
        for fold,(tr,te) in folds.items():
            m,p,meta=fit_fold(args,scheme,model_name,fold,tr,te,root)
            fold_m_list.append(m)
            fold_p_list.append(p)
            param_rows.append(dict(split=scheme,model=model_name,fold=fold,
                                   best_inner_mean_R2=meta['best_inner_mean_R2'],
                                   **meta['params']))
        model_dir=root/scheme/model_name
        fm=pd.concat(fold_m_list,ignore_index=True)
        pp=pd.concat(fold_p_list,ignore_index=True)
        validate_oof(pp,universe)
        sm=scores(pp['y_true'],pp['y_pred'])
        summary=dict(split=scheme,model=model_name,n_oof=len(pp),
                     pooled_R2=sm['R2'],pooled_MAE=sm['MAE'],pooled_RMSE=sm['RMSE'],
                     mean_fold_R2=float(fm['R2'].mean()),sd_fold_R2=float(fm['R2'].std(ddof=1)),
                     mean_fold_MAE=float(fm['MAE'].mean()),sd_fold_MAE=float(fm['MAE'].std(ddof=1)),
                     mean_fold_RMSE=float(fm['RMSE'].mean()),sd_fold_RMSE=float(fm['RMSE'].std(ddof=1)))
        fm.to_csv(model_dir/'fold_metrics_all.csv',index=False,encoding='utf-8-sig')
        pp.to_csv(model_dir/'pooled_predictions.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(param_rows).to_csv(model_dir/'best_params_by_outer_fold.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame([summary]).to_csv(model_dir/'pooled_metrics.csv',index=False,encoding='utf-8-sig')
        print(f'  FINAL {scheme}/{model_name}: pooled R2={sm["R2"]:.5f} MAE={sm["MAE"]:.5f}',flush=True)


def get_xgb(args,scheme):
    path=Path(args.source_xgb_summary if scheme=='source' else args.random_xgb_summary)
    p_oof=Path(args.source_xgb_oof if scheme=='source' else args.random_xgb_oof)
    if not path.is_file() or not p_oof.is_file():
        return None
    # Recompute from frozen, independent XGBoost nested OOF to avoid row selection mistakes.
    data=pd.read_csv(p_oof)
    if not {'sample_id','y_true','y_pred_nested'} <= set(data.columns):
        raise ValueError(f'Unexpected XGB nested OOF columns in {p_oof}')
    ids=data['sample_id'].astype(str)
    if ids.duplicated().any():
        raise ValueError(f'Duplicate XGB nested sample ID in {p_oof}')
    benchmark_split=Path(args.data_root)/('source_kfold' if scheme=='source' else 'random_kfold')
    expected=[]
    for fold in FOLDS:
        d=pd.read_csv(benchmark_split/f'fold{fold}_test.csv',usecols=['sample_id',TARGET])
        expected.append(d.rename(columns={TARGET:'expected_y_true'}))
    ey=pd.concat(expected,ignore_index=True)
    merged=data.merge(ey,on='sample_id',how='outer',validate='one_to_one',indicator=True)
    if not (merged['_merge']=='both').all() or not np.allclose(
        merged['y_true'].to_numpy(dtype=float),merged['expected_y_true'].to_numpy(dtype=float),atol=1e-10):
        raise ValueError(f'Nested XGB OOF is not from the same {scheme} test split')
    m=scores(data['y_true'],data['y_pred_nested'])
    summary=pd.read_csv(path)
    if 'method' in summary:
        wanted=summary[summary['method'].astype(str).str.contains('Nested',case=False,na=False)]
        if len(wanted):
            recorded=float(wanted.iloc[0]['pooled_OOF_R2'])
            if abs(recorded-m['R2'])>2e-5:
                raise ValueError(f'XGB {scheme} OOF summary does not reproduce its predictions')
    return dict(split=scheme,model='XGBoost',n_oof=len(data),
                pooled_R2=m['R2'],pooled_MAE=m['MAE'],pooled_RMSE=m['RMSE'])



def get_s2_nested_table():
    """Reformat existing 5-fold nested XGBoost parameters for Supplementary Table S2."""
    p = Path('models/xgb_source_nested_kfold/nested_best_params.csv')
    if not p.is_file():
        return None
    raw = pd.read_csv(p)
    if 'fold' not in raw.columns or set(pd.to_numeric(raw['fold'], errors='coerce')) != set(FOLDS):
        raise ValueError(f'Expected five distinct XGBoost outer folds in {p}')
    params = [
        ('max_depth', '4-50', 'Integer'),
        ('learning_rate', '0.0001-0.01', 'Log-uniform'),
        ('n_estimators', '100-1000', 'Integer'),
        ('min_child_weight', '2.0-10.0', 'Continuous'),
        ('subsample', '0.1-1.0', 'Continuous'),
        ('colsample_bytree', '0.1-1.0', 'Continuous'),
        ('gamma', '0.1-1.0', 'Continuous'),
        ('reg_alpha', '0.1-1.0', 'Continuous'),
        ('reg_lambda', '0.1-1.0', 'Continuous'),
    ]
    for key,_,_ in params:
        if key not in raw.columns:
            raise ValueError(f'Nested XGBoost best parameters missing {key} in {p}')
    rows=[]
    for k,range_,kind in params:
        row={'Hyperparameter':k,'Search range':range_,'Sampling scale':kind}
        for fold in FOLDS:
            row[f'Fold {fold}']=float(raw.loc[raw.fold==fold,k].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_all(args,root):
    records=[]; folds=[]; params=[]
    for scheme in ('random','source'):
        for model in ALL_MODELS:
            d=root/scheme/model
            p=d/'pooled_metrics.csv'
            if p.is_file():
                records.extend(pd.read_csv(p).to_dict('records'))
                for filename,collection in [('fold_metrics_all.csv',folds),
                                            ('best_params_by_outer_fold.csv',params)]:
                    pp=d/filename
                    if pp.is_file():collection.append(pd.read_csv(pp))
        x=get_xgb(args,scheme)
        if x is not None: records.append(x)
    if not records:
        return
    pooled=pd.DataFrame(records)
    s2=get_s2_nested_table()
    if s2 is not None:
        s2.to_csv(root/'Table_S2_nested.csv',index=False,encoding='utf-8-sig')
    pooled.to_csv(root/'all_nested_pooled_metrics.csv',index=False,encoding='utf-8-sig')
    if folds: pd.concat(folds,ignore_index=True).to_csv(root/'all_nested_fold_metrics.csv',index=False,encoding='utf-8-sig')
    if params: pd.concat(params,ignore_index=True).to_csv(root/'all_nested_best_params.csv',index=False,encoding='utf-8-sig')
    # Original S3 ordering LR/KNN/SVR/RF/MLP/XGBoost. Append GBR only if it was requested.
    model_order=list(STANDARD_MODELS)+(['GBR'] if 'GBR' in set(pooled.model) else [])+['XGBoost']
    rows=[]
    for name in model_order:
        subset=pooled[pooled['model']==name]
        if subset.empty:continue
        row={'Model':name}
        for scheme,label in [('random','Random'),('source','Source')]:
            ss=subset[subset['split']==scheme]
            row[f'{label}_R2']=float(ss.iloc[0]['pooled_R2']) if len(ss) else np.nan
            row[f'{label}_MAE']=float(ss.iloc[0]['pooled_MAE']) if len(ss) else np.nan
            row[f'{label}_RMSE']=float(ss.iloc[0]['pooled_RMSE']) if len(ss) else np.nan
        row['Delta_R2_Source_minus_Random']=(row['Source_R2']-row['Random_R2'])
        rows.append(row)
    s3=pd.DataFrame(rows)
    s3.to_csv(root/'Table_S3_nested.csv',index=False,encoding='utf-8-sig')
    try:
        from openpyxl.styles import Alignment,Font,PatternFill
        from openpyxl.utils import get_column_letter
        sheets=[('Table_S3',s3),('All_Pooled_Metrics',pooled)]
        if s2 is not None: sheets.insert(0,('Table_S2_XGBoost',s2))
        if folds:sheets.append(('Fold_Metrics',pd.concat(folds,ignore_index=True)))
        if params:sheets.append(('Best_Params',pd.concat(params,ignore_index=True)))
        with pd.ExcelWriter(root/'nested_benchmark_results.xlsx',engine='openpyxl') as writer:
            for sheet,data in sheets:
                data.to_excel(writer,sheet_name=sheet,index=False)
                ws=writer.sheets[sheet]
                ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
                for c in ws[1]:
                    c.font=Font(bold=True,color='FFFFFF')
                    c.fill=PatternFill(fill_type='solid',fgColor='17365D')
                    c.alignment=Alignment(vertical='center')
                for j,c in enumerate(data.columns,1):
                    ws.column_dimensions[get_column_letter(j)].width=min(36,max(12,len(str(c))+3))
    except ImportError:
        print('openpyxl unavailable: CSV output is complete.',flush=True)
    print('\nTABLE S3 (partial until BOTH splits finish):',flush=True)
    print(s3.to_string(index=False,float_format=lambda x:f'{x:.5f}'),flush=True)
    print(f'\nCombined outputs: {root}',flush=True)


def main():
    args=cli()
    root=Path(args.out_dir).expanduser().resolve()
    validate_config(root,args)
    print('='*90,flush=True)
    print('NESTED 6-MODEL BENCHMARK (XGBOOST PRECOMPUTED)',flush=True)
    print(f'split={args.split}; models={args.models}; inner={args.inner_folds}, trials/model/outer={args.trials}',flush=True)
    print(f'output={root}',flush=True)
    print('All Optuna selection is inside each outer TRAIN set. No outer test tuning.',flush=True)
    print('='*90,flush=True)
    schemes=['source','random'] if args.split=='both' else [args.split]
    for s in schemes:
        evaluate_split(args,s,root)
        aggregate_all(args,root)
    print('\nALL REQUESTED BENCHMARKS COMPLETE',flush=True)

if __name__=='__main__':
    main()
