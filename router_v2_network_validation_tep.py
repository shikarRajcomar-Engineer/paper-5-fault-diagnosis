#!/usr/bin/env python3
"""
Router V2 Experiment 1 - corrected TEP normal directional network validation.

Uses ONLY TEP_FaultFree_Training.parquet.

Correction versus the previous version
--------------------------------------
The observed edge is selected after scanning lags 1..max_lag. Every circular-
shift surrogate now receives exactly the same best-of-lags search. This removes
an optimisation advantage previously given only to the observed data.

Interpretation: directional predictive network, NOT a proven causal graph.
"""
from __future__ import annotations
import argparse, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

EPS = 1e-12
SENSORS = [f"xmeas_{i}" for i in range(1,42)] + [f"xmv_{i}" for i in range(1,12)]
RUN_NAMES = ["simulationRun","simulation_run","simRun","run","run_id","Run","replicate"]
TIME_NAMES = ["sample","sample_id","time","Time","timestamp","sampleNumber","sample_number"]
META = {"faultNumber","fault_number","fault","Fault","Class","class","label","true_cause","amplitude","fault_type"}

@dataclass(frozen=True)
class Config:
    max_lag:int=10
    n_splits:int=5
    test_fraction:float=0.25
    min_improvement:float=0.01
    min_stability:float=0.60
    min_directionality:float=0.10
    surrogates:int=200
    fdr_alpha:float=0.05
    random_seed:int=42
    ridge:float=1e-6
    max_runs:Optional[int]=None

def find_col(cols, candidates):
    lower={str(c).lower():c for c in cols}
    for x in candidates:
        if x in cols:return x
        if x.lower() in lower:return lower[x.lower()]
    return None

def sensor_key(name):
    s=''.join(ch for ch in str(name).lower() if ch.isalnum())
    if s.startswith('xmeas'):
        d=''.join(ch for ch in s[5:] if ch.isdigit()); return (0,int(d or 999))
    if s.startswith('xmv'):
        d=''.join(ch for ch in s[3:] if ch.isdigit()); return (1,int(d or 999))
    return (2,999)

def canonicalise(df, run_col, time_col):
    named=[c for c in df.columns if ''.join(ch for ch in str(c).lower() if ch.isalnum()).startswith(('xmeas','xmv'))]
    if len(named)==52:
        named=sorted(named,key=sensor_key)
        return df.rename(columns=dict(zip(named,SENSORS))), SENSORS.copy()
    excluded=set(META)|({run_col} if run_col else set())|({time_col} if time_col else set())
    numeric=[c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric)==52:
        print('  Mapping 52 numeric sensor columns in file order to xmeas_1..41,xmv_1..11.')
        return df.rename(columns=dict(zip(numeric,SENSORS))), SENSORS.copy()
    raise SystemExit(f"Could not identify exactly 52 TEP sensors. Numeric candidates={len(numeric)}, named={len(named)}")

def load_normal(path:Path,max_runs):
    if not path.exists(): raise SystemExit(f"Missing {path}")
    print(f"Loading {path.name} ...",flush=True)
    df=pd.read_parquet(path)
    run_col=find_col(df.columns,RUN_NAMES); time_col=find_col(df.columns,TIME_NAMES)
    df,sensors=canonicalise(df,run_col,time_col)
    vals=df[sensors].replace([np.inf,-np.inf],np.nan).interpolate(limit_direction='both')
    if vals.isna().any().any(): raise SystemExit('Unresolved NaNs in sensor data.')
    df.loc[:,sensors]=vals
    runs=[]
    if run_col:
        items=list(df.groupby(run_col,sort=True))
        if max_runs is not None: items=items[:max_runs]
        for _,b in items:
            if time_col:b=b.sort_values(time_col)
            a=b[sensors].to_numpy(np.float64,copy=True)
            if len(a)>2:runs.append(a)
    else:
        b=df.sort_values(time_col) if time_col else df
        runs=[b[sensors].to_numpy(np.float64,copy=True)]
    meta=dict(file=str(path.resolve()),rows=int(sum(map(len,runs))),n_runs=len(runs),run_column=run_col,time_column=time_col,
              run_lengths_min=int(min(map(len,runs))),run_lengths_median=float(np.median(list(map(len,runs)))),run_lengths_max=int(max(map(len,runs))),sensors=sensors)
    return runs,sensors,meta

def design_run(run,src,tgt,lag):
    yv=run[:,tgt]; xv=run[:,src]; n=len(run)
    if n<=lag+2:return np.empty(0),np.empty((0,lag)),np.empty((0,2*lag))
    y=yv[lag:]
    tb=[yv[lag-k:n-k] for k in range(1,lag+1)]
    sb=[xv[lag-k:n-k] for k in range(1,lag+1)]
    return y,np.column_stack(tb),np.column_stack(tb+sb)

def stack_design(runs,src,tgt,lag):
    ys=[]; bs=[]; es=[]
    for r in runs:
        y,b,e=design_run(r,src,tgt,lag)
        if len(y):ys.append(y);bs.append(b);es.append(e)
    if not ys:return np.empty(0),np.empty((0,lag)),np.empty((0,2*lag))
    return np.concatenate(ys),np.vstack(bs),np.vstack(es)

def fit(X,y,ridge):
    A=np.column_stack([np.ones(len(X)),X]); G=A.T@A; P=np.eye(G.shape[0])*ridge;P[0,0]=0
    return np.linalg.solve(G+P,A.T@y)

def pred(X,b):return np.column_stack([np.ones(len(X)),X])@b

def split_runs(runs,test_fraction):
    if len(runs)>=2:
        cut=max(1,min(len(runs)-1,int(math.floor(len(runs)*(1-test_fraction)))))
        return list(runs[:cut]),list(runs[cut:])
    r=runs[0];cut=int(math.floor(len(r)*(1-test_fraction)));return [r[:cut]],[r[cut:]]

def improvement(train,test,src,tgt,lag,ridge):
    yt,bt,et=stack_design(train,src,tgt,lag);yv,bv,ev=stack_design(test,src,tgt,lag)
    if len(yt)<max(40,5*lag) or len(yv)<10:return np.nan,np.nan,np.nan
    ym=yt.mean();ys=yt.std() or 1.;yt=(yt-ym)/ys;yv=(yv-ym)/ys
    def scale(a,b):
        m=a.mean(0);s=a.std(0);s=np.where(s<EPS,1.,s);return (a-m)/s,(b-m)/s
    bt,bv=scale(bt,bv);et,ev=scale(et,ev)
    bb=fit(bt,yt,ridge);be=fit(et,yt,ridge)
    rb=float(np.sqrt(np.mean((yv-pred(bv,bb))**2)));re=float(np.sqrt(np.mean((yv-pred(ev,be))**2)))
    return float((rb-re)/max(rb,EPS)),rb,re

def best_pair(runs,src,tgt,cfg):
    tr,te=split_runs(runs,cfg.test_fraction);vals=[]
    for lag in range(1,cfg.max_lag+1):
        imp,b,e=improvement(tr,te,src,tgt,lag,cfg.ridge)
        if np.isfinite(imp):vals.append((imp,lag,b,e))
    if not vals:return dict(improvement=np.nan,best_lag=np.nan,baseline_rmse=np.nan,extended_rmse=np.nan)
    imp,lag,b,e=max(vals,key=lambda z:z[0]);return dict(improvement=imp,best_lag=lag,baseline_rmse=b,extended_rmse=e)

def blocks(runs,n):
    if len(runs)>=n:return [[runs[int(i)] for i in x] for x in np.array_split(np.arange(len(runs)),n) if len(x)]
    if len(runs)==1:
        r=runs[0]; q=np.linspace(0,len(r),n+1,dtype=int); return [[r[a:b]] for a,b in zip(q[:-1],q[1:]) if b-a>20]
    return [[r] for r in runs]

def pair_table(runs,sensors,cfg):
    bl=blocks(runs,cfg.n_splits); rows=[]; total=len(sensors)*(len(sensors)-1);done=0
    print(f"Testing {total} ordered pairs x lags 1..{cfg.max_lag} ...",flush=True)
    for si in range(len(sensors)):
      for ti in range(len(sensors)):
        if si==ti:continue
        full=best_pair(runs,si,ti,cfg); split=[];lags=[]
        for b in bl:
            if sum(map(len,b))<max(100,8*cfg.max_lag):continue
            z=best_pair(b,si,ti,cfg);split.append(z['improvement']);lags.append(z['best_lag'])
        fin=np.asarray([x for x in split if np.isfinite(x)],float)
        rows.append(dict(source=sensors[si],target=sensors[ti],source_idx=si,target_idx=ti,best_lag=full['best_lag'],improvement=full['improvement'],
                         baseline_rmse=full['baseline_rmse'],extended_rmse=full['extended_rmse'],split_stability=float(np.mean(fin>=cfg.min_improvement)) if len(fin) else np.nan,
                         median_split_improvement=float(np.median(fin)) if len(fin) else np.nan,median_split_lag=float(np.median([x for x in lags if np.isfinite(x)])) if any(np.isfinite(x) for x in lags) else np.nan,n_valid_splits=len(fin)))
        done+=1
        if done%max(1,total//20)==0 or done==total:print(f"  pair progress: {done}/{total}",flush=True)
    t=pd.DataFrame(rows);rev=t[['source','target','improvement']].rename(columns={'source':'target','target':'source','improvement':'reverse_improvement'})
    t=t.merge(rev,on=['source','target'],how='left');t['direction_margin']=t.improvement-t.reverse_improvement
    d=np.maximum(np.maximum(t.improvement.abs(),t.reverse_improvement.abs()),EPS);t['directionality']=t.direction_margin/d
    return t

def shifted_runs(runs,src,rng,max_lag):
    out=[]
    for r in runs:
        c=r.copy(); n=len(c); m=max(2*max_lag+1,int(.05*n))
        if n>2*m+2:c[:,src]=np.roll(c[:,src],int(rng.integers(m,n-m)))
        out.append(c)
    return out

def surrogate_best_of_lags(runs,row,cfg,rng):
    sh=shifted_runs(runs,int(row.source_idx),rng,cfg.max_lag);tr,te=split_runs(sh,cfg.test_fraction);scores=[]
    for lag in range(1,cfg.max_lag+1):
        imp,_,_=improvement(tr,te,int(row.source_idx),int(row.target_idx),lag,cfg.ridge)
        if np.isfinite(imp):scores.append(imp)
    return max(scores) if scores else np.nan

def bh(p):
    out=pd.Series(np.nan,index=p.index);v=p.dropna()
    if v.empty:return out
    o=v.sort_values();a=o.to_numpy();m=len(a);q=a*m/np.arange(1,m+1);q=np.minimum.accumulate(q[::-1])[::-1];out.loc[o.index]=np.clip(q,0,1);return out

def null_test(table,runs,cfg):
    c=table[(table.improvement>=cfg.min_improvement)&(table.split_stability>=cfg.min_stability)&(table.directionality>=cfg.min_directionality)].copy()
    print(f"Null-testing {len(c)} candidate edges with {cfg.surrogates} circular shifts; every surrogate scans lags 1..{cfg.max_lag} ...",flush=True)
    rng=np.random.default_rng(cfg.random_seed); ps=[];med=[];q95=[];mx=[]
    for k,row in enumerate(c.itertuples(index=False),1):
        vals=[]
        for _ in range(cfg.surrogates):
            z=surrogate_best_of_lags(runs,row,cfg,rng)
            if np.isfinite(z):vals.append(z)
        a=np.asarray(vals,float);obs=float(row.improvement)
        ps.append((1+np.sum(a>=obs))/(len(a)+1) if len(a) else np.nan);med.append(float(np.median(a)) if len(a) else np.nan);q95.append(float(np.quantile(a,.95)) if len(a) else np.nan);mx.append(float(np.max(a)) if len(a) else np.nan)
        if k%max(1,len(c)//10)==0 or k==len(c):print(f"  null progress: {k}/{len(c)}",flush=True)
    c['null_p']=ps;c['null_median']=med;c['null_q95']=q95;c['null_max']=mx;c['fdr_q']=bh(c.null_p);c['retained']=c.fdr_q<=cfg.fdr_alpha
    return c

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='.');p.add_argument('--normal-file',default='TEP_FaultFree_Training.parquet');p.add_argument('--out',default='Router_V2_Network_TEP_corrected')
    p.add_argument('--max-lag',type=int,default=10);p.add_argument('--splits',type=int,default=5);p.add_argument('--test-fraction',type=float,default=.25);p.add_argument('--min-improvement',type=float,default=.01);p.add_argument('--min-stability',type=float,default=.60);p.add_argument('--min-directionality',type=float,default=.10);p.add_argument('--surrogates',type=int,default=200);p.add_argument('--fdr-alpha',type=float,default=.05);p.add_argument('--max-runs',type=int,default=None);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    cfg=Config(a.max_lag,a.splits,a.test_fraction,a.min_improvement,a.min_stability,a.min_directionality,a.surrogates,a.fdr_alpha,a.seed,1e-6,a.max_runs)
    print('='*78);print('ROUTER V2 EXPERIMENT 1B - CORRECTED TEP NETWORK VALIDATION');print('='*78);print('Fault-free training data only. Fair best-of-lags surrogate test.\n')
    runs,sensors,meta=load_normal(Path(a.data_dir)/a.normal_file,cfg.max_runs)
    print(f"Normal rows used: {meta['rows']:,}\nDetected normal runs: {meta['n_runs']}\nRun column: {meta['run_column']}\nTime/sample column: {meta['time_column']}\nRun lengths min/median/max: {meta['run_lengths_min']} / {meta['run_lengths_median']:.0f} / {meta['run_lengths_max']}\nSensors: {len(sensors)} ({sensors[0]} ... {sensors[-1]})\n")
    t=pair_table(runs,sensors,cfg);c=null_test(t,runs,cfg);cols=['source','target','null_p','null_median','null_q95','null_max','fdr_q','retained'];t=t.merge(c[cols],on=['source','target'],how='left');t['retained']=t.retained.fillna(False).astype(bool);r=t[t.retained].sort_values(['improvement','split_stability'],ascending=False)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);t.to_csv(out/'all_ordered_edges.csv',index=False);c.to_csv(out/'candidate_edges_null_test.csv',index=False);r.to_csv(out/'normal_network_edges.csv',index=False)
    t[['source','target','best_lag','improvement','split_stability','median_split_improvement','median_split_lag','directionality']].to_csv(out/'edge_stability.csv',index=False)
    t[['source','target','improvement','null_p','fdr_q','null_median','null_q95','null_max','retained']].to_csv(out/'shuffle_null_results.csv',index=False)
    nodes=[]
    for s in sensors:
        o=r[r.source==s];i=r[r.target==s];nodes.append(dict(sensor=s,out_degree=len(o),in_degree=len(i),out_strength=float(o.improvement.clip(lower=0).sum()),in_strength=float(i.improvement.clip(lower=0).sum())))
    nodes=pd.DataFrame(nodes);nodes.to_csv(out/'node_degrees.csv',index=False);mat=pd.DataFrame(0.,index=sensors,columns=sensors)
    for z in r.itertuples(index=False):mat.loc[z.source,z.target]=z.improvement
    mat.to_csv(out/'adjacency_improvement_matrix.csv',index_label='source');possible=len(sensors)*(len(sensors)-1);summary=dict(n_sensors=len(sensors),possible_directed_edges=possible,retained_edges=len(r),density=len(r)/possible)
    with open(out/'network_summary.json','w') as f:json.dump(dict(config=asdict(cfg),source=meta,network=summary,faulty_files_used=False,null_test='circular shift within each run; each surrogate scans exactly the same lags 1..max_lag and retains its best improvement',interpretation='directional predictive relationships, not proven causal edges'),f,indent=2)
    print('\nVALIDATION SUMMARY\n'+'-'*78);print(f"Retained edges: {len(r)} / {possible} (density {100*len(r)/possible:.2f}%)")
    if len(r):print('\nStrongest retained edges:\n'+r[['source','target','best_lag','improvement','split_stability','directionality','null_q95','null_max','fdr_q']].head(20).round(4).to_string(index=False))
    else:print('No edges passed the predeclared gates.')
    print('\nHighest out-degree sensors:\n'+nodes.sort_values(['out_degree','out_strength'],ascending=False).head(10).round(4).to_string(index=False));print(f"\nResults written to {out}/")
if __name__=='__main__':main()
