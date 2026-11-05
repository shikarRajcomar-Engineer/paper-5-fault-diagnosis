"""V2.5.9 — generic structural discrimination audit (diagnostic only).

All candidate sensors use the SAME V2.5.8 features. True-origin labels
are only consulted AFTER per-fault candidate ranks are computed.
Fault-level post-hoc ranks are NOT deployable run-level accuracy.
The healthy network is predictive, not necessarily causal.
No retraining or changes to frozen V2.5.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path('/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution')
SOURCE = BASE/'V2_5_8_All_Fault_Structural_Relationship_Audit'/'v2_5_8_all_sensor_structural_features.csv'
OUT = BASE/'V2_5_9_Structural_Discrimination_Audit'
SENSORS = [f'xmeas_{i}' for i in range(1,42)] + [f'xmv_{i}' for i in range(1,12)]
ORIGINS = {1:'xmeas_4',2:'xmeas_4',3:'xmeas_23',4:'xmv_10',5:'xmeas_22',
           6:'xmv_3',7:'xmeas_4',8:'xmeas_4',9:'xmeas_23',10:'xmeas_4',
           11:'xmv_10',12:'xmv_11',13:'xmeas_9',14:'xmv_10',15:'xmv_11'}

def add_features(x):
    """Features/ranks computed without using labels or is_known_origin."""
    x=x.copy()
    seen=x['detected_fraction']>0
    out=x['healthy_out_degree'].to_numpy(dtype=float)
    inward=x['healthy_in_degree'].to_numpy(dtype=float)
    a=x['median_out_edges_to_earlier'].to_numpy(dtype=float)
    b=x['median_in_edges_from_earlier'].to_numpy(dtype=float)
    x['outward_fraction']=np.divide(a,out,out=np.full(len(x),np.nan),where=out>0)
    x['inward_fraction']=np.divide(b,inward,out=np.full(len(x),np.nan),where=inward>0)
    x.loc[~seen,['outward_fraction','inward_fraction']]=np.nan
    x['direction_contrast']=x['outward_fraction']-x['inward_fraction']
    x['outward_available']=seen & (x['healthy_out_degree']>0)
    x['contrast_available']=x['outward_available'] & (x['healthy_in_degree']>0)
    x['outward_rank']=np.nan
    x['contrast_rank']=np.nan
    x['outward_n']=0
    x['contrast_n']=0
    for _,p in x.groupby('faultNumber'):
        ix=p.index
        for prefix,col in [('outward','outward_fraction'),('contrast','direction_contrast')]:
            elig=p[prefix+'_available'] & p[col].notna()
            x.loc[ix,prefix+'_n']=int(elig.sum())
            # Average ranks prevent arbitrary sensor-order tie breaking.
            x.loc[p.index[elig],prefix+'_rank']=p.loc[elig,col].rank(
                ascending=False,method='average').to_numpy()
    return x

def evaluate(x):
    """Origin labels used for reporting only, not to calculate features."""
    rows=[]
    for fault,p in x.groupby('faultNumber'):
        origin=ORIGINS[int(fault)]
        o=p.loc[p.sensor==origin]
        assert len(o)==1
        o=o.iloc[0]
        result={'faultNumber':int(fault),'true_origin':origin,
                'origin_detected_fraction':float(o['detected_fraction']),
                'origin_median_earlier_sensors':o['median_earlier_sensors']}
        for prefix,col in [('outward','outward_fraction'),('contrast','direction_contrast')]:
            valid=p.loc[p[prefix+'_available']]
            score=o[col]
            rank=o[prefix+'_rank']
            result[prefix+'_n']=int(len(valid))
            result['origin_'+prefix+'_score']=score
            result['origin_'+prefix+'_rank']=rank
            result['origin_'+prefix+'_diagnostic_top3']=bool(pd.notna(rank) and rank<=3)
            peers=valid.loc[valid.sensor!=origin,col]
            result[prefix+'_fraction_competitors_below']=(
                float((peers<score).mean()) if pd.notna(rank) else np.nan)
            result[prefix+'_fraction_competitors_tied']=(
                float((peers==score).mean()) if pd.notna(rank) else np.nan)
        if bool(o['outward_available']):
            matched=p.loc[p['outward_available'] & p.sensor.ne(origin) &
                          ((p.healthy_out_degree-o.healthy_out_degree).abs()<=2)]
            result['degree_matched_n']=len(matched)
            result['degree_matched_above_origin']=int(
                (matched['outward_fraction']>o['outward_fraction']).sum())
        else:
            result['degree_matched_n']=0
            result['degree_matched_above_origin']=np.nan
        rows.append(result)
    return pd.DataFrame(rows)

def main():
    print('='*85,flush=True)
    print('V2.5.9 — GENERIC STRUCTURAL DISCRIMINATION AUDIT',flush=True)
    print('='*85,flush=True)
    print(f'Reading: {SOURCE}',flush=True)
    if not SOURCE.exists():
        raise FileNotFoundError('Run V2.5.8 first to produce all-sensor feature CSV')
    raw=pd.read_csv(SOURCE)
    needed={'faultNumber','sensor','detected_fraction','healthy_out_degree',
            'healthy_in_degree','median_out_edges_to_earlier',
            'median_in_edges_from_earlier','median_earlier_sensors'}
    missing=needed-set(raw.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if len(raw)!=780 or set(raw.faultNumber)!=set(ORIGINS):
        raise ValueError('Expected 15 faults x 52 sensors')
    for fault,p in raw.groupby('faultNumber'):
        if len(p)!=52 or set(p.sensor)!=set(SENSORS):
            raise ValueError(f'Fault {fault}: sensor mismatch')
    candidates=add_features(raw)
    results=evaluate(candidates)
    OUT.mkdir(parents=True,exist_ok=True)
    out1=OUT/'v2_5_9_all_candidate_structural_discrimination.csv'
    out2=OUT/'v2_5_9_origin_discrimination_diagnostic.csv'
    candidates.to_csv(out1,index=False)
    results.to_csv(out2,index=False)
    cols=['faultNumber','true_origin','origin_detected_fraction',
          'outward_n','origin_outward_rank','contrast_n','origin_contrast_rank',
          'degree_matched_n','degree_matched_above_origin']
    print('\nORIGIN VS COMPETITORS (DIAGNOSTIC MIDRANKS; NOT ATTRIBUTION ACCURACY)',flush=True)
    print(results[cols].to_string(index=False,float_format=lambda v:f'{v:.3f}'),flush=True)
    for prefix in ('outward','contrast'):
        valid=results['origin_'+prefix+'_rank'].notna()
        n=int(valid.sum())
        top=int(results.loc[valid,'origin_'+prefix+'_diagnostic_top3'].sum())
        print(f'{prefix}: origin feature available in {n}/15 faults; '
              f'origin diagnostic rank <=3 in {top}/{n} of these',flush=True)
    print(f'\nSaved: {out1}\nSaved: {out2}',flush=True)
    print('Exploratory, post-label fault-level audit; not held-out validation.',flush=True)
    print('V2.5.9 COMPLETE — frozen V2.5 unchanged.',flush=True)

if __name__=='__main__':
    main()
