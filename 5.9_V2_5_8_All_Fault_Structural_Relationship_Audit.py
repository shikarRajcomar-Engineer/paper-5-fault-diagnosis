"""V2.5.8: generic all-fault structural relationship audit.

For ALL 52 sensors and 15 mechanisms apply the same fixed healthy-network
and q99, three-sample leading-edge rules. Known origin is an evaluation
annotation ONLY. Network links are predictive associations, not causation.
No new ranking, training, or changes to frozen V2.5.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BASE = Path('/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution')
NORMAL = BASE/'TEP_cache/normal_error_df.parquet'
FAULT = BASE/'TEP_cache/fault_error_df.parquet'
RAW = Path('/content/drive/MyDrive/PHD/Paper5_Datasets/TEP_FaultFree_Training.parquet')
NETWORK = Path('/content/drive/MyDrive/PHD/Router_V2_Granger_Final/normal_network_edges.csv')
OUT = BASE/'V2_5_8_All_Fault_Structural_Relationship_Audit'
SENSORS = [f'xmeas_{i}' for i in range(1,42)] + [f'xmv_{i}' for i in range(1,12)]
ORIGINS = {1:'xmeas_4',2:'xmeas_4',3:'xmeas_23',4:'xmv_10',5:'xmeas_22',
           6:'xmv_3',7:'xmeas_4',8:'xmeas_4',9:'xmeas_23',10:'xmeas_4',
           11:'xmv_10',12:'xmv_11',13:'xmeas_9',14:'xmv_10',15:'xmv_11'}

def to_runs(df, label):
    df = df.sort_values(['simulationRun','sample'], kind='stable')
    sizes = df.groupby('simulationRun').size()
    assert len(sizes)==500 and (sizes==500).all(), f'{label}: expected 500 runs of 500'
    assert np.array_equal(df['sample'].to_numpy(),np.tile(np.arange(1,501),500)), f'{label}: sample order'
    return df['simulationRun'].drop_duplicates().to_numpy(), df[SENSORS].to_numpy(dtype=np.float32).reshape(500,500,52)

def get_onsets(a, threshold):
    above = a > threshold[None,None,:]
    sustained = above[:,:-2,:] & above[:,1:-1,:] & above[:,2:,:]
    onset = sustained.argmax(axis=1).astype(float)+1
    onset[~sustained.any(axis=1)] = np.inf
    return onset

def load_network():
    df = pd.read_csv(NETWORK)
    source = next((c for c in ('source','source_sensor','from_sensor') if c in df.columns),None)
    target = next((c for c in ('target','target_sensor','to_sensor') if c in df.columns),None)
    if source is None or target is None:
        raise ValueError(f'Unrecognized network columns: {list(df.columns)}')
    if 'retained' in df.columns:
        df = df.loc[df['retained'].astype(str).str.lower().isin(['true','1','yes'])]
    df = df.loc[df[source].isin(SENSORS) & df[target].isin(SENSORS)]
    m = np.zeros((52,52), dtype=bool)
    index = {name:i for i,name in enumerate(SENSORS)}
    for src,dst in zip(df[source],df[target]):
        i,j=index[src],index[dst]
        if i!=j: m[i,j]=True
    assert m.sum()>0, 'No matched retained network edges'
    return m

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('='*86, flush=True)
    print('V2.5.8 — ALL-FAULT GENERIC STRUCTURAL AUDIT', flush=True)
    print('='*86, flush=True)
    normal_err = pd.read_parquet(NORMAL,columns=SENSORS)
    normal_ids = pd.read_parquet(RAW,columns=['simulationRun','sample'])
    assert len(normal_err)==len(normal_ids)==250000
    ids = pd.concat([normal_ids.reset_index(drop=True), normal_err.reset_index(drop=True)],axis=1)
    _, normal_a = to_runs(ids,'Normal')
    threshold = np.quantile(normal_a,0.99,axis=(0,1))
    del ids,normal_a,normal_err,normal_ids
    links = load_network()
    print(f'Healthy predictive network: {int(links.sum())} edges; '
          f'{int((links.sum(axis=1)==0).sum())} sensors without outgoing edges',flush=True)
    data = ds.dataset(str(FAULT),format='parquet')
    comparison, candidates = [], []

    for fault_id, origin in ORIGINS.items():
        frame = data.to_table(
            filter=ds.field('faultNumber')==fault_id,
            columns=['simulationRun','sample']+SENSORS).to_pandas()
        _,a = to_runs(frame,f'Fault {fault_id}')
        t = get_onsets(a,threshold)
        present = np.isfinite(t)
        sensor_summaries = []
        for i,name in enumerate(SENSORS):
            seen = present[:,i]
            earlier = present & (t < t[:,i,None])
            later = present & (t > t[:,i,None])
            n_earlier = earlier.sum(axis=1)
            outward_earlier = (earlier & links[i][None,:]).sum(axis=1)
            inward_earlier = (earlier & links[:,i][None,:]).sum(axis=1)
            outward_later = (later & links[i][None,:]).sum(axis=1)
            covered_earlier = (earlier & (links[i][None,:] | links[:,i][None,:])).sum(axis=1)
            def med(x):
                return float(np.median(x[seen])) if seen.any() else np.nan
            features = {
                'faultNumber':fault_id, 'sensor':name,
                'is_known_origin':name==origin,
                'detected_fraction':float(seen.mean()),
                'healthy_out_degree':int(links[i].sum()),
                'healthy_in_degree':int(links[:,i].sum()),
                'median_onset_when_detected':med(t[:,i]),
                'median_earlier_sensors':med(n_earlier),
                'median_out_edges_to_earlier':med(outward_earlier),
                'median_in_edges_from_earlier':med(inward_earlier),
                'median_out_edges_to_later':med(outward_later),
                'median_directly_connected_earlier':med(covered_earlier),
                'runs_origin_to_earlier_any':float(np.mean(seen & (outward_earlier>0))),
                'runs_earlier_to_origin_any':float(np.mean(seen & (inward_earlier>0))),
            }
            sensor_summaries.append(features)
            candidates.append(features)
        # Ground-truth mapping only selects the already-computed feature row
        # for post-hoc comparison. It NEVER affects sensor calculations.
        origin_row = next(row for row in sensor_summaries if row['sensor']==origin)
        comparison.append(origin_row)
        print(f'Fault {fault_id:02d} {origin:9s} detected={origin_row["detected_fraction"]:.1%} '
              f'earlier={origin_row["median_earlier_sensors"]:.1f} '
              f'origin->earlier={origin_row["median_out_edges_to_earlier"]:.1f} '
              f'earlier->origin={origin_row["median_in_edges_from_earlier"]:.1f}',flush=True)
        del frame,a

    all_df = pd.DataFrame(candidates)
    origin_df = pd.DataFrame(comparison)
    p1 = OUT/'v2_5_8_all_sensor_structural_features.csv'
    p2 = OUT/'v2_5_8_origin_posthoc_comparison.csv'
    all_df.to_csv(p1,index=False)
    origin_df.to_csv(p2,index=False)
    print('\nFINAL ALL-FAULT COMPARISON',flush=True)
    print(origin_df.to_string(index=False,float_format=lambda x:f'{x:.3f}'),flush=True)
    print(f'\nSaved: {p1}\nSaved: {p2}',flush=True)
    print('V2.5.8 COMPLETE — diagnostic only, frozen V2.5 unchanged.',flush=True)

if __name__ == '__main__':
    main()
