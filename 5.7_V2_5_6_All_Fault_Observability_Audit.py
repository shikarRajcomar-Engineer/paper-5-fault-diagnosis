"""V2.5.6 — All-fault origin observability audit (diagnostic, not attribution).

Normal and fault AE errors only. Thresholds: per-sensor normal q99, K=3.
No model training, label-informed ranking, or V2.5 changes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BASE = Path('/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution')
NORMAL_FILE = BASE / 'TEP_cache/normal_error_df.parquet'
FAULT_FILE = BASE / 'TEP_cache/fault_error_df.parquet'
RAW_NORMAL_FILE = Path('/content/drive/MyDrive/PHD/Paper5_Datasets/TEP_FaultFree_Training.parquet')
OUT = BASE / 'V2_5_6_All_Fault_Observability_Audit'
OUT.mkdir(parents=True, exist_ok=True)
SENSORS = [f'xmeas_{i}' for i in range(1, 42)] + [f'xmv_{i}' for i in range(1, 12)]
ORIGINS = {1:'xmeas_4', 2:'xmeas_4', 3:'xmeas_23', 4:'xmv_10',
           5:'xmeas_22', 6:'xmv_3', 7:'xmeas_4', 8:'xmeas_4',
           9:'xmeas_23', 10:'xmeas_4', 11:'xmv_10', 12:'xmv_11',
           13:'xmeas_9', 14:'xmv_10', 15:'xmv_11'}
K = 3


def onset_matrix(a, thresholds):
    """Return 1-based first sustained crossing; NaN for no onset.
    a shape (runs, 500, 52)."""
    above = a > thresholds[None, None, :]
    sustained = above[:, :-2, :] & above[:, 1:-1, :] & above[:, 2:, :]
    has = sustained.any(axis=1)
    first = sustained.argmax(axis=1).astype(float) + 1
    first[~has] = np.nan
    return first


def rank_of_origin(onsets, strengths, origin_idx):
    """Diagnostic ordinal rank with onset first, threshold ratio as tie-break.
    Unobserved origins have no rank (NaN), rather than arbitrary column order.
    """
    t = onsets[:, origin_idx]
    s = strengths[:, origin_idx]
    valid = np.isfinite(t)
    earlier = np.sum(np.isfinite(onsets) & (onsets < t[:, None]), axis=1)
    tied_stronger = np.sum((onsets == t[:, None]) & (strengths > s[:, None]), axis=1)
    # exact ties use midrank rather than input column order
    exact = np.sum((onsets == t[:, None]) & (strengths == s[:, None]), axis=1)
    rank = 1.0 + earlier + tied_stronger + (exact - 1) / 2.0
    rank[~valid] = np.nan
    return rank


def describe_origin(a, thresholds, origin_idx):
    onsets = onset_matrix(a, thresholds)
    detected = np.isfinite(onsets)
    # Normalized strength at sustained onset, not the maximum over the whole run
    run_idx = np.arange(a.shape[0])[:, None]
    sensor_idx = np.arange(a.shape[2])[None, :]
    sample_idx = np.where(detected, onsets - 1, 0).astype(int)
    strengths = a[run_idx, sample_idx, sensor_idx] / np.maximum(thresholds[None, :], 1e-15)
    strengths[~detected] = 0.0
    rank = rank_of_origin(onsets, strengths, origin_idx)
    origin_onset = onsets[:, origin_idx]
    return {
        'onsets': onsets,
        'detected': detected,
        'rank': rank,
        'origin_onset': origin_onset,
        'origin_strength': strengths[:, origin_idx],
        'origin_detect_rate': float(np.mean(np.isfinite(origin_onset))),
        'origin_onset_median': float(np.nanmedian(origin_onset)) if np.isfinite(origin_onset).any() else np.nan,
        'origin_rank_median_detected': float(np.nanmedian(rank)) if np.isfinite(rank).any() else np.nan,
        'origin_top3_detected_fraction_all_runs': float(np.mean(rank <= 3)),
        'origin_top5_detected_fraction_all_runs': float(np.mean(rank <= 5)),
        'origin_earliest_fraction_all_runs': float(np.mean(np.isfinite(origin_onset) & (origin_onset == np.nanmin(np.where(detected, onsets, np.inf), axis=1)))),
        'median_other_sensors_earlier': float(np.median(np.sum(np.isfinite(onsets) & (onsets < origin_onset[:, None]), axis=1)[np.isfinite(origin_onset)])) if np.isfinite(origin_onset).any() else np.nan,
    }


def reshape_runs(df, cols, label):
    assert df['simulationRun'].nunique() == 500, f'{label}: expected 500 runs'
    df = df.sort_values(['simulationRun','sample'])
    counts = df.groupby('simulationRun').size()
    assert (counts == 500).all(), f'{label}: non-500 run length'
    assert np.array_equal(df['sample'].to_numpy(), np.tile(np.arange(1,501),500)), f'{label}: invalid sample sequence'
    return df[cols].to_numpy(dtype=np.float32).reshape(500,500,len(cols))


def main():
    print('='*85)
    print('V2.5.6 — ALL-FAULT ORIGIN OBSERVABILITY AUDIT')
    print('='*85)
    print('Loading normal reconstruction errors and verified raw normal run IDs...')
    normal_err = pd.read_parquet(NORMAL_FILE, columns=SENSORS)
    normal_ids = pd.read_parquet(RAW_NORMAL_FILE, columns=['simulationRun','sample'])
    assert len(normal_err) == len(normal_ids) == 250000