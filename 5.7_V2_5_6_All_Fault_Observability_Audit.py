"""V2.5.6: All-fault origin observability audit (diagnostic only).

Uses existing per-sensor AE reconstruction-error caches. Does not train,
change V2.5, or use ground-truth origins to produce sensor rankings.
Ground-truth origins are used only to audit the resulting evidence.
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
SENSORS = [f'xmeas_{i}' for i in range(1, 42)] + [f'xmv_{i}' for i in range(1, 12)]
ORIGINS = {1:'xmeas_4', 2:'xmeas_4', 3:'xmeas_23', 4:'xmv_10',
           5:'xmeas_22', 6:'xmv_3', 7:'xmeas_4', 8:'xmeas_4',
           9:'xmeas_23', 10:'xmeas_4', 11:'xmv_10', 12:'xmv_11',
           13:'xmeas_9', 14:'xmv_10', 15:'xmv_11'}
QUANTILE = 0.99
K = 3
N_RUNS = 500
N_SAMPLES = 500


def reshape_runs(df, label):
    """Validate run boundaries and return (500, 500, 52) array."""
    df = df.sort_values(['simulationRun', 'sample'], kind='stable')
    runs = df['simulationRun'].to_numpy()
    samples = df['sample'].to_numpy()
    unique, counts = np.unique(runs, return_counts=True)
    if len(unique) != N_RUNS or not np.all(counts == N_SAMPLES):
        raise ValueError(f'{label}: expected 500 runs of 500 samples')
    if not np.array_equal(samples, np.tile(np.arange(1, N_SAMPLES + 1), N_RUNS)):
        raise ValueError(f'{label}: sample ordering/numbering invalid')
    return df[SENSORS].to_numpy(dtype=np.float32).reshape(N_RUNS, N_SAMPLES, len(SENSORS))


def onset_and_strength(a, thresholds):
    """First K consecutive exceedances; strength at the first crossing."""
    above = a > thresholds[None, None, :]
    sustained = above[:, :-K + 1, :].copy()
    for offset in range(1, K):
        sustained &= above[:, offset:offset + sustained.shape[1], :]
    detected = sustained.any(axis=1)
    onset = sustained.argmax(axis=1).astype(float) + 1  # 1-based sample
    onset[~detected] = np.nan
    sample_idx = np.where(detected, onset - 1, 0).astype(int)
    strength = a[np.arange(a.shape[0])[:, None], sample_idx,
                 np.arange(a.shape[2])[None, :]] / np.maximum(thresholds[None, :], 1e-15)
    strength[~detected] = 0.0
    return onset, strength


def origin_rank(onsets, strengths, idx):
    """Midrank among detected sensors; undetected origin gets NaN.

    Unlike frozen V2.5, this diagnostic avoids arbitrary ordering of
    undetected sensors. This is NOT a replacement for V2.5 rankings.
    """
    t = onsets[:, idx]
    s = strengths[:, idx]
    earlier = np.sum(np.isfinite(onsets) & (onsets < t[:, None]), axis=1)
    stronger_ties = np.sum((onsets == t[:, None]) & (strengths > s[:, None]), axis=1)
    exact_ties = np.sum((onsets == t[:, None]) & (strengths == s[:, None]), axis=1)
    rank = 1 + earlier + stronger_ties + (exact_ties - 1) / 2.0
    rank[~np.isfinite(t)] = np.nan
    return rank, earlier


def cliffs_delta(x, y):
    """Exact run-level Cliff's delta, x=Fault, y=Normal (500 each)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float((np.sum(x[:, None] > y[None, :]) -
                  np.sum(x[:, None] < y[None, :])) / (len(x) * len(y)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('=' * 95, flush=True)
    print('V2.5.6 — ALL-FAULT ORIGIN OBSERVABILITY AUDIT', flush=True)
    print('=' * 95, flush=True)
    print('Loading normal reconstruction errors and verified run identifiers...', flush=True)

    normal_err = pd.read_parquet(NORMAL_FILE, columns=SENSORS)
    normal_ids = pd.read_parquet(RAW_NORMAL_FILE, columns=['simulationRun', 'sample'])
    if len(normal_err) != len(normal_ids) or len(normal_err) != N_RUNS * N_SAMPLES:
        raise ValueError('Normal cache and raw run IDs have incompatible lengths')
    # Prior V2.5.3a verification established identical row order.
    normal_df = pd.concat([normal_ids.reset_index(drop=True),
                           normal_err.reset_index(drop=True)], axis=1)
    normal_array = reshape_runs(normal_df, 'Normal')