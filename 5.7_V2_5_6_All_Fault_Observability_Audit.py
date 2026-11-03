"""V2.5.6 — All-fault origin observability audit (diagnostic only).

Compare normal vs fault reconstruction errors for all 15 TEP faults with known
origins. Frozen V2.5 thresholds (normal sensor q99) and onset rule (K=3).

Ground-truth origins are used ONLY to evaluate origin observability. This does
not train models, select ranking weights, or modify the frozen V2.5 method.
TEP labels were already studied in earlier experiments; this is exploratory.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BASE = Path('/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution')
NORMAL_FILE = BASE / 'TEP_cache/normal_error_df.parquet'
FAULT_FILE = BASE / 'TEP_cache/fault_error_df.parquet'
RAW_NORMAL_FILE = Path('/content/drive/MyDrive/PHD/Paper5_Datasets/TEP_FaultFree_Training.parquet')
OUT = BASE / 'V2_5_6_All_Fault_Observability_Audit'
SENSORS = [f'xmeas_{i}' for i in range(1, 42)] + [f'xmv_{i}' for i in range(1, 12)]
ORIGINS = {
    1: 'xmeas_4', 2: 'xmeas_4', 3: 'xmeas_23', 4: 'xmv_10',
    5: 'xmeas_22', 6: 'xmv_3', 7: 'xmeas_4', 8: 'xmeas_4',
    9: 'xmeas_23', 10: 'xmeas_4', 11: 'xmv_10', 12: 'xmv_11',
    13: 'xmeas_9', 14: 'xmv_10', 15: 'xmv_11',
}
QUANTILE = 0.99
K = 3
N_RUNS = 500
N_SAMPLES = 500


def reshape_runs(df, label):
    """Validate run IDs and reshape to (500 runs, 500 samples, 52 sensors)."""
    df = df.sort_values(['simulationRun', 'sample'], kind='stable')
    run_ids = df['simulationRun'].to_numpy()
    samples = df['sample'].to_numpy()
    unique_ids, counts = np.unique(run_ids, return_counts=True)
    if len(unique_ids) != N_RUNS or not np.all(counts == N_SAMPLES):
        raise ValueError(f'{label}: expected 500 runs with 500 samples per run')
    if not np.array_equal(samples, np.tile(np.arange(1, N_SAMPLES + 1), N_RUNS)):
        raise ValueError(f'{label}: unexpected sample numbering/order')
    return unique_ids, df[SENSORS].to_numpy(dtype=np.float32).reshape(
        N_RUNS, N_SAMPLES, len(SENSORS)
    )


def get_onsets_and_strengths(a, thresholds):
    """First 3 consecutive errors above q99, with 1-based onset samples."""
    above = a > thresholds[None, None, :]
    sustained = above[:, :N_SAMPLES - K + 1, :].copy()
    for j in range(1, K):
        sustained &= above[:, j:j + sustained.shape[1], :]
    detected = sustained.any(axis=1)
    onsets = sustained.argmax(axis=1).astype(float) + 1
    onsets[~detected] = np.nan
    first = np.where(detected, onsets - 1, 0).astype(int)
    strengths = a[
        np.arange(a.shape[0])[:, None], first,
        np.arange(a.shape[2])[None, :]
    ] / np.maximum(thresholds[None, :], 1e-15)
    strengths[~detected] = 0.0
    return onsets, strengths


def origin_midrank(onsets, strengths, index):
    """Diagnostic rank; undetected origin is unranked, not arbitrarily last."""
    origin_time = onsets[:, index]
    origin_strength = strengths[:, index]
    earlier = np.sum(np.isfinite(onsets) & (onsets < origin_time[:, None]), axis=1)
    stronger_at_same_time = np.sum(
        (onsets == origin_time[:, None]) &
        (strengths > origin_strength[:, None]), axis=1
    )
    exact_ties = np.sum(
        (onsets == origin_time[:, None]) &
        (strengths == origin_strength[:, None]), axis=1
    )
    ranks = 1 + earlier + stronger_at_same_time + (exact_ties - 1) / 2
    ranks[~np.isfinite(origin_time)] = np.nan
    return ranks, earlier


def cliffs_delta(fault, normal):
    """Run-level Cliff's delta: 500 fault vs 500 healthy median errors."""
    fault = np.asarray(fault, dtype=float)
    normal = np.asarray(normal, dtype=float)
    return float((np.sum(fault[:, None] > normal[None, :]) -
                  np.sum(fault[:, None] < normal[None, :])) /
                 (len(fault) * len(normal)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('=' * 90, flush=True)
    print('V2.5.6 — ALL-FAULT ORIGIN OBSERVABILITY AUDIT', flush=True)
    print('=' * 90, flush=True)
    print('Loading normal cache and original run IDs...', flush=True)

    # Prior V2.5.3a verified the raw normal data and cache row order.
    normal_errors = pd.read_parquet(NORMAL_FILE, columns=SENSORS)
    normal_ids = pd.read_parquet(RAW_NORMAL_FILE, columns=['simulationRun', 'sample'])
    if len(normal_errors) != 250000 or len(normal_ids) != len(normal_errors):
        raise ValueError('Normal error cache and raw run IDs must each have 250,000 rows')
    normal_df = pd.concat([
        normal_ids.reset_index(drop=True), normal_errors.reset_index(drop=True)
    ], axis=1)
    _, normal_array = reshape_runs(normal_df, 'Healthy')
    del normal_df, normal_errors, normal_ids

    thresholds = np.quantile(normal_array, QUANTILE, axis=(0, 1))
    normal_onsets, _ = get_onsets_and_strengths(normal_array, thresholds)
    normal_detection = np.isfinite(normal_onsets).mean(axis=0)
    normal_run_medians = np.median(normal_array, axis=1)
    print(f'Normal array: {normal_array.shape}', flush=True)
    print(f'Using frozen normal q99 and K={K}', flush=True)

    dataset = ds.dataset(str(FAULT_FILE), format='parquet')
    required = {'faultNumber', 'simulationRun', 'sample', *SENSORS}
    missing = required.difference(dataset.schema.names)
    if missing:
        raise ValueError(f'Fault cache missing columns: {sorted(missing)}')

    summary_rows = []
    run_rows = []
    print('\nProcessing 15 known-origin faults:', flush=True)
    for fault_number, origin in ORIGINS.items():
        table = dataset.to_table(
            filter=(ds.field('faultNumber') == fault_number),
            columns=['simulationRun', 'sample'] + SENSORS,
        )
        frame = table.to_pandas()
        run_ids, fault_array = reshape_runs(frame, f'Fault {fault_number}')
        index = SENSORS.index(origin)
        onsets, strengths = get_onsets_and_strengths(fault_array, thresholds)
        ranks, earlier = origin_midrank(onsets, strengths, index)
        observed = np.isfinite(onsets[:, index])

        normal_rate = float(normal_detection[index])
        fault_rate = float(np.mean(observed))
        lift = fault_rate - normal_rate  # absolute percentage-point difference / 100
        normal_medians = normal_run_medians[:, index]
        fault_medians = np.median(fault_array[:, :, index], axis=1)
        delta = cliffs_delta(fault_medians, normal_medians)
        healthy_median = float(np.median(normal_medians))
        ratio = (float(np.median(fault_medians)) / healthy_median
                 if healthy_median > 0 else np.nan)
        earliest_per_run = np.min(
            np.where(np.isfinite(onsets), onsets, np.inf), axis=1
        )

        summary_rows.append({
            'faultNumber': fault_number,
            'true_origin': origin,
            'normal_origin_detection_rate': normal_rate,
            'fault_origin_detection_rate': fault_rate,
            'origin_detection_lift': lift,
            'origin_run_median_error_ratio': ratio,
            'origin_run_median_cliffs_delta': delta,
            'origin_earliest_fraction_all_runs': float(np.mean(
                observed & (onsets[:, index] == earliest_per_run)
            )),
            'origin_top3_fraction_all_runs': float(np.mean(ranks <= 3)),
            'origin_top5_fraction_all_runs': float(np.mean(ranks <= 5)),
            'origin_median_rank_when_detected': (
                float(np.median(ranks[observed])) if observed.any() else np.nan
            ),
            'median_other_sensors_earlier_when_detected': (
                float(np.median(earlier[observed])) if observed.any() else np.nan
            ),
            'origin_median_onset_when_detected': (
                float(np.median(onsets[observed, index])) if observed.any() else np.nan
            ),
            'n_runs': len(run_ids),
        })

        for n, run_id in enumerate(run_ids):
            run_rows.append({
                'faultNumber': fault_number,
                'simulationRun': int(run_id),
                'true_origin': origin,
                'origin_detected': bool(observed[n]),
                'origin_onset': (float(onsets[n, index]) if observed[n] else np.nan),
                'origin_rank_if_detected': (float(ranks[n]) if observed[n] else np.nan),
                'other_sensors_earlier': (int(earlier[n]) if observed[n] else np.nan),
            })

        print(
            f'Fault {fault_number:02d} | {origin:9s} | '
            f'healthy {normal_rate*100:5.1f}% | '
            f'fault {fault_rate*100:5.1f}% | '
            f'lift {lift*100:+6.1f} pp | '
            f'Top3 {np.mean(ranks <= 3)*100:5.1f}% | '
            f'delta {delta:+.3f}', flush=True
        )
        del table, frame, fault_array

    summary = pd.DataFrame(summary_rows)
    per_run = pd.DataFrame(run_rows)
    summary_path = OUT / 'v2_5_6_all_fault_origin_observability.csv'
    run_path = OUT / 'v2_5_6_origin_audit_by_run.csv'
    summary.to_csv(summary_path, index=False)
    per_run.to_csv(run_path, index=False)

    print('\n' + '=' * 90, flush=True)
    print('FINAL SUMMARY — ALL 15 FAULTS', flush=True)
    print('=' * 90, flush=True)
    columns = [
        'faultNumber', 'true_origin', 'fault_origin_detection_rate',
        'origin_detection_lift', 'origin_run_median_cliffs_delta',
        'origin_top3_fraction_all_runs', 'origin_median_rank_when_detected',
    ]
    print(summary[columns].to_string(index=False, float_format=lambda x: f'{x:.3f}'), flush=True)
    print(f'\nSaved: {summary_path}', flush=True)
    print(f'Saved: {run_path}', flush=True)
    print('V2.5.6 COMPLETE. Frozen V2.5 unchanged.', flush=True)


if __name__ == '__main__':
    main()
