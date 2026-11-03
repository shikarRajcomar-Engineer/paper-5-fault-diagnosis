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
    normal_df = pd.concat([normal_ids.reset_index(drop=True), normal_err.reset_index(drop=True)], axis=1)
    normal_a = reshape_runs(normal_df, SENSORS, 'Normal')
    thresholds = np.quantile(normal_a.reshape(-1,52), .99, axis=0)
    normal_onsets = onset_matrix(normal_a, thresholds)
    normal_detect_rate = np.isfinite(normal_onsets).mean(axis=0)
    normal_median = np.median(normal_a, axis=(0,1))
    print('Normal baseline ready. Comparing 15 fault mechanisms...')

    dataset = ds.dataset(str(FAULT_FILE), format="parquet")
    all_rows = []
    # Stream row groups to avoid loading the full 5m x 52 cache into RAM.
    for fault_id, origin in ORIGINS.items():
        # Predicate filtering avoids converting all 5 million fault rows to pandas.
        table = dataset.to_table(
            columns=['faultNumber','simulationRun','sample']+SENSORS,
            filter=ds.field('faultNumber') == fault_id
        )
        fault_df = table.to_pandas()
        a = reshape_runs(fault_df, SENSORS, f'Fault {fault_id}')
        origin_idx = SENSORS.index(origin)
        d = describe_origin(a, thresholds, origin_idx)
        fault_detect_rates = d['detected'].mean(axis=0)
        fault_median = np.median(a, axis=(0,1))
        # Delta: sustained detection above the rate seen in normal runs.
        detection_lift = float(fault_detect_rates[origin_idx] - normal_detect_rate[origin_idx])
        # Relative median AE-error shift (not a formal significance test).
        median_ratio = float(fault_median[origin_idx] / max(normal_median[origin_idx], 1e-15))
        # Other sensors showing a stronger-than-normal sustained detection rate.
        other_lift = fault_detect_rates - normal_detect_rate
        others = [(SENSORS[i], round(float(other_lift[i]),3)) for i in range(52) if i != origin_idx]
        others.sort(key=lambda x: x[1], reverse=True)
        row = {
            'faultNumber': fault_id, 'origin': origin,
            'normal_origin_detection_rate': round(float(normal_detect_rate[origin_idx]),4),
            'fault_origin_detection_rate': round(d['origin_detect_rate'],4),
            'origin_detection_lift': round(detection_lift,4),
            'origin_fault_normal_median_error_ratio': round(median_ratio,4),
            'origin_median_onset_when_detected': d['origin_onset_median'],
            'origin_median_rank_when_detected': d['origin_rank_median_detected'],
            'origin_top3_fraction_all_runs': round(d['origin_top3_detected_fraction_all_runs'],4),
            'origin_top5_fraction_all_runs': round(d['origin_top5_detected_fraction_all_runs'],4),
            'origin_earliest_fraction_all_runs': round(d['origin_earliest_fraction_all_runs'],4),
            'median_number_other_sensors_earlier_when_detected': d['median_other_sensors_earlier'],
            'number_sensors_detection_lift_gt_20pp': int(np.sum(other_lift > .20)),
            'largest_other_sensor_detection_lifts': '; '.join(f'{s}:{v:+.3f}' for s,v in others[:3]),
        }
        all_rows.append(row)
        print(f"Fault {fault_id:2d} | {origin:9s} | normal detect {100*normal_detect_rate[origin_idx]:5.1f}% | fault detect {100*d['origin_detect_rate']:5.1f}% | lift {100*detection_lift:+6.1f}pp | median detected rank {d['origin_rank_median_detected']:.1f}", flush=True)
        del table, fault_df, a

    result = pd.DataFrame(all_rows)
    path = OUT/'v2_5_6_all_fault_origin_observability.csv'
    result.to_csv(path, index=False)
    print('\n'+'='*85+'\nSUMMARY\n'+'='*85)
    show = ['faultNumber','origin','normal_origin_detection_rate','fault_origin_detection_rate','origin_detection_lift','origin_fault_normal_median_error_ratio','origin_median_rank_when_detected','origin_top3_fraction_all_runs']
    print(result[show].to_string(index=False))
    print('\nSaved:', path)
    print('\nNOTE: These are diagnostic associations, NOT a new classifier or a proof of causality.')
    print('A detection rate is only evidence of fault observability when it exceeds the healthy-run rate.')
    print('No V2.5 ranking was modified. Missing onsets remain unranked rather than assigned arbitrary last-place ranks.')

if __name__ == '__main__':
    main()
