"""V3.0.0 — dataset-agnostic unsupervised episode-aware attribution.

Takes reconstruction-error arrays (time x features), normal calibration runs,
and sequence IDs. Neither anomaly times, fault IDs, nor source labels are used
by detector or attribution. Fault labels are available only to the separate
post-hoc evaluator.

IMPORTANT:
* This is an INITIAL fixed candidate, not a validated general solution.
* Calibration requires representative fault-free sequences. For completely
  unlabeled data, obtain a defensible normal reference first.
* Training, validation and test sequences MUST use the SAME trained AEs,
  preprocessing and error definition. Never refit on test.
* Attributions are observational associations, NOT proven causal origins.
* The benchmark's original separate test runs were already examined during
  V2.5 development, so V3.0 test scores are test-set-exposed follow-up.

General algorithm:
  (1) per-feature normal-training 99th percentile of error;
  (2) positive log error excess over threshold;
  (3) causal trailing 10-sample mean per feature;
  (4) max across features for episode score;
  (5) 95th percentile of normal-calibration PER-RUN MAXIMA for alarm level;
  (6) 3 consecutive episode-score exceedances => confirmed episode.
  (7) At confirmed episode, score each candidate by positive increase of
      mean log excess in next 30 samples over previous 20 samples.
      Onset detection never uses future values; attribution uses a declared
      30-sample observation delay. When no episode was detected, make a
      provisional ranking by maximum smoothed error and mark ABSTAINED.
      Never report no-alarm rankings as confirmed fault diagnoses.
  (8) rank scores decreasing, with deterministic feature-name tie breaking.

The relative changes are dimensionless, allowing the algorithm to operate
with any number of sensor-error columns and any fault mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# These are general, fixed prototype parameters, not dataset/fault-specific.
SENSOR_Q = 0.99
EPISODE_RUNMAX_Q = 0.95
SMOOTHING_SAMPLES = 10
ALARM_CONSECUTIVE = 3
PRE_SAMPLES = 20
POST_SAMPLES = 30
EPS = 1e-12

# TEP is an *adapter* for the generic core; only metadata and paths are here.
TEP_BASE = Path("/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution")
TEP_NORMAL = TEP_BASE / "TEP_cache/normal_error_df.parquet"
TEP_TRAIN = TEP_BASE / "TEP_cache/fault_error_df.parquet"
TEP_TEST_DIR = TEP_BASE / "TEP_testing_error_cache_frozen_v2_5"
TEP_OUTPUT = TEP_BASE / "V3_0_Development/Unsupervised_Episode_Attribution_V3_0_0"


def detect_features(columns):
    """Use every sensor error field; only explicit identifiers are excluded."""
    identifiers = {"faultNumber", "simulationRun", "sample", "Class", "label"}
    features = [c for c in columns if c not in identifiers]
    if len(features) < 1:
        raise ValueError("Expected at least one sensor reconstruction-error column")
    return features


def trailing_mean(x, window):
    x = np.asarray(x, dtype=np.float64)
    prefix = np.vstack([np.zeros((1, x.shape[1])),
                        np.cumsum(x, axis=0)])
    end = np.arange(1, len(x)+1)
    start = np.maximum(end-window, 0)
    return (prefix[end] - prefix[start]) / (end-start)[:, None]


def smooth_excess(errors, q):
    """Causal, independent-of-ground-truth sensor evidence."""
    x = np.asarray(errors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(q):
        raise ValueError("Sensor shape mismatch")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Errors must be finite and nonnegative")
    if not np.isfinite(q).all() or (q <= 0).any():
        raise ValueError("Normal q99 needs positive finite thresholds")
    exceedance = np.maximum(x / np.maximum(q[None,:], EPS) - 1.0, 0.0)
    return trailing_mean(np.log1p(exceedance), SMOOTHING_SAMPLES)


def first_confirmed_alarm(values, level):
    above = np.asarray(values) > level
    if len(above) < ALARM_CONSECUTIVE:
        return None
    starts = np.flatnonzero(np.convolve(
        above.astype(np.int8),
        np.ones(ALARM_CONSECUTIVE, dtype=np.int8),
        mode="valid") == ALARM_CONSECUTIVE)
    if not len(starts):
        return None
    # Zero-indexed confirmation position. Must wait for all 3 readings.
    return int(starts[0] + ALARM_CONSECUTIVE - 1)


def extract_runs(df, features, run_col, sample_col=None):
    """Validate run order and monotonically increasing samples when available."""
    if run_col not in df:
        raise ValueError(f"Missing run identifier: {run_col}")
    if not set(features).issubset(df.columns):
        raise ValueError("Sensor feature mismatch")
    for rid, group in df.groupby(run_col, sort=True):
        if sample_col and sample_col in group:
            group = group.sort_values(sample_col)
            samples = group[sample_col].to_numpy()
            if not np.isfinite(samples).all() or len(np.unique(samples)) != len(samples):
                raise ValueError(f"Non-unique or invalid samples for run {rid}")
            if np.any(np.diff(samples) <= 0):
                raise ValueError(f"Non-monotonic samples for run {rid}")
        else:
            samples = np.arange(1, len(group)+1, dtype=np.int64)
        values = group[features].to_numpy(dtype=np.float64)
        yield int(rid), samples, values


def fit_normal(normal_df, features, normal_run_col, max_cal_runs=350):
    """All calibration derived from normal error data, no source labels."""
    if max_cal_runs < 2:
        raise ValueError("Need at least 2 normal calibration runs")
    normal_runs = list(extract_runs(normal_df, features, normal_run_col))
    if len(normal_runs) < max_cal_runs:
        raise ValueError(f"Only {len(normal_runs)} normal runs; need {max_cal_runs}")
    calibration = normal_runs[:max_cal_runs]
    cal_array = np.vstack([v for _, _, v in calibration])
    q = np.quantile(cal_array, SENSOR_Q, axis=0)
    if (q <= EPS).any():
        raise ValueError("Some sensor q99 are near zero; needs a reviewed baseline")
    peaks = np.array([
        smooth_excess(v, q).max(axis=1).max()
        for _, _, v in calibration
    ])
    level = float(np.quantile(peaks, EPISODE_RUNMAX_Q))
    if not np.isfinite(level):
        raise ValueError("Invalid normal episode level")
    return {
        "features": features,
        "q99": q,
        "alarm_level": level,
        "normal_calibration_run_ids": [r for r, _, _ in calibration],
        "normal_validation_run_ids": [r for r, _, _ in normal_runs[max_cal_runs:]],
        "normal_run_arrays": normal_runs,
        "calibration_peak_values": peaks,
    }


def infer_one(values, features, q, level, samples):
    evidence = smooth_excess(values, q)
    process_score = evidence.max(axis=1)
    idx = first_confirmed_alarm(process_score, level)
    if idx is None:
        # Always return a reproducible ordering, but explicitly abstain.
        pre = np.zeros(len(features), dtype=float)
        post = evidence.max(axis=0)
        delta = post.copy()
        status = "no_episode_abstain"
    else:
        # Alarm time uses no future data. A fixed delayed diagnosis uses
        # post-event evidence available after the 30-sample observation period.
        start = max(0, idx-PRE_SAMPLES)
        pre = evidence[start:idx].mean(axis=0) if idx > start else np.zeros(len(features))
        post = evidence[idx:min(len(evidence), idx+POST_SAMPLES)].mean(axis=0)
        delta = np.maximum(post-pre, 0.0)
        status = "episode_confirmed"
    order = sorted(range(len(features)), key=lambda i: (-float(delta[i]), features[i]))
    rankings = []
    for rank, i in enumerate(order, 1):
        rankings.append({
            "sensor": features[i], "rank": rank, "change_score": float(delta[i]),
            "pre_evidence": float(pre[i]), "post_evidence": float(post[i]),
        })
    return {
        "episode_detected": idx is not None,
        "episode_confirmation_sample": (
            samples[idx].item() if idx is not None and hasattr(samples[idx], "item")
            else samples[idx] if idx is not None else None
        ),
        "diagnosis_available_sample": (
            samples[min(len(samples)-1, idx+POST_SAMPLES-1)].item()
            if idx is not None else None
        ),
        "status": status,
        "episode_peak_score": float(process_score.max()),
        "ranking": rankings,
    }


def add_fault_rows(frame, features, q, level, run_col, sample_col, fault_id):
    ranked = []
    overview = []
    for run, samples, values in extract_runs(frame, features, run_col, sample_col):
        ans = infer_one(values, features, q, level, samples)
        overview.append({
            "faultNumber": fault_id, "simulationRun": run,
            **{k:v for k,v in ans.items() if k != "ranking"},
        })
        for r in ans["ranking"]:
            ranked.append({"faultNumber": fault_id, "simulationRun": run,
                           "episode_detected": ans["episode_detected"], **r})
    return pd.DataFrame(ranked), pd.DataFrame(overview)


def save_inputs_metadata(out, cfg):
    f = out / "method_metadata.json"
    # Refuse to quietly mix calibrations/configurations across reruns.
    if f.exists():
        old = json.loads(f.read_text())
        if old != cfg:
            raise RuntimeError(f"Existing output has a different configuration: {f}")
    else:
        f.write_text(json.dumps(cfg, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--normal", type=Path, default=TEP_NORMAL)
    p.add_argument("--train-cache", type=Path, default=TEP_TRAIN)
    p.add_argument("--test-cache-dir", type=Path, default=TEP_TEST_DIR)
    p.add_argument("--output-base", type=Path, default=TEP_OUTPUT)
    p.add_argument("--calibration-runs", type=int, default=350)
    p.add_argument("--faults", type=int, nargs="+", default=list(range(1,16)),
                   help="Fault IDs to process; IDs organize data but are not used for ranking")
    args = p.parse_args()

    print("="*88,flush=True)
    print(f"V3.0.0 GENERIC LABEL-FREE EPISODE ATTRIBUTION | {args.split.upper()}",flush=True)
    print("="*88,flush=True)
    normal_df = pd.read_parquet(args.normal)
    features = detect_features(normal_df.columns)
    print(f"Detected {len(features)} sensor-error features",flush=True)
    fitted = fit_normal(normal_df, features, "simulationRun", args.calibration_runs)
    del normal_df
    q = fitted["q99"]
    level = fitted["alarm_level"]
    print(f"Normal calibration runs: {len(fitted['normal_calibration_run_ids'])}",flush=True)
    print(f"Normal-calibrated episode alarm level: {level:.6f}",flush=True)
    out = args.output_base / args.split
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "algorithm": "V3.0.0 label-free episode-aware attribution",
        "split": args.split, "features": features,
        "sensor_quantile": SENSOR_Q,
        "episode_runmax_quantile": EPISODE_RUNMAX_Q,
        "trailing_window": SMOOTHING_SAMPLES,
        "sustain_samples": ALARM_CONSECUTIVE,
        "pre_samples": PRE_SAMPLES,
        "post_samples": POST_SAMPLES,
        "episode_alarm_threshold": level,
        "normal_calibration_run_ids": fitted["normal_calibration_run_ids"],
        "normal_reference": str(args.normal),
        "normal_reference_sha256": hashlib.sha256(args.normal.read_bytes()).hexdigest(),
        "training_cache": str(args.train_cache),
        "testing_cache_dir": str(args.test_cache_dir),
        "processed_faults": args.faults,
        "training_based_calibration_only": True,
        "origin_labels_at_inference": False,
        "no_episode_behavior": "provide provisional ranking but abstain from diagnosis",
        "note": "All parameters fixed; independent TEP testing previously inspected for V2.5.",
    }
    save_inputs_metadata(out, cfg)
    # Independently measure healthy monitoring false-alarm rates on normal
    # training runs NOT used for normal-calibration.
    if args.split == "train":
        healthy = []
        for run, samples, values in fitted["normal_run_arrays"]:
            ans = infer_one(values, features, q, level, samples)
            healthy.append({
                "simulationRun": run,
                "calibration_or_validation": (
                    "calibration" if run in fitted["normal_calibration_run_ids"]
                    else "normal_validation"
                ),
                "episode_detected": ans["episode_detected"],
                "confirmation_sample": ans["episode_confirmation_sample"],
            })
        pd.DataFrame(healthy).to_csv(out/"normal_run_episode_metrics.csv",index=False)
        print("Normal validation false-alarm rate: "
              f"{100*pd.DataFrame(healthy).query('calibration_or_validation == \"normal_validation\"').episode_detected.mean():.2f}%",
              flush=True)

    import pyarrow.dataset as ds
    dataset = ds.dataset(str(args.train_cache),format="parquet") if args.split=="train" else None
    outputs = []
    for fault in args.faults:
        rp = out / f"rankings_fault_{fault:02d}.parquet"
        ep = out / f"episodes_fault_{fault:02d}.csv"
        if rp.exists() or ep.exists():
            if rp.exists() and ep.exists():
                print(f"Fault {fault:02d} already completed; skipping",flush=True)
                continue
            raise RuntimeError(f"Partial output for fault {fault}; review before retry")
        if args.split == "test":
            source = args.test_cache_dir / f"fault_{fault:02d}_errors.parquet"
            if not source.exists():
                raise FileNotFoundError(source)
            frame = pd.read_parquet(source)
        else:
            frame = dataset.to_table(
                filter=ds.field("faultNumber")==fault
            ).to_pandas()
        expected = set(features + ["faultNumber","simulationRun","sample"])
        if not expected.issubset(frame.columns):
            raise ValueError(f"Fault {fault} missing columns: {expected-set(frame.columns)}")
        if not (frame["faultNumber"] == fault).all():
            raise ValueError(f"Fault IDs mismatch in dataset for {fault}")
        ranks, episodes = add_fault_rows(
            frame, features, q, level, "simulationRun", "sample", fault
        )
        if ranks.empty or episodes.empty:
            raise RuntimeError(f"Fault {fault}: no runs to rank")
        if len(ranks) != len(episodes)*len(features):
            raise RuntimeError(f"Fault {fault}: incomplete rankings")
        ranks.to_parquet(rp,index=False)
        episodes.to_csv(ep,index=False)
        print(
            f"Fault {fault:02d}: {len(episodes)} runs, "
            f"episode coverage={100*episodes.episode_detected.mean():.2f}%, "
            f"rankings saved",flush=True
        )
        del frame, ranks, episodes
    print(f"Saved label-free outputs in {out}",flush=True)
    print("Next: run post-hoc evaluator with source labels stored in a SEPARATE file.",
          flush=True)


if __name__ == "__main__":
    main()
