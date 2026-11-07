"""V3.0 EXPERIMENT 1 — label-free collective abnormal-episode detection.

DEVELOPMENT ONLY: reads normal TRAINING error cache to calibrate and fault
TRAINING errors to evaluate. Does not touch TEP Testing or frozen V2.5.

Predeclared method:
  - Calibrate each of 52 sensor error q99 thresholds on normal runs 1..350.
  - At each time sample count how many sensors exceed their threshold.
  - Form a trailing 5-sample rolling mean of this count (causal).
  - Set an episode alarm level to the 95th percentile of the PER-RUN MAXIMUM
    rolling score among normal calibration runs 1..350.
  - Detect earliest 3 consecutive rolling scores STRICTLY above alarm level.
  - No injected fault time / fault origin used by the detector.

Evaluation ONLY: use benchmark training injection sample 21 to compute
false-before-fault, post-fault detection, and detection latency.
Hold out normal and fault run IDs 351..500 for internal validation.
These are not independent untouched test sets: they belong to training data.

Run:
  python -u 7.0_V3_0_Training_Episode_Detector.py

Outputs remain separate under V3_0_Development/Episode_Experiment_1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BASE = Path("/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution")
NORMAL_PATH = BASE / "TEP_cache/normal_error_df.parquet"
FAULT_PATH = BASE / "TEP_cache/fault_error_df.parquet"
OUT = BASE / "V3_0_Development/Episode_Experiment_1"
SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]

# Fixed, documented first-prototype configuration; no tuning on fault/test labels.
SENSOR_QUANTILE = 0.99
RUN_MAX_QUANTILE = 0.95
TRAILING_WINDOW = 5
SUSTAINED_SAMPLES = 3
FAULT_INJECTION_SAMPLE = 21   # EVALUATION ONLY
NORMAL_CALIBRATION_RUNS = range(1, 351)
INTERNAL_VALIDATION_RUNS = range(351, 501)
FAULTS = list(range(1, 16))
RUN_LENGTH = 500


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Past-and-present trailing mean, including partial initial windows."""
    cumsum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - width)
    return (cumsum[end] - cumsum[start]) / (end - start)


def first_episode(scores: np.ndarray, threshold: float) -> int | None:
    """Return zero-based onset of first K consecutive strict exceedances."""
    flags = scores > threshold
    if len(flags) < SUSTAINED_SAMPLES:
        return None
    positions = np.flatnonzero(
        np.convolve(flags.astype(np.int8),
                    np.ones(SUSTAINED_SAMPLES, dtype=np.int8),
                    mode="valid") == SUSTAINED_SAMPLES
    )
    return int(positions[0]) if len(positions) else None


def score_block(errors: np.ndarray, sensor_thresholds: np.ndarray) -> np.ndarray:
    if not np.isfinite(errors).all():
        raise ValueError("Missing or nonfinite sensor reconstruction errors")
    counts = np.sum(errors > sensor_thresholds[None, :], axis=1)
    return trailing_mean(counts, TRAILING_WINDOW)


def normal_runs():
    """Normal cache lacks a sample field; preserve its existing row order."""
    df = pd.read_parquet(NORMAL_PATH)
    if not set(SENSORS + ["simulationRun"]).issubset(df.columns):
        raise ValueError("Normal cache schema is incompatible")
    if len(df) != 500 * RUN_LENGTH:
        raise ValueError("Unexpected normal-training rows; check cache provenance")
    # Explicit check that rows are contiguous by run; sample order has been
    # preserved by the original cache construction from training input.
    vals = df["simulationRun"].to_numpy()
    runs = {}
    for run in range(1, 501):
        where = np.flatnonzero(vals == run)
        if len(where) != RUN_LENGTH or np.any(np.diff(where) != 1):
            raise ValueError(f"Normal run {run} not contiguous with 500 rows")
        runs[run] = df.iloc[where][SENSORS].to_numpy(dtype=np.float32, copy=True)
    return runs


def summarize(rows: pd.DataFrame):
    """No post-fault-only detection credited when the first alarm was pre-fault."""
    def summarize_subset(subset):
        return {
            "runs": int(len(subset)),
            "any_episode_%": 100 * float(subset.episode_detected.mean()),
            "first_episode_before_injection_%": 100 * float(subset.early_alarm.mean()),
            "first_episode_on_or_after_injection_%": 100 * float(subset.postfault_first_alarm.mean()),
            "median_postfault_detection_delay": (
                float(subset.loc[subset.postfault_first_alarm, "detection_delay"].median())
                if subset.postfault_first_alarm.any() else None
            ),
        }
    overall = []
    for (split, fault), frame in rows.groupby(["split", "faultNumber"], sort=True):
        overall.append({"split": split, "faultNumber": int(fault), **summarize_subset(frame)})
    return pd.DataFrame(overall)


def main():
    print("=" * 88, flush=True)
    print("V3.0 EXPERIMENT 1 — TRAINING-ONLY LABEL-FREE EPISODE DETECTION", flush=True)
    print("=" * 88, flush=True)
    for path in [NORMAL_PATH, FAULT_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)
    print("Calibrating from normal TRAINING runs 1..350 (no testing data).",
          flush=True)
    normal = normal_runs()
    calibrating = np.concatenate(
        [normal[r] for r in NORMAL_CALIBRATION_RUNS], axis=0
    )
    sensor_q99 = np.quantile(calibrating, SENSOR_QUANTILE, axis=0)
    del calibrating
    if not np.all(np.isfinite(sensor_q99)):
        raise ValueError("Invalid normal sensor thresholds")

    max_normal_scores = np.array([
        float(np.max(score_block(normal[r], sensor_q99)))
        for r in NORMAL_CALIBRATION_RUNS
    ])
    episode_threshold = float(np.quantile(max_normal_scores, RUN_MAX_QUANTILE))
    print(f"Sensor thresholds: 52 normal-training q99 values", flush=True)
    print(f"Collective-score alarm threshold: {episode_threshold:.6f}",flush=True)
    print(f"Score: trailing {TRAILING_WINDOW}-sample mean of abnormal-sensor count",
          flush=True)
    print(f"Alarm: {SUSTAINED_SAMPLES} consecutive scores > alarm threshold",
          flush=True)

    normal_rows = []
    for run in range(1, 501):
        score = score_block(normal[run], sensor_q99)
        onset = first_episode(score, episode_threshold)
        normal_rows.append({
            "simulationRun": run,
            "split": "normal_calibration" if run <= 350 else "normal_validation",
            "episode_detected": onset is not None,
            "episode_sample": None if onset is None else onset + 1,
            "max_score": float(np.max(score)),
        })
    del normal
    normal_df = pd.DataFrame(normal_rows)
    print("Normal calibration false-episode rate: "
          f"{100*normal_df.loc[normal_df.split=='normal_calibration','episode_detected'].mean():.2f}%",
          flush=True)
    print("Held-out normal validation false-episode rate: "
          f"{100*normal_df.loc[normal_df.split=='normal_validation','episode_detected'].mean():.2f}%",
          flush=True)

    dataset = ds.dataset(str(FAULT_PATH), format="parquet")
    needed = ["faultNumber", "simulationRun", "sample", *SENSORS]
    if not set(needed).issubset(dataset.schema.names):
        raise ValueError("Fault cache columns are incompatible")
    all_rows = []
    for fault in FAULTS:
        table = dataset.to_table(
            columns=needed, filter=ds.field("faultNumber") == fault
        )
        frame = table.to_pandas()
        del table
        if len(frame) != 500 * RUN_LENGTH:
            raise ValueError(
                f"Fault {fault}: got {len(frame):,} rows instead of {500 * RUN_LENGTH:,}"
            )
        for run, run_df in frame.groupby("simulationRun", sort=True):
            run = int(run)
            run_df = run_df.sort_values("sample")
            samples = run_df["sample"].to_numpy(dtype=np.int32)
            if not np.array_equal(samples, np.arange(1, RUN_LENGTH + 1)):
                raise ValueError(f"Invalid fault {fault}, run {run} sample grid")
            score = score_block(run_df[SENSORS].to_numpy(dtype=np.float32),
                                sensor_q99)
            onset_idx = first_episode(score, episode_threshold)
            onset = None if onset_idx is None else int(samples[onset_idx])
            early = onset is not None and onset < FAULT_INJECTION_SAMPLE
            post = onset is not None and onset >= FAULT_INJECTION_SAMPLE
            all_rows.append({
                "faultNumber": fault,
                "simulationRun": run,
                "split": "development" if run <= 350 else "internal_validation",
                "episode_detected": onset is not None,
                "episode_sample": onset,
                "early_alarm": bool(early),
                "postfault_first_alarm": bool(post),
                "detection_delay": (onset - FAULT_INJECTION_SAMPLE) if post else None,
                "max_score": float(score.max()),
            })
        del frame
        print(f"Evaluated training Fault {fault:02d}/15, 500 runs", flush=True)

    runs = pd.DataFrame(all_rows)
    if len(runs) != 7500:
        raise ValueError(f"Unexpected {len(runs)} evaluated fault runs")
    by_fault = summarize(runs)
    OUT.mkdir(parents=True, exist_ok=True)
    normal_df.to_csv(OUT / "normal_episode_evaluation_by_run.csv", index=False)
    runs.to_csv(OUT / "fault_training_episode_evaluation_by_run.csv", index=False)
    by_fault.to_csv(OUT / "fault_training_episode_metrics.csv", index=False)
    pd.DataFrame({
        "sensor": SENSORS, "q99_training_calibration": sensor_q99
    }).to_csv(OUT / "sensor_thresholds.csv", index=False)
    metrics = {
        "dataset": "TEP TRAINING CACHES ONLY",
        "sensor_quantile": SENSOR_QUANTILE,
        "episode_run_max_quantile": RUN_MAX_QUANTILE,
        "rolling_samples": TRAILING_WINDOW,
        "sustained_samples": SUSTAINED_SAMPLES,
        "process_alarm_threshold": episode_threshold,
        "normal_calibration_runs": "1..350",
        "normal_validation_runs": "351..500",
        "fault_development_runs": "1..350",
        "fault_internal_validation_runs": "351..500",
        "fault_injection_sample_for_evaluation_only": FAULT_INJECTION_SAMPLE,
        "training_only_method": True,
        "testing_data_consulted": False,
        "note": "No causal root-origin attribution attempted yet.",
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metrics, indent=2))
    print("\n" + "=" * 88,flush=True)
    print("EPISODE DETECTOR — INTERNAL VALIDATION ON FAULT-TRAINING RUNS 351..500",
          flush=True)
    print("=" * 88,flush=True)
    validation = by_fault[by_fault.split == "internal_validation"]
    print(validation.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
          flush=True)
    print("\nAggregate internal validation:",flush=True)
    part = runs[runs.split == "internal_validation"]
    print(f"  Fault runs: {len(part):,}",flush=True)
    print(f"  First alarm before injection: {100*part.early_alarm.mean():.2f}%",
          flush=True)
    print(f"  First alarm on/after injection: {100*part.postfault_first_alarm.mean():.2f}%",
          flush=True)
    print(f"  No episode: {100*(~part.episode_detected).mean():.2f}%",
          flush=True)
    print(f"Results: {OUT}",flush=True)
    print("This experiment does NOT evaluate or change the original V2.5 "
          "fault-origin rankings.",flush=True)


if __name__ == "__main__":
    main()
