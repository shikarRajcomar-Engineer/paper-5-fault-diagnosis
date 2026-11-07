"""V3.0 Experiment 2 — localized severity/persistence alarm.

Separate prototype. Reads ONLY original TRAINING AE-error caches; V2.5 and
earlier V3.0 experiments are unchanged. No origin labels or true injection
times enter alarm calculation.

PREDECLARED detector (not a parameter search):
  1. Per-sensor q99 error thresholds from healthy runs 1..350.
  2. Per sample, positive excess = log1p(max(error/q99 - 1, 0)).
  3. Trailing 10-sample mean of positive excess for EACH sensor separately.
  4. Process-level localized score is the MAX across 52 channels.
  5. Alarm level = 95th percentile of per-run MAX localized score
     across 350 healthy calibration runs (full 500 samples).
  6. An episode is confirmed after 3 consecutive STRICT exceedances.
     Report the actual confirmation sample, NOT an unobservable start.

Runs 351..500 are a *within-training* check for both healthy and faulty data.
The AE weights were themselves trained on normal training data, so this is
not external model generalization. Do not consult TEP testing results for
parameter selection or present these outcomes as unseen test performance.

Run: python -u 7.3_V3_0_Localized_Persistence_Detector.py
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
OUT = BASE / "V3_0_Development/Localized_Persistence_Experiment_2"
SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
CALIBRATION = range(1, 351)
VALIDATION = range(351, 501)
FAULTS = range(1, 16)
SAMPLE_COUNT = 500

SENSOR_Q = 0.99
RUNMAX_Q = 0.95
WINDOW = 10
PERSIST = 3
INJECTION = 21  # ONLY for evaluation metrics, never passed into detector


def rolling_mean_2d(x: np.ndarray, width: int) -> np.ndarray:
    """Causal, full-history trailing mean with partial windows initially."""
    c = np.vstack((np.zeros((1, x.shape[1]), dtype=np.float64),
                   np.cumsum(x, axis=0, dtype=np.float64)))
    ends = np.arange(1, len(x) + 1)
    starts = np.maximum(ends - width, 0)
    return (c[ends] - c[starts]) / (ends - starts)[:, None]


def scores(errors: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """No time, fault labels, injection point or true origin used."""
    if errors.shape != (SAMPLE_COUNT, len(SENSORS)):
        raise ValueError(f"Expected 500 x 52 errors, got {errors.shape}")
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Invalid nonnegative reconstruction errors")
    if np.any(q99 <= 0) or not np.isfinite(q99).all():
        raise ValueError("Invalid normal-training q99 threshold")
    excess = np.log1p(np.maximum(errors / q99[None, :] - 1.0, 0.0))
    return np.max(rolling_mean_2d(excess, WINDOW), axis=1)


def confirmed_alarm(scores_1d: np.ndarray, level: float) -> int | None:
    """Return 1-indexed *confirmation* sample after PERSIST exceedances."""
    above = scores_1d > level
    if len(above) < PERSIST:
        return None
    starts = np.flatnonzero(np.convolve(
        above.astype(np.int8),
        np.ones(PERSIST, dtype=np.int8),
        mode="valid"
    ) == PERSIST)
    return int(starts[0] + PERSIST) if len(starts) else None


def normal_run_arrays() -> dict[int, np.ndarray]:
    frame = pd.read_parquet(NORMAL_PATH)
    if len(frame) != SAMPLE_COUNT * 500:
        raise ValueError("Unexpected normal training cache length")
    if not set(SENSORS + ["simulationRun"]).issubset(frame):
        raise ValueError("Normal cache lacks sensors or simulationRun")
    grouped = {}
    for r in range(1, 501):
        group = frame.loc[frame.simulationRun == r, SENSORS]
        if len(group) != SAMPLE_COUNT:
            raise ValueError(f"Normal run {r}: not exactly 500 rows")
        grouped[r] = group.to_numpy(np.float32)
    return grouped


def main() -> None:
    print("=" * 89, flush=True)
    print("V3.0 EXPERIMENT 2 — LOCALIZED SEVERITY + PERSISTENCE EPISODE ALARM",
          flush=True)
    print("=" * 89, flush=True)
    for p in [NORMAL_PATH, FAULT_PATH]:
        if not p.is_file():
            raise FileNotFoundError(p)
    normal = normal_run_arrays()
    calibration_errors = np.vstack([normal[r] for r in CALIBRATION])
    thresholds = np.quantile(calibration_errors, SENSOR_Q, axis=0)
    del calibration_errors
    run_peaks = np.array([
        scores(normal[r], thresholds).max() for r in CALIBRATION
    ], dtype=np.float64)
    alarm_level = float(np.quantile(run_peaks, RUNMAX_Q))
    print(f"Sensor baseline: normal runs 1..350, q{SENSOR_Q}", flush=True)
    print(f"Localized score: max over sensors of {WINDOW}-sample trailing "
          "mean positive log-excess", flush=True)
    print(f"Alarm level: q{RUNMAX_Q} of NORMAL per-run maximum={alarm_level:.6g}",
          flush=True)
    print(f"Consecutive exceedances needed: {PERSIST}", flush=True)
    print("Reported timestamps are CONFIRMATION samples.", flush=True)

    normal_rows = []
    for r in range(1, 501):
        trace = scores(normal[r], thresholds)
        alarm = confirmed_alarm(trace, alarm_level)
        normal_rows.append({
            "simulationRun": r,
            "split": "calibration" if r <= 350 else "internal_validation",
            "alarm_confirmed": alarm is not None,
            "alarm_confirmation_sample": alarm,
            "max_score": float(trace.max()),
        })
    del normal
    normal_results = pd.DataFrame(normal_rows)
    for split in ["calibration", "internal_validation"]:
        sub = normal_results[normal_results.split == split]
        print(f"Healthy {split} any-alarm rate: "
              f"{100*sub.alarm_confirmed.mean():.2f}% ({len(sub)} runs)",
              flush=True)

    dataset = ds.dataset(str(FAULT_PATH), format="parquet")
    columns = ["faultNumber", "simulationRun", "sample", *SENSORS]
    if not set(columns).issubset(dataset.schema.names):
        raise ValueError("Fault cache missing required columns")
    output_rows = []
    for fault in FAULTS:
        data = dataset.to_table(
            columns=columns, filter=ds.field("faultNumber") == fault
        ).to_pandas()
        if len(data) != 500 * SAMPLE_COUNT:
            raise ValueError(f"Fault {fault}: unexpected {len(data)} rows")
        for run, group in data.groupby("simulationRun", sort=True):
            run = int(run)
            group = group.sort_values("sample")
            if not np.array_equal(
                group["sample"].to_numpy(dtype=np.int32),
                np.arange(1, SAMPLE_COUNT + 1)
            ):
                raise ValueError(f"Fault {fault} run {run}: bad sample sequence")
            trace = scores(group[SENSORS].to_numpy(dtype=np.float32), thresholds)
            alarm = confirmed_alarm(trace, alarm_level)
            # The benchmark injection time enters ONLY here, after detection.
            early = alarm is not None and alarm < INJECTION
            post = alarm is not None and alarm >= INJECTION
            output_rows.append({
                "faultNumber": fault,
                "simulationRun": run,
                "split": "development" if run <= 350 else "internal_validation",
                "alarm_confirmed": alarm is not None,
                "confirmation_sample": alarm,
                "early_alarm": bool(early),
                "postfault_first_alarm": bool(post),
                "postfault_detection_delay": (
                    int(alarm - INJECTION) if post else None
                ),
                "max_score": float(trace.max()),
            })
        print(f"Evaluated fault {fault:02d}/15: 500 training runs", flush=True)
        del data

    faults = pd.DataFrame(output_rows)
    assert len(faults) == 7500
    by_fault = []
    for (split, fault), block in faults.groupby(
        ["split", "faultNumber"], sort=True
    ):
        delays = block.loc[block.postfault_first_alarm,
                           "postfault_detection_delay"]
        by_fault.append({
            "split": split, "faultNumber": int(fault), "runs": len(block),
            "any_alarm_%": 100 * block.alarm_confirmed.mean(),
            "early_alarm_%": 100 * block.early_alarm.mean(),
            "postfault_first_alarm_%": 100 * block.postfault_first_alarm.mean(),
            "median_postfault_confirmation_delay": (
                float(delays.median()) if len(delays) else None
            ),
        })
    summary = pd.DataFrame(by_fault)
    OUT.mkdir(parents=True, exist_ok=True)
    normal_results.to_csv(OUT / "normal_alarm_by_run.csv", index=False)
    faults.to_csv(OUT / "fault_alarm_by_run.csv", index=False)
    summary.to_csv(OUT / "fault_alarm_by_fault.csv", index=False)
    pd.DataFrame({
        "sensor": SENSORS,
        "q99_normal_training_runs_1_to_350": thresholds,
    }).to_csv(OUT / "sensor_q99.csv", index=False)
    metadata = {
        "data": "TEP TRAINING reconstruction error caches only",
        "normal_calibration_runs": "1..350",
        "normal_check_runs": "351..500",
        "fault_development_runs": "1..350",
        "fault_internal_validation_runs": "351..500",
        "sensor_quantile": SENSOR_Q,
        "normal_run_max_quantile": RUNMAX_Q,
        "trailing_window": WINDOW,
        "sustained_exceedances": PERSIST,
        "alarm_level": alarm_level,
        "reported_time": "first confirmed sample, NOT start of exceedance",
        "injection_sample_for_metrics_only": INJECTION,
        "no_external_test_data": True,
        "no_origin_labels": True,
        "caveat": "This is a predeclared prototype, not an independently validated FDD model.",
    }
    (OUT / "method_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\n" + "=" * 89, flush=True)
    print("V3.0 EXPERIMENT 2 — INTERNAL VALIDATION (FAULT TRAINING RUNS 351..500)",
          flush=True)
    print("=" * 89, flush=True)
    check = summary[summary.split == "internal_validation"]
    print(check.to_string(index=False, float_format=lambda x: f"{x:.2f}"),
          flush=True)
    validation = faults[faults.split == "internal_validation"]
    print("\nAggregate internal fault validation:", flush=True)
    print(f"  Fault runs: {len(validation)}", flush=True)
    print(f"  Alarm before injection: {100*validation.early_alarm.mean():.2f}%",
          flush=True)
    print(f"  First confirmed alarm on/after injection: "
          f"{100*validation.postfault_first_alarm.mean():.2f}%", flush=True)
    print(f"  No alarm: {100*(~validation.alarm_confirmed).mean():.2f}%",
          flush=True)
    print(f"  Normal internal-validation false alarm rate: "
          f"{100*normal_results.loc[normal_results.split=='internal_validation','alarm_confirmed'].mean():.2f}%",
          flush=True)
    print(f"Results: {OUT}", flush=True)
    print("NOT yet origin attribution. No original testing data used.", flush=True)


if __name__ == "__main__":
    main()
