"""V3.0 Experiment 1A — why does the collective detector miss some faults?

READ-ONLY diagnostic on existing TRAINING caches and Experiment 1 outputs.
Does not fit thresholds, change the detector, use external TEP testing,
or produce a new attribution score.

Measures, for internal validation runs 351..500:
- How many sensor errors exceed calibrated normal q99 after injection
- Whether ANY individual channel has a sustained (3 sample) exceedance
- How many simultaneously abnormal channels occur at the busiest sample
- Whether and when the process-wide detector finds an episode
- Whether the original first-episode sample is the START of a sustained
  sequence, two samples before the earliest implementable confirmation.

Injection sample 21 is used ONLY to segment the post-injection DIAGNOSTIC
interval and score observations, not for detection or threshold fitting.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BASE = Path("/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution")
PREVIOUS = BASE / "V3_0_Development/Episode_Experiment_1"
OUT = BASE / "V3_0_Development/Episode_Experiment_1A"
SOURCE = BASE / "TEP_cache/fault_error_df.parquet"
SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
VALID_RUNS = range(351, 501)
FAULTS = range(1, 16)
INJECTION = 21
SUSTAIN = 3


def first_sustained(flags, k=SUSTAIN):
    """First START index (zero-based) of k consecutive True entries."""
    if len(flags) < k:
        return None
    counts = np.convolve(flags.astype(np.int8), np.ones(k, dtype=np.int8), "valid")
    found = np.flatnonzero(counts == k)
    return int(found[0]) if len(found) else None


def main():
    for p in (SOURCE, PREVIOUS / "sensor_thresholds.csv",
              PREVIOUS / "fault_training_episode_evaluation_by_run.csv",
              PREVIOUS / "experiment_metadata.json"):
        if not p.is_file():
            raise FileNotFoundError(p)
    meta = json.loads((PREVIOUS / "experiment_metadata.json").read_text())
    if meta.get("training_only_method") is not True:
        raise ValueError("Experiment 1 metadata incompatible")
    threshold_df = pd.read_csv(PREVIOUS / "sensor_thresholds.csv")
    if threshold_df["sensor"].tolist() != SENSORS:
        raise ValueError("Sensor thresholds not in canonical order")
    threshold = threshold_df["q99_training_calibration"].to_numpy(dtype=np.float64)
    if not np.isfinite(threshold).all():
        raise ValueError("Invalid saved training thresholds")
    old = pd.read_csv(PREVIOUS / "fault_training_episode_evaluation_by_run.csv")
    old = old.loc[old["split"] == "internal_validation"].copy()
    if len(old) != 15 * 150:
        raise ValueError("Missing previous internal validation rows")
    original = old.set_index(["faultNumber", "simulationRun"])

    dataset = ds.dataset(str(SOURCE), format="parquet")
    requested = ["faultNumber", "simulationRun", "sample", *SENSORS]
    if not set(requested).issubset(dataset.schema.names):
        raise ValueError("Training error cache missing required columns")
    rows = []
    for fault in FAULTS:
        subset = dataset.to_table(
            columns=requested,
            filter=(ds.field("faultNumber") == fault) &
                   (ds.field("simulationRun") >= 351) &
                   (ds.field("simulationRun") <= 500)
        ).to_pandas()
        if len(subset) != 150 * 500:
            raise ValueError(f"Fault {fault}: unexpected rows {len(subset)}")
        for run, group in subset.groupby("simulationRun", sort=True):
            run = int(run)
            group = group.sort_values("sample")
            if not np.array_equal(group["sample"].to_numpy(), np.arange(1, 501)):
                raise ValueError(f"Fault {fault}, run {run}: invalid sample grid")
            err = group[SENSORS].to_numpy(dtype=np.float32)
            if not np.isfinite(err).all():
                raise ValueError("Nonfinite AE errors")
            active = err > threshold[None, :]
            after = active[INJECTION - 1:]
            counts = after.sum(axis=1)
            first_per_sensor = [first_sustained(after[:, i]) for i in range(52)]
            observable = [(i, onset) for i, onset in enumerate(first_per_sensor)
                          if onset is not None]
            earliest = min((onset for _, onset in observable), default=None)
            prior = original.loc[(fault, run)]
            original_start = prior["episode_sample"]
            found = bool(prior["episode_detected"])
            if found != bool(pd.notna(original_start)):
                raise ValueError("Previous episode flags inconsistent")
            confirmed = int(original_start) + (SUSTAIN - 1) if found else None
            rows.append({
                "faultNumber": fault,
                "simulationRun": run,
                "original_collective_episode_detected": found,
                "original_collective_episode_start": (int(original_start) if found else None),
                "collective_confirmation_sample": confirmed,
                "collective_confirmation_delay": (confirmed - INJECTION if found else None),
                "any_individual_sustained_postfault": bool(observable),
                "earliest_individual_start": (
                    INJECTION + earliest if earliest is not None else None
                ),
                "earliest_individual_confirmation": (
                    INJECTION + earliest + (SUSTAIN - 1)
                    if earliest is not None else None
                ),
                "n_individual_sensors_sustained_postfault": len(observable),
                "max_simultaneously_abnormal_postfault": int(counts.max()),
                "median_simultaneously_abnormal_postfault": float(np.median(counts)),
                "max_simultaneously_abnormal_first_60_postfault": int(counts[:60].max()),
                "max_simultaneously_abnormal_first_120_postfault": int(counts[:120].max()),
            })
        print(f"Audited fault {fault:02d}/15 (150 held-out training runs)", flush=True)
        del subset

    frame = pd.DataFrame(rows)
    if len(frame) != 2250:
        raise RuntimeError("Expected 2,250 runs")
    summaries = []
    for f, sub in frame.groupby("faultNumber"):
        missed = sub.loc[~sub.original_collective_episode_detected]
        summaries.append({
            "fault": int(f),
            "runs": int(len(sub)),
            "collective_detected_%": 100*float(sub.original_collective_episode_detected.mean()),
            "individual_sustained_postfault_%": 100*float(sub.any_individual_sustained_postfault.mean()),
            "individual_sustained_when_collective_missed_%": (
                100*float(missed.any_individual_sustained_postfault.mean())
                if len(missed) else None
            ),
            "median_n_sustained_sensors": float(sub.n_individual_sensors_sustained_postfault.median()),
            "median_peak_simultaneously_abnormal": float(sub.max_simultaneously_abnormal_postfault.median()),
            "median_early_60_peak_simultaneously_abnormal": float(
                sub.max_simultaneously_abnormal_first_60_postfault.median()
            ),
            "median_collective_confirmation_delay": (
                float(sub.collective_confirmation_delay.median())
                if sub.original_collective_episode_detected.any() else None
            ),
        })
    per_fault = pd.DataFrame(summaries)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "training_validation_evidence_by_run.csv", index=False)
    per_fault.to_csv(OUT / "training_validation_evidence_per_fault.csv", index=False)
    (OUT / "diagnostic_metadata.json").write_text(json.dumps({
        "source_experiment": str(PREVIOUS),
        "training_only": True,
        "run_split": "internal validation, simulationRun 351..500",
        "injection_sample_for_evaluation_only": INJECTION,
        "sustained_samples": SUSTAIN,
        "original_episode_timestamp_semantics": (
            "start of 3 consecutive exceedances; implementable "
            "confirmation is +2 samples"
        ),
        "purpose": "descriptive signal-availability audit, no change to V3.0 detector",
    }, indent=2))
    print("\n" + "=" * 100, flush=True)
    print("V3.0 DIAGNOSTIC: COLLECTIVE VS SINGLE-CHANNEL SIGNAL AVAILABILITY", flush=True)
    print("=" * 100, flush=True)
    print(per_fault.to_string(index=False,float_format=lambda x:f"{x:.2f}"),flush=True)
    print("\nAggregate:", flush=True)
    print(f"  Collective episode rate: {100*frame.original_collective_episode_detected.mean():.2f}%",flush=True)
    print(f"  At least 1 sensor has sustained post-fault exceedance: "
          f"{100*frame.any_individual_sustained_postfault.mean():.2f}%",flush=True)
    missed = frame.loc[~frame.original_collective_episode_detected]
    print(f"  Individual sustained abnormality among collective misses: "
          f"{100*missed.any_individual_sustained_postfault.mean():.2f}%",flush=True)
    print("  Original first-episode timestamp is the start of the 3-sample"
          " sequence; deployable confirmation is 2 samples later.", flush=True)
    print(f"Results saved: {OUT}",flush=True)


if __name__ == "__main__":
    main()
