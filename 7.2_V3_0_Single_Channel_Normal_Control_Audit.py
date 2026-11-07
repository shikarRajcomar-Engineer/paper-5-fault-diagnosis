"""V3.0 Experiment 1B — normal-control specificity of single-channel sustained alarms.

READ-ONLY diagnostic on TRAINING reconstruction-error caches.
Uses precomputed Experiment 1 q99 thresholds; NO fitting, model changes,
fault origins, V2.5 edits, or access to separate TEP TESTING data.

Compare healthy and faulty validation runs 351..500 on identical relative
observation periods:
    sample 21..500 (480 samples)
    sample 21..80  (60 samples)
    sample 21..140 (120 samples)
The intervention at sample 21 is used ONLY for post-hoc diagnostic alignment,
not as an inferred onset. Results are NOT deployable fault-detection scores.

Individual detector: first 3 consecutive absolute AE errors above its saved
normal-training q99 threshold. A run is positive if ANY of 52 channels has
one such onset in the observation interval.

Additional descriptive metrics: numbers of sustained channels, first alarm
confirmation sample (start + 2), and distributions across runs.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BASE=Path("/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution")
NORMAL=BASE/"TEP_cache/normal_error_df.parquet"
FAULT=BASE/"TEP_cache/fault_error_df.parquet"
THRESHOLDS=BASE/"V3_0_Development/Episode_Experiment_1/sensor_thresholds.csv"
OUT=BASE/"V3_0_Development/Episode_Experiment_1B"
SENSORS=[f"xmeas_{i}" for i in range(1,42)]+[f"xmv_{i}" for i in range(1,12)]
FIRST=21
HORIZONS=[60,120,480]
K=3
N_SAMPLES=500
VAL_RUNS=range(351,501)
FAULTS=range(1,16)


def audit_one(values, thresholds, length):
    """Returns count of channels with >=K consecutive strict exceedances.

    Does not use true fault identity, true origin, or fault timestamp except
    that the caller provides a fixed diagnostic window.
    """
    window=values[FIRST-1:FIRST-1+length]
    if window.shape != (length,52):
        raise ValueError("Unexpected input window")
    above=window>thresholds[None,:]
    sustained=above[:-2] & above[1:-1] & above[2:]
    onset_by_channel=np.any(sustained,axis=0)
    if sustained.any():
        earliest_start=int(np.where(np.any(sustained,axis=1))[0][0])
        # Observable alarm is at final sample of qualifying K-run.
        confirmation=FIRST+earliest_start+K-1
    else:
        confirmation=None
    return bool(onset_by_channel.any()),int(onset_by_channel.sum()),confirmation


def main():
    print("="*88,flush=True)
    print("V3.0 EXPERIMENT 1B — INDIVIDUAL ALARMS VS HEALTHY CONTROLS",flush=True)
    print("="*88,flush=True)
    for p in (NORMAL,FAULT,THRESHOLDS):
        if not p.is_file(): raise FileNotFoundError(p)
    threshold_df=pd.read_csv(THRESHOLDS)
    if threshold_df.sensor.tolist()!=SENSORS:
        raise ValueError("Saved thresholds order mismatch")
    thresholds=threshold_df.q99_training_calibration.to_numpy(np.float64)
    if not np.isfinite(thresholds).all(): raise ValueError("Invalid saved q99")
    print("Reading only training caches; comparing the same 480-sample "
          "window in healthy and faulty runs.",flush=True)
    normal=pd.read_parquet(NORMAL)
    if len(normal)!=250000 or not set(["simulationRun",*SENSORS]).issubset(normal.columns):
        raise ValueError("Normal training cache shape/schema mismatch")
    rows=[]
    for r in VAL_RUNS:
        group=normal.loc[normal.simulationRun==r,SENSORS]
        if len(group)!=N_SAMPLES: raise ValueError(f"Normal run {r} wrong length")
        arr=group.to_numpy(np.float32)
        if not np.isfinite(arr).all(): raise ValueError("Nonfinite normal errors")
        for length in HORIZONS:
            has,n,confirm=audit_one(arr,thresholds,length)
            rows.append(dict(data_type="healthy",faultNumber=0,
                 simulationRun=r,horizon_samples=length,any_sustained=has,
                 n_sustained_sensors=n,first_confirmation_sample=confirm))
    del normal
    print("Audited 150 healthy internal-validation training runs",flush=True)
    dataset=ds.dataset(str(FAULT),format="parquet")
    needed=["faultNumber","simulationRun","sample",*SENSORS]
    if not set(needed).issubset(dataset.schema.names):
        raise ValueError("Fault training cache schema mismatch")
    for fault in FAULTS:
        table=dataset.to_table(columns=needed,filter=(
            (ds.field("faultNumber")==fault)
            &(ds.field("simulationRun")>=351)
            &(ds.field("simulationRun")<=500)))
        frame=table.to_pandas()
        del table
        if len(frame)!=150*N_SAMPLES:
            raise ValueError(f"Fault {fault}: wrong row count")
        for r,group in frame.groupby("simulationRun",sort=True):
            group=group.sort_values("sample")
            if not np.array_equal(group["sample"].to_numpy(),np.arange(1,N_SAMPLES+1)):
                raise ValueError(f"Fault {fault}, run {r}: sample order invalid")
            arr=group[SENSORS].to_numpy(np.float32)
            if not np.isfinite(arr).all(): raise ValueError("Nonfinite fault errors")
            for length in HORIZONS:
                has,n,confirm=audit_one(arr,thresholds,length)
                rows.append(dict(data_type="faulty",faultNumber=fault,
                     simulationRun=int(r),horizon_samples=length,
                     any_sustained=has,n_sustained_sensors=n,
                     first_confirmation_sample=confirm))
        print(f"Audited Fault {fault:02d}/15 (150 validation runs)",flush=True)
        del frame
    result=pd.DataFrame(rows)
    expected=(150+15*150)*len(HORIZONS)
    if len(result)!=expected: raise RuntimeError("Unexpected output row count")
    summaries=[]
    for (typ,fault,length),group in result.groupby(
        ["data_type","faultNumber","horizon_samples"],sort=True):
        successes=group.loc[group.any_sustained]
        summaries.append(dict(data_type=typ,faultNumber=int(fault),
           horizon_samples=int(length),runs=len(group),
           at_least_one_sustained_pct=100*group.any_sustained.mean(),
           median_n_sustained_sensors=float(group.n_sustained_sensors.median()),
           median_first_confirmation_sample=(
               float(successes.first_confirmation_sample.median())
               if len(successes) else None)))
    summary=pd.DataFrame(summaries)
    OUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(OUT/"validation_single_channel_alarms_by_run.csv",index=False)
    summary.to_csv(OUT/"validation_single_channel_alarms_by_fault_and_horizon.csv",
                   index=False)
    (OUT/"experiment_metadata.json").write_text(json.dumps(dict(
        dataset="TEP TRAINING ONLY",
        purpose="diagnostic comparison with matched observation horizon",
        normal_validation_runs="351..500",
        faulty_validation_runs="351..500",
        sensor_q99_source=str(THRESHOLDS),
        k_consecutive=K,
        horizons=HORIZONS,
        diagnostic_window_start=FIRST,
        episode_confirmation_is_last_exceedance=True,
        no_test_data=True,
        no_parameters_fitted=True,
        caveat=("Uses known injection sample 21 to align retrospective windows. "
                "A sustained normal error excursion is not proof of fault detection."),
    ),indent=2))
    print("\n"+"="*88,flush=True)
    print("HEALTHY CONTROLS (FALSE ALARM RATES FOR ANY INDIVIDUAL SENSOR)",flush=True)
    print("="*88,flush=True)
    print(summary[summary.data_type=="healthy"].to_string(index=False),flush=True)
    print("\nFAULTY RUNS, BY FAULT AND OBSERVATION HORIZON",flush=True)
    print(summary[summary.data_type=="faulty"].to_string(index=False),flush=True)
    for length in HORIZONS:
        h=result[(result.data_type=="healthy")&(result.horizon_samples==length)]
        f=result[(result.data_type=="faulty")&(result.horizon_samples==length)]
        print(f"\nHorizon {length} samples: healthy any-sensor alarms "
              f"{100*h.any_sustained.mean():.2f}% vs fault any-sensor alarms "
              f"{100*f.any_sustained.mean():.2f}%",flush=True)
    print("\nThese are diagnostic alarm rates, NOT deployable accuracy or V3.0 attribution.",flush=True)
    print(f"Results saved: {OUT}",flush=True)


if __name__=="__main__":
    main()
