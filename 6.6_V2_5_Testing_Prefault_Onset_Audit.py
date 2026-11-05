"""V2.5.10: independent TESTING onset-timing audit (READ-ONLY).

Uses already saved frozen V2.5 testing rankings. No recalculation of
attribution, no threshold changes, no new candidate ranking. Checks
whether the winning and other early sensors crossed thresholds before
the known benchmark intervention (testing sample 161).

This is DIAGNOSTIC ONLY, never a revised V2.5 headline score.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE=Path("/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution")
IN_DIR=BASE/"V2_5_Leading_Edge_Testing"
OUT=IN_DIR/"onset_timing_audit"
FAULT_START=161
CAUSES={
    1:"xmeas_4",2:"xmeas_4",3:"xmeas_23",4:"xmv_10",
    5:"xmeas_22",6:"xmv_3",7:"xmeas_4",8:"xmeas_4",
    9:"xmeas_23",10:"xmeas_4",11:"xmv_10",12:"xmv_11",
    13:"xmeas_9",14:"xmv_10",15:"xmv_11",
}
rows=[]
for fault,cause in CAUSES.items():
    p=IN_DIR/f"v2_5_testing_rankings_fault_{fault:02d}.parquet"
    df=pd.read_parquet(p)
    if len(df)!=500*52 or df.simulationRun.nunique()!=500:
        raise ValueError(f"Unexpected rankings for fault {fault}")
    for run,block in df.groupby("simulationRun",sort=True):
        ranked=block.sort_values("v2_5_rank")
        if not np.array_equal(ranked.v2_5_rank.to_numpy(),np.arange(1,53)):
            raise ValueError(f"Invalid rank sequence: fault={fault}, run={run}")
        winning=ranked.iloc[0]
        origin=ranked.loc[ranked.sensor==cause]
        if len(origin)!=1:
            raise ValueError(f"Missing cause: fault={fault}, run={run}")
        origin=origin.iloc[0]
        early=(ranked.onset<FAULT_START)
        n_early=int(early.sum())
        n_early_ahead=int(((ranked.onset<FAULT_START)&
                           (ranked.v2_5_rank<origin.v2_5_rank)).sum())
        rows.append({
            "faultNumber":fault,"simulationRun":int(run),"true_sensor":cause,
            "origin_rank":int(origin.v2_5_rank),
            "winner_sensor":winning.sensor,
            "winner_onset":float(winning.onset),
            "winner_before_fault":bool(winning.onset<FAULT_START),
            "origin_onset":float(origin.onset),
            "origin_before_fault":bool(origin.onset<FAULT_START),
            "origin_detected":bool(np.isfinite(origin.onset)),
            "n_sensors_abnormal_before_fault":n_early,
            "n_prefault_sensors_ranked_ahead_of_origin":n_early_ahead,
            "top3":bool(origin.v2_5_rank<=3),
            "top5":bool(origin.v2_5_rank<=5),
        })
    print(f"Audited fault {fault:02d} / 15",flush=True)
runs=pd.DataFrame(rows)
assert len(runs)==7500
summary=[]
for f,df in runs.groupby("faultNumber"):
    summary.append({
        "fault":int(f),"origin":CAUSES[int(f)],"runs":len(df),
        "top3_%":round(100*df.top3.mean(),2),
        "winner_prefault_%":round(100*df.winner_before_fault.mean(),2),
        "origin_prefault_%":round(100*df.origin_before_fault.mean(),2),
        "origin_detected_%":round(100*df.origin_detected.mean(),2),
        "median_prefault_sensor_count":float(df.n_sensors_abnormal_before_fault.median()),
        "median_prefault_sensors_ahead":float(
            df.n_prefault_sensors_ranked_ahead_of_origin.median()),
        "median_origin_onset_when_detected":float(
            df.loc[df.origin_detected,"origin_onset"].median())
            if df.origin_detected.any() else None,
    })
per_fault=pd.DataFrame(summary)
OUT.mkdir(parents=True,exist_ok=True)
runs.to_csv(OUT/"v2_5_testing_onset_timing_by_run.csv",index=False)
per_fault.to_csv(OUT/"v2_5_testing_onset_timing_per_fault.csv",index=False)
stats={
    "fault_start_sample":FAULT_START,
    "n_runs":len(runs),
    "winner_prefault_percentage":100*float(runs.winner_before_fault.mean()),
    "origin_prefault_percentage":100*float(runs.origin_before_fault.mean()),
    "origin_detected_percentage":100*float(runs.origin_detected.mean()),
    "median_prefault_sensors_per_run":float(
        runs.n_sensors_abnormal_before_fault.median()),
    "median_prefault_sensors_ranked_ahead_of_origin":float(
        runs.n_prefault_sensors_ranked_ahead_of_origin.median()),
    "interpretation":"Descriptive audit of frozen V2.5 rankings; not an alternate accuracy score.",
}
(OUT/"audit_metadata.json").write_text(json.dumps(stats,indent=2))
print("\n" + "="*85)
print("FROZEN V2.5 TESTING — PRE-FAULT ONSET AUDIT")
print("="*85)
for k,v in stats.items():
    print(f"{k}: {v}")
print("\nPER FAULT:")
print(per_fault.to_string(index=False))
print(f"\nSaved read-only audit results: {OUT}",flush=True)
