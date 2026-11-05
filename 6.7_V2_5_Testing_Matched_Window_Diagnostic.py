"""V2.5.11: FIXED diagnostic of testing-window length, not a new method.

Compare frozen full-960 V2.5 results (already saved) with a PREDECLARED,
single 500-sample testing window 141..640. Fault injection is at 161, so
this contains exactly 20 pre-injection samples, matching the 500-sample
fault-training run's intervention at sample 21.

This uses known injection time ONLY for a post-hoc controlled diagnostic.
It is not label-free deployment, not frozen V2.5 headline performance,
and MUST NOT replace the independent full-sequence score of 25.37% Top3.
No parameter search/tuning, no model fitting, no changed q99 or K=3.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parent
ORIGINAL=REPO/"5.1_V2_5_Leading_Edge_Attribution.py.py"
BASE=Path("/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution")
CACHED=BASE/"TEP_testing_error_cache_frozen_v2_5"
RANKED=BASE/"V2_5_Leading_Edge_Testing"
OUT=RANKED/"matched_500_sample_diagnostic"
START=141
END=640
INJECTION=161
SENSORS=[f"xmeas_{i}" for i in range(1,42)]+[f"xmv_{i}" for i in range(1,12)]
CAUSES={
  1:"xmeas_4",2:"xmeas_4",3:"xmeas_23",4:"xmv_10",5:"xmeas_22",
  6:"xmv_3",7:"xmeas_4",8:"xmeas_4",9:"xmeas_23",10:"xmeas_4",
  11:"xmv_10",12:"xmv_11",13:"xmeas_9",14:"xmv_10",15:"xmv_11",
}

def frozen_functions():
    source=ast.parse(ORIGINAL.read_text(), filename=str(ORIGINAL))
    names={"calculate_thresholds","attribute_run","find_sustained_onset","calculate_metrics"}
    picked=[n for n in source.body
            if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in picked}!=names:
        raise RuntimeError("Original frozen functions missing")
    module=ast.Module(body=picked,type_ignores=[])
    ast.fix_missing_locations(module)
    scope={"pd":pd,"np":np}
    exec(compile(module,str(ORIGINAL),"exec"),scope)
    return scope

def main():
    assert END-START+1==500 and INJECTION-START+1==21
    fns=frozen_functions()
    normal=pd.read_parquet(BASE/"TEP_cache/normal_error_df.parquet",columns=SENSORS)
    thresholds=fns["calculate_thresholds"](normal,SENSORS,0.99)
    del normal
    results=[]
    winner_rows=[]
    for fault,cause in CAUSES.items():
        cache=CACHED/f"fault_{fault:02d}_errors.parquet"
        original_rank=RANKED/f"v2_5_testing_rankings_fault_{fault:02d}.parquet"
        if not cache.is_file() or not original_rank.is_file():
            raise FileNotFoundError(f"Missing already-frozen testing outputs: fault {fault}")
        raw=pd.read_parquet(cache)
        old=pd.read_parquet(original_rank)
        if len(raw)!=480000 or len(old)!=26000:
            raise ValueError(f"Unexpected run/ranking count for {fault}")
        matched=raw.loc[raw["sample"].between(START,END)]
        if len(matched)!=250000:
            raise ValueError(f"Unexpected matched window length for {fault}")
        old_truth=old.loc[old.sensor==cause,["simulationRun","v2_5_rank"]]
        old_truth=old_truth.set_index("simulationRun")["v2_5_rank"]
        if len(old_truth)!=500 or old_truth.index.nunique()!=500:
            raise ValueError(f"Bad original rank grid, fault {fault}")
        early_winners=old.loc[old.v2_5_rank==1,["simulationRun","sensor","onset"]]
        for v in early_winners.itertuples(index=False):
            winner_rows.append({
                "faultNumber":fault,"simulationRun":int(v.simulationRun),
                "full_960_winner":v.sensor,
                "full_960_winner_onset":float(v.onset),
                "full_960_winner_prefault":bool(v.onset<INJECTION)
            })
        for run,run_df in matched.groupby("simulationRun",sort=True):
            if len(run_df)!=500 or run_df["sample"].nunique()!=500:
                raise ValueError(f"Fault {fault} run {run}: invalid window")
            # EXACT frozen function. Note: original sample numbers are kept.
            ranking=fns["attribute_run"](
                run_df=run_df,sensors=SENSORS,thresholds=thresholds,sustain_k=3
            )
            truth=ranking.loc[ranking.sensor==cause]
            if len(truth)!=1:
                raise RuntimeError(f"True origin absent, fault {fault}")
            origin_rank=int(truth.iloc[0]["v2_5_rank"])
            results.append({
                "faultNumber":fault,"simulationRun":int(run),
                "true_sensor":cause,
                "full_960_rank":int(old_truth.loc[run]),
                "matched_500_rank":origin_rank,
                "matched_500_origin_onset":float(truth.iloc[0]["onset"]),
                "matched_500_winner":str(ranking.iloc[0]["sensor"]),
                "matched_500_winner_onset":float(ranking.iloc[0]["onset"])
            })
        print(f"Compared fault {fault:02d}/15: 500 runs",flush=True)
    df=pd.DataFrame(results)
    winners=pd.DataFrame(winner_rows)
    assert len(df)==len(winners)==7500
    per=[]
    for fault,part in df.groupby("faultNumber"):
        per.append({
            "fault":int(fault),"cause":CAUSES[int(fault)],
            "full_top1_%":100*float((part.full_960_rank<=1).mean()),
            "matched_top1_%":100*float((part.matched_500_rank<=1).mean()),
            "full_top3_%":100*float((part.full_960_rank<=3).mean()),
            "matched_top3_%":100*float((part.matched_500_rank<=3).mean()),
            "full_top5_%":100*float((part.full_960_rank<=5).mean()),
            "matched_top5_%":100*float((part.matched_500_rank<=5).mean()),
            "median_rank_full":float(part.full_960_rank.median()),
            "median_rank_matched":float(part.matched_500_rank.median()),
        })
    summary=pd.DataFrame(per)
    overall={
        "full_960_top1_%":100*float((df.full_960_rank<=1).mean()),
        "matched_500_top1_%":100*float((df.matched_500_rank<=1).mean()),
        "full_960_top3_%":100*float((df.full_960_rank<=3).mean()),
        "matched_500_top3_%":100*float((df.matched_500_rank<=3).mean()),
        "full_960_top5_%":100*float((df.full_960_rank<=5).mean()),
        "matched_500_top5_%":100*float((df.matched_500_rank<=5).mean()),
        "full_960_median_rank":float(df.full_960_rank.median()),
        "matched_500_median_rank":float(df.matched_500_rank.median()),
        "full_960_prefault_winner_%":100*float(winners.full_960_winner_prefault.mean()),
    }
    OUT.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUT/"matched_500_run_comparison.csv",index=False)
    summary.to_csv(OUT/"matched_500_per_fault.csv",index=False)
    winners.to_csv(OUT/"original_960_winners_by_run.csv",index=False)
    top_winners=(winners.loc[winners.full_960_winner_prefault]
                 .groupby("full_960_winner").size()
                 .sort_values(ascending=False).head(15))
    (OUT/"matched_500_metadata.json").write_text(json.dumps({
        "purpose":"post-hoc matched-window sensitivity diagnostic only",
        "fixed_window_start":START,"fixed_window_end":END,
        "testing_fault_injection_sample":INJECTION,
        "method":"unchanged V2.5 function applied to predefined subset",
        "reference":"full 960 sample independent V2.5 score remains primary",
        "no_scoring_parameter_tuned":True,
        "overall":overall,
        "top_prefault_winners":top_winners.to_dict(),
    },indent=2))
    print("\n"+"="*82,flush=True)
    print("MATCHED 500-SAMPLE TIMING SENSITIVITY — NOT A DEPLOYABLE SCORE",flush=True)
    print("="*82,flush=True)
    for k,v in overall.items():
        print(f"{k}: {v:.2f}",flush=True)
    print("\nPer fault:\n"+summary.to_string(index=False),flush=True)
    print("\nMost common pre-injection winning sensors:\n"
          +top_winners.to_string(),flush=True)
    print(f"\nResults: {OUT}",flush=True)
    print("Keep full-sequence frozen V2.5 Top3=25.37% as primary test result.",
          flush=True)

if __name__=="__main__":
    main()
