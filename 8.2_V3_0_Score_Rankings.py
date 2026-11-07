"""Post-hoc evaluation ONLY; ranker never receives labels.

Supports any sensor set and any dataset with a JSON {true_causes: {id: sensor}}
mapping. Requires complete per-run rankings; reports all-run Top-k AND
conditional metrics for runs with a confirmed process episode.

Usage:
  python -u 8.2_V3_0_Score_Rankings.py --split train
  python -u 8.2_V3_0_Score_Rankings.py --split test
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

BASE=Path("/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution")
DEFAULT=BASE/"V3_0_Development/Unsupervised_Episode_Attribution_V3_0_0"
MAPPING=Path(__file__).resolve().parent/"8.1_TEP_Origin_Labels_Evaluation_Only.json"


def pct(series):
    return 100.0*float(series.mean()) if len(series) else float("nan")


def metrics(frame):
    return {
        "runs": int(len(frame)),
        "coverage_%":pct(frame.episode_detected),
        "top1_all_%":pct(frame.true_rank<=1),
        "top3_all_%":pct(frame.true_rank<=3),
        "top5_all_%":pct(frame.true_rank<=5),
        "median_rank_all":float(frame.true_rank.median()) if len(frame) else None,
        "top1_confirmed_%":pct(frame.loc[frame.episode_detected,"true_rank"]<=1),
        "top3_confirmed_%":pct(frame.loc[frame.episode_detected,"true_rank"]<=3),
        "top5_confirmed_%":pct(frame.loc[frame.episode_detected,"true_rank"]<=5),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split", choices=["train","test"],required=True)
    p.add_argument("--base", type=Path,default=DEFAULT)
    p.add_argument("--labels", type=Path,default=MAPPING)
    args=p.parse_args()
    if not args.labels.exists():
        raise FileNotFoundError(args.labels)
    true_causes={int(k):v for k,v in json.loads(args.labels.read_text())["true_causes"].items()}
    out=args.base/args.split
    metadata=json.loads((out/"method_metadata.json").read_text())
    faults=metadata["processed_faults"]
    features=metadata["features"]
    rows=[]
    for fid in faults:
        if fid not in true_causes:
            print(f"Fault {fid}: no verified origin label; excluded from Top-k scoring",flush=True)
            continue
        true_sensor=true_causes[fid]
        if true_sensor not in features:
            raise ValueError(f"Fault {fid} origin '{true_sensor}' not among model features")
        file=out/f"rankings_fault_{fid:02d}.parquet"
        frame=pd.read_parquet(file)
        if frame.empty or frame["simulationRun"].isna().any():
            raise ValueError(f"Invalid ranking file: {file}")
        for run,g in frame.groupby("simulationRun",sort=True):
            if len(g)!=len(features) or set(g.sensor)!=set(features):
                raise ValueError(f"Run {run} fault {fid}: missing/duplicate sensors")
            if not np.array_equal(np.sort(g["rank"].to_numpy()),np.arange(1,len(features)+1)):
                raise ValueError(f"Run {run} fault {fid}: incomplete ranks")
            if g.episode_detected.nunique()!=1:
                raise ValueError("Inconsistent episode flag within run")
            hit=g.loc[g.sensor==true_sensor].iloc[0]
            rows.append({
                "faultNumber":fid, "simulationRun":int(run),
                "true_sensor":true_sensor, "true_rank":int(hit["rank"]),
                "episode_detected":bool(hit["episode_detected"]),
            })
        print(f"Scored Fault {fid:02d}",flush=True)
    if not rows:
        raise RuntimeError("No faults with verified causes to score")
    result=pd.DataFrame(rows)
    per=[{"fault":int(f), "true_origin":true_causes[int(f)],**metrics(g)}
         for f,g in result.groupby("faultNumber",sort=True)]
    overall=metrics(result)
    score_dir=out/"POST_HOC_EVALUATION"
    score_dir.mkdir(parents=True,exist_ok=True)
    result.to_csv(score_dir/"ranks_by_run.csv",index=False)
    pd.DataFrame(per).to_csv(score_dir/"scores_by_fault.csv",index=False)
    (score_dir/"overall_metrics.json").write_text(json.dumps({
        "split":args.split,
        "attribution_method":metadata["algorithm"],
        "labels_file":str(args.labels),
        "labels_used_only_after_ranking":True,
        "no_episode_all_run_ranks_are_provisional":True,
        "sample_size":len(result),
        "metrics":overall,
        "warning":"This is not a pristine external test if the split has been inspected previously."
    },indent=2))
    print("\n"+"="*82,flush=True)
    print(f"V3.0.0 {args.split.upper()} POST-HOC METRICS | {len(result):,} RUNS",flush=True)
    print("="*82,flush=True)
    for k,v in overall.items():
        print(f"{k}: {v}",flush=True)
    print("\nPer-fault:\n"+pd.DataFrame(per).to_string(index=False),flush=True)
    print(f"Evaluation saved in {score_dir}",flush=True)


if __name__=="__main__":
    main()
