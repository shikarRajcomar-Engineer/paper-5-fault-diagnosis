"""Independent TEP TESTING evaluation using the unchanged V2.5 functions.

Import only the frozen V2.5 FUNCTION DEFINITIONS via AST (not its top-level
training-cache execution or output writes). V2.5's source is not modified.
Normal q99 thresholds are calculated ONLY from original normal TRAINING
reconstruction errors. Known-origin labels enter ONLY after rankings saved.

Input:  6.4 full testing cache, fault_01_errors.parquet...fault_15_errors.parquet
Output: separate V2_5_Leading_Edge_Testing/ folder.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent
ORIGINAL = REPO / "5.1_V2_5_Leading_Edge_Attribution.py.py"
BASE = Path("/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution")
NORMAL_CACHE = BASE / "TEP_cache/normal_error_df.parquet"
TEST_CACHE = BASE / "TEP_testing_error_cache_frozen_v2_5"
OUT = BASE / "V2_5_Leading_Edge_Testing"

SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
TRUE_CAUSES = {
    1: "xmeas_4", 2: "xmeas_4", 3: "xmeas_23", 4: "xmv_10",
    5: "xmeas_22", 6: "xmv_3", 7: "xmeas_4", 8: "xmeas_4",
    9: "xmeas_23", 10: "xmeas_4", 11: "xmv_10", 12: "xmv_11",
    13: "xmeas_9", 14: "xmv_10", 15: "xmv_11",
}
FUNCTION_NAMES = (
    "calculate_thresholds",
    "find_sustained_onset",
    "attribute_run",
    "calculate_metrics",
)


def original_functions():
    source = ORIGINAL.read_text()
    tree = ast.parse(source, filename=str(ORIGINAL))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in FUNCTION_NAMES]
    if {n.name for n in selected} != set(FUNCTION_NAMES):
        raise RuntimeError("Frozen V2.5 function definitions not found unchanged")
    # Compile ORIGINAL AST nodes, with only original source code.
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "pd": pd}
    exec(compile(module, str(ORIGINAL), "exec"), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def validate_run_ids(df, fault):
    if len(df) != 500 * 960:
        raise ValueError(f"Fault {fault}: expected 480,000 rows, got {len(df)}")
    if list(df.columns) != ["faultNumber", "simulationRun", "sample", *SENSORS]:
        raise ValueError(f"Fault {fault}: unexpected columns/order")
    if not (df.faultNumber == fault).all():
        raise ValueError(f"Fault {fault}: wrong faultNumber in shard")
    run = df.simulationRun.to_numpy(dtype=np.int64)
    sample = df["sample"].to_numpy(dtype=np.int64)
    if np.any(run < 1) or np.any(run > 500) or np.any(sample < 1) or np.any(sample > 960):
        raise ValueError(f"Fault {fault}: invalid run/sample")
    linear = (run - 1) * 960 + (sample - 1)
    if not np.array_equal(np.sort(linear), np.arange(500 * 960)):
        raise ValueError(f"Fault {fault}: incomplete or duplicate run/sample grid")
    if not np.isfinite(df[SENSORS].to_numpy()).all():
        raise ValueError(f"Fault {fault}: nonfinite reconstruction errors")


def main():
    print("=" * 85, flush=True)
    print("V2.5 — INDEPENDENT TEP TESTING EVALUATION (FROZEN ALGORITHM)", flush=True)
    print("=" * 85, flush=True)
    funcs, algorithm_sha = original_functions()
    if not NORMAL_CACHE.is_file():
        raise FileNotFoundError(NORMAL_CACHE)
    for fault in sorted(TRUE_CAUSES):
        part = TEST_CACHE / f"fault_{fault:02d}_errors.parquet"
        if not part.is_file():
            raise FileNotFoundError(f"Missing testing cache: {part}")
        if pq.ParquetFile(part).metadata.num_rows != 480_000:
            raise ValueError(f"Wrong row count in {part}")
    cache_meta_path = TEST_CACHE / "cache_metadata.json"
    if not cache_meta_path.is_file():
        raise FileNotFoundError(cache_meta_path)
    cache_meta = json.loads(cache_meta_path.read_text())
    if cache_meta.get("runs_per_fault") != 500 or cache_meta.get("faults") != list(range(1,16)):
        raise ValueError("Testing cache metadata incompatible with independent evaluation")
    if cache_meta.get("error_metric") != "absolute error, scaled space":
        raise ValueError("Testing reconstruction error metric differs from training")
    print(f"Original V2.5 SHA256: {algorithm_sha}", flush=True)
    print("Threshold reference: normal TRAINING reconstruction-error cache", flush=True)
    print("Applying frozen q99 threshold and 3-consecutive-sample onset", flush=True)

    normal = pd.read_parquet(NORMAL_CACHE, columns=SENSORS)
    thresholds = funcs["calculate_thresholds"](normal, SENSORS, 0.99)
    del normal
    OUT.mkdir(parents=True, exist_ok=True)
    thresh_path = OUT / "v2_5_normal_training_thresholds.csv"
    threshold_df = pd.DataFrame({"sensor": SENSORS,
                                 "threshold": [thresholds[s] for s in SENSORS]})
    # Verify numeric agreement with original training evaluation, if available.
    old_thresh = BASE / "V2_5_Leading_Edge/v2_5_normal_thresholds.csv"
    if old_thresh.is_file():
        old = pd.read_csv(old_thresh).set_index("sensor")["threshold"]
        delta = max(abs(thresholds[s] - float(old.loc[s])) for s in SENSORS)
        # CSV serialization can round q99 values by a few float32 ULPs.
        # This is a verification tolerance ONLY; the q99 thresholds used
        # by the frozen ranking function are never modified or rounded here.
        if not np.isfinite(delta) or delta > 1e-6:
            raise RuntimeError(f"Frozen training threshold mismatch: {delta}")
        print(
            f"Original saved thresholds match within CSV precision: "
            f"max delta={delta:.3e}, tolerance=1.0e-6",
            flush=True,
        )
    threshold_df.to_csv(thresh_path, index=False)
    print("Ground-truth labels are NOT used in the ranking stage.", flush=True)

    rank_paths = []
    for fault in sorted(TRUE_CAUSES):
        source = TEST_CACHE / f"fault_{fault:02d}_errors.parquet"
        print(f"Attributing fault {fault:02d} / 15 (500 testing runs)...", flush=True)
        frame = pd.read_parquet(source)
        validate_run_ids(frame, fault)
        ranked = []
        for run_number, run_frame in frame.groupby("simulationRun", sort=True):
            result = funcs["attribute_run"](
                run_df=run_frame, sensors=SENSORS,
                thresholds=thresholds, sustain_k=3
            )
            result.insert(0, "simulationRun", int(run_number))
            result.insert(0, "faultNumber", int(fault))
            ranked.append(result)
        del frame
        output = OUT / f"v2_5_testing_rankings_fault_{fault:02d}.parquet"
        if output.exists():
            # We do not overwrite test outputs silently.
            raise FileExistsError(f"Existing ranking file: {output}")
        combined = pd.concat(ranked, ignore_index=True)
        if len(combined) != 500 * 52:
            raise RuntimeError(f"Fault {fault}: unexpected ranking count")
        combined.to_parquet(output, index=False)
        rank_paths.append(output)
        print(f"  Saved 500 x 52 sensor rankings: {output.name}", flush=True)
        del ranked, combined
    print("\nALL LABEL-FREE TESTING RANKINGS SAVED.", flush=True)
    print("GROUND-TRUTH MAPPINGS ENTER ONLY FOR THE EVALUATION BELOW.\n", flush=True)

    evaluation_rows = []
    per_fault_rows = []
    for fault, source in zip(sorted(TRUE_CAUSES), rank_paths):
        rankings = pd.read_parquet(source)
        cause = TRUE_CAUSES[fault]
        truth = rankings.loc[
            rankings["sensor"] == cause,
            ["faultNumber", "simulationRun", "v2_5_rank", "onset", "onset_strength"]
        ].copy()
        if len(truth) != 500 or truth.simulationRun.nunique() != 500:
            raise RuntimeError(f"Fault {fault}: missing origin rank")
        truth = truth.rename(columns={
            "v2_5_rank": "true_rank", "onset": "true_onset",
            "onset_strength": "true_onset_strength",
        })
        truth.insert(2, "true_sensor", cause)
        evaluation_rows.append(truth)
        scores = funcs["calculate_metrics"](truth)
        per_fault_rows.append({"fault": fault, "true_sensor": cause, **scores})
        print(f"Fault {fault:02d} | top1={scores['Top1_%']:.2f}% "
              f"top3={scores['Top3_%']:.2f}% top5={scores['Top5_%']:.2f}% "
              f"median rank={scores['Median_rank']:.1f}", flush=True)
    eval_df = pd.concat(evaluation_rows, ignore_index=True)
    per_fault = pd.DataFrame(per_fault_rows)
    overall = funcs["calculate_metrics"](eval_df)
    eval_df.to_csv(OUT / "v2_5_testing_evaluation_by_run.csv", index=False)
    per_fault.to_csv(OUT / "v2_5_testing_per_fault_results.csv", index=False)
    summary = {
        "dataset": "TEP_Faulty_Testing",
        "n_known_origin_faults": 15,
        "n_test_runs": int(len(eval_df)),
        "normal_reference": str(NORMAL_CACHE),
        "testing_cache": str(TEST_CACHE),
        "original_v2_5_source": str(ORIGINAL),
        "original_v2_5_sha256": algorithm_sha,
        "threshold_quantile": 0.99,
        "sustain_k": 3,
        "ranking_code": "exact original AST FunctionDef nodes",
        "true_causes_evaluation_only": TRUE_CAUSES,
        "overall": {k: float(v) for k, v in overall.items()},
    }
    (OUT / "v2_5_testing_metadata.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 85, flush=True)
    print("V2.5 INDEPENDENT TEST PERFORMANCE — 7,500 RUNS, 15 FAULTS", flush=True)
    print("=" * 85, flush=True)
    for k, v in overall.items():
        print(f"{k:<16}: {v:.2f}", flush=True)
    print("\nHistorical training-set baseline: Top1 20.25%, Top3 39.63%, "
          "Top5 41.15%, median rank 12", flush=True)
    print("CAUTION: testing fault injection occurs later in longer runs; "
          "differences in pre-fault duration can affect first-onset ranking. "
          "We did NOT change or crop the frozen algorithm.", flush=True)
    print(f"Results: {OUT}", flush=True)


if __name__ == "__main__":
    main()
