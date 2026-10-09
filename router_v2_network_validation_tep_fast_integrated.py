#!/usr/bin/env python3
"""Router V2 Experiment 1B: corrected, accelerated TEP network validation.

Learns from TEP_FaultFree_Training.parquet only.
Observed edges and circular-shift surrogates both scan lags 1..max_lag.
The accelerated null engine caches target-only regressions and reuses each
shifted source across all candidate targets from that source.

Interpretation: directional predictive relationships, not proven causality.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
RUN_NAMES = ["simulationRun", "simulation_run", "simRun", "run", "run_id", "Run", "replicate"]
TIME_NAMES = ["sample", "sample_id", "time", "Time", "timestamp", "sampleNumber", "sample_number"]
META = {"faultNumber", "fault_number", "fault", "Fault", "Class", "class", "label", "true_cause", "amplitude", "fault_type"}


@dataclass(frozen=True)
class Config:
    max_lag: int = 10
    n_splits: int = 5
    test_fraction: float = 0.25
    min_improvement: float = 0.01
    min_stability: float = 0.60
    min_directionality: float = 0.10
    surrogates: int = 200
    fdr_alpha: float = 0.05
    random_seed: int = 42
    ridge: float = 1e-6
    max_runs: Optional[int] = None


@dataclass
class TargetLagCache:
    train_y: np.ndarray
    test_y: np.ndarray
    train_target: np.ndarray
    test_target: np.ndarray
    baseline_rmse: float
    train_run_ids: np.ndarray
    test_run_ids: np.ndarray


def find_col(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def sensor_key(name):
    value = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if value.startswith("xmeas"):
        digits = "".join(ch for ch in value[5:] if ch.isdigit())
        return 0, int(digits or 999)
    if value.startswith("xmv"):
        digits = "".join(ch for ch in value[3:] if ch.isdigit())
        return 1, int(digits or 999)
    return 2, 999


def canonicalise(df, run_col, time_col):
    named = [c for c in df.columns if "".join(ch for ch in str(c).lower() if ch.isalnum()).startswith(("xmeas", "xmv"))]
    if len(named) == 52:
        named = sorted(named, key=sensor_key)
        return df.rename(columns=dict(zip(named, SENSORS))), SENSORS.copy()
    excluded = set(META) | ({run_col} if run_col else set()) | ({time_col} if time_col else set())
    numeric = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) == 52:
        print("  Mapping 52 numeric columns in file order to xmeas_1..41,xmv_1..11.")
        return df.rename(columns=dict(zip(numeric, SENSORS))), SENSORS.copy()
    raise SystemExit(f"Could not identify exactly 52 sensors: named={len(named)}, numeric={len(numeric)}")


def load_normal(path: Path, max_runs: Optional[int]):
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    print(f"Loading {path.name} ...", flush=True)
    df = pd.read_parquet(path)
    run_col = find_col(df.columns, RUN_NAMES)
    time_col = find_col(df.columns, TIME_NAMES)
    df, sensors = canonicalise(df, run_col, time_col)
    values = df[sensors].replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    if values.isna().any().any():
        raise SystemExit("Unresolved NaNs in sensor data.")
    df.loc[:, sensors] = values
    runs = []
    if run_col:
        grouped = list(df.groupby(run_col, sort=True))
        if max_runs is not None:
            grouped = grouped[:max_runs]
        for _, block in grouped:
            if time_col:
                block = block.sort_values(time_col)
            array = block[sensors].to_numpy(np.float64, copy=True)
            if len(array) > 2:
                runs.append(array)
    else:
        block = df.sort_values(time_col) if time_col else df
        runs = [block[sensors].to_numpy(np.float64, copy=True)]
    lengths = list(map(len, runs))
    meta = {
        "file": str(path.resolve()), "rows": int(sum(lengths)), "n_runs": len(runs),
        "run_column": run_col, "time_column": time_col,
        "run_lengths_min": int(min(lengths)), "run_lengths_median": float(np.median(lengths)),
        "run_lengths_max": int(max(lengths)), "sensors": sensors,
    }
    return runs, sensors, meta


def split_runs(runs, test_fraction):
    if len(runs) >= 2:
        cut = max(1, min(len(runs) - 1, int(math.floor(len(runs) * (1.0 - test_fraction)))))
        return list(runs[:cut]), list(runs[cut:])
    run = runs[0]
    cut = int(math.floor(len(run) * (1.0 - test_fraction)))
    return [run[:cut]], [run[cut:]]


def design_run(run, source, target, lag):
    yv, xv, n = run[:, target], run[:, source], len(run)
    if n <= lag + 2:
        return np.empty(0), np.empty((0, lag)), np.empty((0, 2 * lag))
    y = yv[lag:]
    target_lags = [yv[lag-k:n-k] for k in range(1, lag + 1)]
    source_lags = [xv[lag-k:n-k] for k in range(1, lag + 1)]
    return y, np.column_stack(target_lags), np.column_stack(target_lags + source_lags)


def stack_design(runs, source, target, lag):
    ys, baselines, extended = [], [], []
    for run in runs:
        y, baseline, ext = design_run(run, source, target, lag)
        if len(y):
            ys.append(y); baselines.append(baseline); extended.append(ext)
    if not ys:
        return np.empty(0), np.empty((0, lag)), np.empty((0, 2 * lag))
    return np.concatenate(ys), np.vstack(baselines), np.vstack(extended)


def ridge_fit(X, y, ridge):
    A = np.empty((len(X), X.shape[1] + 1), dtype=np.float64)
    A[:, 0] = 1.0
    A[:, 1:] = X
    gram = A.T @ A
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, A.T @ y)


def ridge_predict(X, beta):
    return beta[0] + X @ beta[1:]


def standardise_train_test(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def model_improvement(train, test, source, target, lag, ridge):
    ytr, btr, etr = stack_design(train, source, target, lag)
    yte, bte, ete = stack_design(test, source, target, lag)
    if len(ytr) < max(40, 5 * lag) or len(yte) < 10:
        return np.nan, np.nan, np.nan
    mean, std = ytr.mean(), ytr.std()
    std = std if std >= EPS else 1.0
    ytr, yte = (ytr - mean) / std, (yte - mean) / std
    btr, bte = standardise_train_test(btr, bte)
    etr, ete = standardise_train_test(etr, ete)
    baseline_beta = ridge_fit(btr, ytr, ridge)
    extended_beta = ridge_fit(etr, ytr, ridge)
    baseline_rmse = float(np.sqrt(np.mean((yte - ridge_predict(bte, baseline_beta)) ** 2)))
    extended_rmse = float(np.sqrt(np.mean((yte - ridge_predict(ete, extended_beta)) ** 2)))
    return (baseline_rmse - extended_rmse) / max(baseline_rmse, EPS), baseline_rmse, extended_rmse


def best_pair(runs, source, target, cfg):
    train, test = split_runs(runs, cfg.test_fraction)
    scores = []
    for lag in range(1, cfg.max_lag + 1):
        imp, base, ext = model_improvement(train, test, source, target, lag, cfg.ridge)
        if np.isfinite(imp):
            scores.append((imp, lag, base, ext))
    if not scores:
        return {"improvement": np.nan, "best_lag": np.nan, "baseline_rmse": np.nan, "extended_rmse": np.nan}
    imp, lag, base, ext = max(scores, key=lambda item: item[0])
    return {"improvement": imp, "best_lag": lag, "baseline_rmse": base, "extended_rmse": ext}


def stability_blocks(runs, n_splits):
    if len(runs) >= n_splits:
        return [[runs[int(i)] for i in indices] for indices in np.array_split(np.arange(len(runs)), n_splits) if len(indices)]
    if len(runs) == 1:
        run = runs[0]
        cuts = np.linspace(0, len(run), n_splits + 1, dtype=int)
        return [[run[a:b]] for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 20]
    return [[run] for run in runs]


def pair_table(runs, sensors, cfg):
    blocks = stability_blocks(runs, cfg.n_splits)
    rows = []
    total = len(sensors) * (len(sensors) - 1)
    done = 0
    print(f"Testing {total} ordered pairs x lags 1..{cfg.max_lag} ...", flush=True)
    for source in range(len(sensors)):
        for target in range(len(sensors)):
            if source == target:
                continue
            full = best_pair(runs, source, target, cfg)
            split_scores, split_lags = [], []
            for block in blocks:
                if sum(map(len, block)) < max(100, 8 * cfg.max_lag):
                    continue
                result = best_pair(block, source, target, cfg)
                split_scores.append(result["improvement"])
                split_lags.append(result["best_lag"])
            finite = np.asarray([v for v in split_scores if np.isfinite(v)], dtype=float)
            finite_lags = [v for v in split_lags if np.isfinite(v)]
            rows.append({
                "source": sensors[source], "target": sensors[target],
                "source_idx": source, "target_idx": target,
                "best_lag": full["best_lag"], "improvement": full["improvement"],
                "baseline_rmse": full["baseline_rmse"], "extended_rmse": full["extended_rmse"],
                "split_stability": float(np.mean(finite >= cfg.min_improvement)) if len(finite) else np.nan,
                "median_split_improvement": float(np.median(finite)) if len(finite) else np.nan,
                "median_split_lag": float(np.median(finite_lags)) if finite_lags else np.nan,
                "n_valid_splits": len(finite),
            })
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                print(f"  pair progress: {done}/{total}", flush=True)
    table = pd.DataFrame(rows)
    reverse = table[["source", "target", "improvement"]].rename(
        columns={"source": "target", "target": "source", "improvement": "reverse_improvement"}
    )
    table = table.merge(reverse, on=["source", "target"], how="left")
    table["direction_margin"] = table["improvement"] - table["reverse_improvement"]
    denominator = np.maximum(np.maximum(table["improvement"].abs(), table["reverse_improvement"].abs()), EPS)
    table["directionality"] = table["direction_margin"] / denominator
    return table


def target_design(runs, run_ids, target, lag):
    ys, xs = [], []
    for rid in run_ids:
        values = runs[int(rid)][:, target]
        n = len(values)
        if n <= lag + 2:
            continue
        ys.append(values[lag:])
        xs.append(np.column_stack([values[lag-k:n-k] for k in range(1, lag + 1)]))
    if not ys:
        return np.empty(0), np.empty((0, lag))
    return np.concatenate(ys), np.vstack(xs)


def build_target_cache(runs, target_indices, cfg):
    cut = max(1, min(len(runs) - 1, int(np.floor(len(runs) * (1.0 - cfg.test_fraction)))))
    train_ids, test_ids = np.arange(cut), np.arange(cut, len(runs))
    cache: Dict[Tuple[int, int], TargetLagCache] = {}
    print("  precomputing target-only regressions ...", flush=True)
    for target in sorted(set(map(int, target_indices))):
        for lag in range(1, cfg.max_lag + 1):
            ytr, xtr = target_design(runs, train_ids, target, lag)
            yte, xte = target_design(runs, test_ids, target, lag)
            if len(ytr) < max(40, 5 * lag) or len(yte) < 10:
                continue
            y_mean, y_std = ytr.mean(), ytr.std()
            y_std = y_std if y_std >= EPS else 1.0
            ytr, yte = (ytr - y_mean) / y_std, (yte - y_mean) / y_std
            xtr, xte = standardise_train_test(xtr, xte)
            beta = ridge_fit(xtr, ytr, cfg.ridge)
            baseline_rmse = float(np.sqrt(np.mean((yte - ridge_predict(xte, beta)) ** 2)))
            cache[(target, lag)] = TargetLagCache(ytr, yte, xtr, xte, baseline_rmse, train_ids, test_ids)
    return cache


def draw_shifts(runs, rng, max_lag):
    shifts = np.zeros(len(runs), dtype=np.int32)
    for index, run in enumerate(runs):
        n = len(run)
        margin = max(2 * max_lag + 1, int(0.05 * n))
        if n > 2 * margin + 2:
            shifts[index] = int(rng.integers(margin, n - margin))
    return shifts


def shifted_source_lags(runs, run_ids, source, shifts, lag):
    blocks = []
    for rid in run_ids:
        rid = int(rid)
        values = runs[rid][:, source]
        shift = int(shifts[rid])
        shifted = np.roll(values, shift) if shift else values
        n = len(shifted)
        if n > lag + 2:
            blocks.append(np.column_stack([shifted[lag-k:n-k] for k in range(1, lag + 1)]))
    return np.vstack(blocks) if blocks else np.empty((0, lag))


def surrogate_best(runs, source, target, shifts, cache, cfg):
    best = -np.inf
    for lag in range(1, cfg.max_lag + 1):
        fixed = cache.get((target, lag))
        if fixed is None:
            continue
        source_train = shifted_source_lags(runs, fixed.train_run_ids, source, shifts, lag)
        source_test = shifted_source_lags(runs, fixed.test_run_ids, source, shifts, lag)
        if len(source_train) != len(fixed.train_y) or len(source_test) != len(fixed.test_y):
            continue
        source_train, source_test = standardise_train_test(source_train, source_test)
        train_x = np.concatenate((fixed.train_target, source_train), axis=1)
        test_x = np.concatenate((fixed.test_target, source_test), axis=1)
        beta = ridge_fit(train_x, fixed.train_y, cfg.ridge)
        extended_rmse = float(np.sqrt(np.mean((fixed.test_y - ridge_predict(test_x, beta)) ** 2)))
        score = (fixed.baseline_rmse - extended_rmse) / max(fixed.baseline_rmse, EPS)
        best = max(best, score)
    return float(best) if np.isfinite(best) else np.nan


def bh_fdr(pvalues):
    output = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna()
    if valid.empty:
        return output
    ordered = valid.sort_values()
    values = ordered.to_numpy(dtype=float)
    qvalues = values * len(values) / np.arange(1, len(values) + 1)
    qvalues = np.minimum.accumulate(qvalues[::-1])[::-1]
    output.loc[ordered.index] = np.clip(qvalues, 0.0, 1.0)
    return output


def fast_null_test(table, runs, cfg):
    candidate = table[
        (table["improvement"] >= cfg.min_improvement)
        & (table["split_stability"] >= cfg.min_stability)
        & (table["directionality"] >= cfg.min_directionality)
    ].copy()
    print(
        f"Fast null-testing {len(candidate)} candidate edges with {cfg.surrogates} circular shifts; "
        f"every surrogate scans lags 1..{cfg.max_lag} ...", flush=True,
    )
    if candidate.empty:
        for col in ["null_p", "null_median", "null_q95", "null_max", "fdr_q"]:
            candidate[col] = np.nan
        candidate["retained"] = False
        return candidate
    cache = build_target_cache(runs, candidate["target_idx"].unique(), cfg)
    rng = np.random.default_rng(cfg.random_seed)
    null_values = {int(idx): np.full(cfg.surrogates, np.nan) for idx in candidate.index}
    groups = list(candidate.groupby("source_idx", sort=True))
    total_sources = len(groups)
    for source_number, (source_idx, source_edges) in enumerate(groups, 1):
        records = list(source_edges.itertuples())
        for surrogate_id in range(cfg.surrogates):
            shifts = draw_shifts(runs, rng, cfg.max_lag)
            for row in records:
                null_values[int(row.Index)][surrogate_id] = surrogate_best(
                    runs, int(source_idx), int(row.target_idx), shifts, cache, cfg
                )
        print(
            f"  source progress: {source_number}/{total_sources} "
            f"({len(records)} edges x {cfg.surrogates} surrogates)", flush=True,
        )
    pvalues, medians, q95s, maxima = [], [], [], []
    for idx, row in candidate.iterrows():
        values = null_values[int(idx)]
        values = values[np.isfinite(values)]
        observed = float(row["improvement"])
        if not len(values):
            pvalues.append(np.nan); medians.append(np.nan); q95s.append(np.nan); maxima.append(np.nan)
        else:
            pvalues.append(float((1 + np.sum(values >= observed)) / (len(values) + 1)))
            medians.append(float(np.median(values)))
            q95s.append(float(np.quantile(values, 0.95)))
            maxima.append(float(np.max(values)))
    candidate["null_p"] = pvalues
    candidate["null_median"] = medians
    candidate["null_q95"] = q95s
    candidate["null_max"] = maxima
    candidate["fdr_q"] = bh_fdr(candidate["null_p"])
    candidate["retained"] = candidate["fdr_q"] <= cfg.fdr_alpha
    return candidate


def write_outputs(table, candidate, retained, sensors, cfg, meta, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "all_ordered_edges.csv", index=False)
    candidate.to_csv(output_dir / "candidate_edges_null_test.csv", index=False)
    retained.to_csv(output_dir / "normal_network_edges.csv", index=False)
    table[["source", "target", "best_lag", "improvement", "split_stability", "median_split_improvement", "median_split_lag", "directionality"]].to_csv(output_dir / "edge_stability.csv", index=False)
    table[["source", "target", "improvement", "null_p", "fdr_q", "null_median", "null_q95", "null_max", "retained"]].to_csv(output_dir / "shuffle_null_results.csv", index=False)
    nodes = []
    for sensor in sensors:
        outgoing, incoming = retained[retained.source == sensor], retained[retained.target == sensor]
        nodes.append({
            "sensor": sensor, "out_degree": len(outgoing), "in_degree": len(incoming),
            "out_strength": float(outgoing.improvement.clip(lower=0).sum()),
            "in_strength": float(incoming.improvement.clip(lower=0).sum()),
        })
    nodes = pd.DataFrame(nodes)
    nodes.to_csv(output_dir / "node_degrees.csv", index=False)
    adjacency = pd.DataFrame(0.0, index=sensors, columns=sensors)
    for row in retained.itertuples(index=False):
        adjacency.loc[row.source, row.target] = row.improvement
    adjacency.to_csv(output_dir / "adjacency_improvement_matrix.csv", index_label="source")
    possible = len(sensors) * (len(sensors) - 1)
    summary = {
        "config": asdict(cfg), "source": meta, "faulty_files_used": False,
        "network": {"n_sensors": len(sensors), "possible_directed_edges": possible,
                    "retained_edges": len(retained), "density": len(retained) / possible},
        "null_test": "within-run circular shifts; observed and every surrogate scan lags 1..max_lag and retain their best improvement",
        "optimisation": "target-only regressions cached; shifted source reused across candidate targets",
        "interpretation": "directional predictive relationships, not proven causal edges",
    }
    with open(output_dir / "network_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return nodes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--normal-file", default="TEP_FaultFree_Training.parquet")
    parser.add_argument("--out", default="Router_V2_Network_TEP_fast")
    parser.add_argument("--max-lag", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--min-stability", type=float, default=0.60)
    parser.add_argument("--min-directionality", type=float, default=0.10)
    parser.add_argument("--surrogates", type=int, default=200)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        max_lag=args.max_lag, n_splits=args.splits, test_fraction=args.test_fraction,
        min_improvement=args.min_improvement, min_stability=args.min_stability,
        min_directionality=args.min_directionality, surrogates=args.surrogates,
        fdr_alpha=args.fdr_alpha, random_seed=args.seed, max_runs=args.max_runs,
    )
    print("=" * 78)
    print("ROUTER V2 EXPERIMENT 1B - FAST CORRECTED TEP NETWORK VALIDATION")
    print("=" * 78)
    print("Fault-free training data only. Fair best-of-lags surrogate test.\n")
    runs, sensors, meta = load_normal(Path(args.data_dir) / args.normal_file, cfg.max_runs)
    print(
        f"Normal rows used: {meta['rows']:,}\nDetected normal runs: {meta['n_runs']}\n"
        f"Run column: {meta['run_column']}\nTime/sample column: {meta['time_column']}\n"
        f"Run lengths min/median/max: {meta['run_lengths_min']} / {meta['run_lengths_median']:.0f} / {meta['run_lengths_max']}\n"
        f"Sensors: {len(sensors)} ({sensors[0]} ... {sensors[-1]})\n"
    )
    table = pair_table(runs, sensors, cfg)
    candidate = fast_null_test(table, runs, cfg)
    merge_columns = ["source", "target", "null_p", "null_median", "null_q95", "null_max", "fdr_q", "retained"]
    table = table.merge(candidate[merge_columns], on=["source", "target"], how="left")
    table["retained"] = table["retained"].fillna(False).astype(bool)
    retained = table[table.retained].sort_values(["improvement", "split_stability"], ascending=False)
    nodes = write_outputs(table, candidate, retained, sensors, cfg, meta, Path(args.out))
    possible = len(sensors) * (len(sensors) - 1)
    print("\nVALIDATION SUMMARY\n" + "-" * 78)
    print(f"Retained edges: {len(retained)} / {possible} (density {100 * len(retained) / possible:.2f}%)")
    if len(retained):
        columns = ["source", "target", "best_lag", "improvement", "split_stability", "directionality", "null_q95", "null_max", "fdr_q"]
        print("\nStrongest retained edges:\n" + retained[columns].head(20).round(4).to_string(index=False))
    else:
        print("No edges passed the predeclared gates.")
    print("\nHighest out-degree sensors:\n" + nodes.sort_values(["out_degree", "out_strength"], ascending=False).head(10).round(4).to_string(index=False))
    print(f"\nResults written to {args.out}/")


if __name__ == "__main__":
    main()
