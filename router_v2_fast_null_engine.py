"""
Fast drop-in null-testing engine for the Router V2 TEP validation script.

Replace the original shifted_runs(), surrogate_best_of_lags(), and null_test()
with this module's fast_null_test() call:

    c = fast_null_test(t, runs, cfg)

The statistical test is preserved:
  * circular shifts are independent inside each simulation run;
  * every surrogate scans lags 1..max_lag;
  * the maximum surrogate improvement is compared with the observed best lag;
  * Monte-Carlo p-values and BH-FDR are calculated as before.

Speed changes:
  * never copies all 52 columns to shift one source;
  * precomputes target-only lag matrices, scaling, and baseline RMSE once;
  * groups candidate edges by source;
  * creates one shifted source per source/surrogate and reuses it across targets;
  * stores only summary statistics and exceedance counts, not all model objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass
class TargetLagCache:
    """Objects that do not change across source surrogates."""
    lag: int
    train_y: np.ndarray
    test_y: np.ndarray
    train_target: np.ndarray
    test_target: np.ndarray
    baseline_rmse: float
    train_run_ids: np.ndarray
    test_run_ids: np.ndarray
    train_rows_by_run: List[int]
    test_rows_by_run: List[int]


def _ridge_fit(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    A = np.empty((X.shape[0], X.shape[1] + 1), dtype=np.float64)
    A[:, 0] = 1.0
    A[:, 1:] = X
    gram = A.T @ A
    gram.flat[:: gram.shape[0] + 1] += ridge
    gram[0, 0] -= ridge
    return np.linalg.solve(gram, A.T @ y)


def _ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


def _split_run_indices(n_runs: int, test_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    if n_runs < 2:
        raise ValueError("Fast engine expects at least two independent runs.")
    cut = max(1, min(n_runs - 1, int(np.floor(n_runs * (1.0 - test_fraction)))))
    return np.arange(cut), np.arange(cut, n_runs)


def _target_design_for_runs(
    runs: Sequence[np.ndarray], run_ids: np.ndarray, target: int, lag: int
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    ys, xs, sizes = [], [], []
    for rid in run_ids:
        values = runs[int(rid)][:, target]
        n = len(values)
        if n <= lag + 2:
            sizes.append(0)
            continue
        y = values[lag:]
        x = np.column_stack([values[lag-k:n-k] for k in range(1, lag+1)])
        ys.append(y)
        xs.append(x)
        sizes.append(len(y))
    if not ys:
        return np.empty(0), np.empty((0, lag)), sizes
    return np.concatenate(ys), np.vstack(xs), sizes


def _standardise_train_test(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return (train - mean) / std, (test - mean) / std


def build_target_cache(
    runs: Sequence[np.ndarray],
    target_indices: Sequence[int],
    max_lag: int,
    test_fraction: float,
    ridge: float,
) -> Dict[Tuple[int, int], TargetLagCache]:
    """Precompute the unchanged target-only side of every regression."""
    train_ids, test_ids = _split_run_indices(len(runs), test_fraction)
    cache: Dict[Tuple[int, int], TargetLagCache] = {}

    for target in sorted(set(map(int, target_indices))):
        for lag in range(1, max_lag + 1):
            ytr, xtr, tr_sizes = _target_design_for_runs(runs, train_ids, target, lag)
            yte, xte, te_sizes = _target_design_for_runs(runs, test_ids, target, lag)
            if len(ytr) < max(40, 5 * lag) or len(yte) < 10:
                continue

            y_mean = ytr.mean()
            y_std = ytr.std()
            if y_std < EPS:
                y_std = 1.0
            ytr = (ytr - y_mean) / y_std
            yte = (yte - y_mean) / y_std
            xtr, xte = _standardise_train_test(xtr, xte)

            beta = _ridge_fit(xtr, ytr, ridge)
            rmse = float(np.sqrt(np.mean((yte - _ridge_predict(xte, beta)) ** 2)))
            cache[(target, lag)] = TargetLagCache(
                lag=lag,
                train_y=ytr,
                test_y=yte,
                train_target=xtr,
                test_target=xte,
                baseline_rmse=rmse,
                train_run_ids=train_ids,
                test_run_ids=test_ids,
                train_rows_by_run=tr_sizes,
                test_rows_by_run=te_sizes,
            )
    return cache


def _draw_shifts(
    runs: Sequence[np.ndarray], rng: np.random.Generator, max_lag: int
) -> np.ndarray:
    shifts = np.zeros(len(runs), dtype=np.int32)
    for i, run in enumerate(runs):
        n = len(run)
        margin = max(2 * max_lag + 1, int(0.05 * n))
        if n > 2 * margin + 2:
            shifts[i] = int(rng.integers(margin, n - margin))
    return shifts


def _shifted_source_lags(
    runs: Sequence[np.ndarray],
    run_ids: np.ndarray,
    source: int,
    shifts: np.ndarray,
    lag: int,
) -> np.ndarray:
    """Build only source-lag columns. No run or 52-column array is copied."""
    blocks = []
    for rid in run_ids:
        rid = int(rid)
        source_values = runs[rid][:, source]
        shift = int(shifts[rid])
        shifted = np.roll(source_values, shift) if shift else source_values
        n = len(shifted)
        if n <= lag + 2:
            continue
        blocks.append(np.column_stack([
            shifted[lag-k:n-k] for k in range(1, lag+1)
        ]))
    return np.vstack(blocks) if blocks else np.empty((0, lag))


def surrogate_improvement(
    runs: Sequence[np.ndarray],
    source: int,
    target: int,
    shifts: np.ndarray,
    cache: Dict[Tuple[int, int], TargetLagCache],
    max_lag: int,
    ridge: float,
) -> float:
    """Return the surrogate's best improvement across all candidate lags."""
    best = -np.inf
    for lag in range(1, max_lag + 1):
        fixed = cache.get((target, lag))
        if fixed is None:
            continue

        strn = _shifted_source_lags(
            runs, fixed.train_run_ids, source, shifts, lag
        )
        stst = _shifted_source_lags(
            runs, fixed.test_run_ids, source, shifts, lag
        )
        if len(strn) != len(fixed.train_y) or len(stst) != len(fixed.test_y):
            continue
        strn, stst = _standardise_train_test(strn, stst)

        xtr = np.concatenate((fixed.train_target, strn), axis=1)
        xte = np.concatenate((fixed.test_target, stst), axis=1)
        beta = _ridge_fit(xtr, fixed.train_y, ridge)
        ext_rmse = float(np.sqrt(np.mean(
            (fixed.test_y - _ridge_predict(xte, beta)) ** 2
        )))
        score = (fixed.baseline_rmse - ext_rmse) / max(fixed.baseline_rmse, EPS)
        if score > best:
            best = score
    return float(best) if np.isfinite(best) else np.nan


def bh_fdr(p: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    ordered = valid.sort_values()
    values = ordered.to_numpy(dtype=float)
    m = len(values)
    q = values * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out.loc[ordered.index] = np.clip(q, 0.0, 1.0)
    return out


def fast_null_test(table: pd.DataFrame, runs: Sequence[np.ndarray], cfg) -> pd.DataFrame:
    candidate = table[
        (table.improvement >= cfg.min_improvement)
        & (table.split_stability >= cfg.min_stability)
        & (table.directionality >= cfg.min_directionality)
    ].copy()

    print(
        f"Fast null-testing {len(candidate)} candidate edges with "
        f"{cfg.surrogates} circular shifts; every surrogate scans "
        f"lags 1..{cfg.max_lag} ...",
        flush=True,
    )
    if candidate.empty:
        for col in ("null_p", "null_median", "null_q95", "null_max", "fdr_q"):
            candidate[col] = np.nan
        candidate["retained"] = False
        return candidate

    cache = build_target_cache(
        runs=runs,
        target_indices=candidate.target_idx.unique(),
        max_lag=cfg.max_lag,
        test_fraction=cfg.test_fraction,
        ridge=cfg.ridge,
    )

    rng = np.random.default_rng(cfg.random_seed)
    null_values: Dict[int, np.ndarray] = {
        int(idx): np.full(cfg.surrogates, np.nan, dtype=np.float64)
        for idx in candidate.index
    }

    groups = list(candidate.groupby("source_idx", sort=True))
    completed = 0
    total = len(candidate) * cfg.surrogates

    for source_idx, source_edges in groups:
        source_idx = int(source_idx)
        edge_records = list(source_edges.itertuples())

        for surrogate_id in range(cfg.surrogates):
            # One independent within-run circular shift for this source and
            # surrogate, reused for all candidate targets from that source.
            shifts = _draw_shifts(runs, rng, cfg.max_lag)
            for row in edge_records:
                null_values[int(row.Index)][surrogate_id] = surrogate_improvement(
                    runs=runs,
                    source=source_idx,
                    target=int(row.target_idx),
                    shifts=shifts,
                    cache=cache,
                    max_lag=cfg.max_lag,
                    ridge=cfg.ridge,
                )
                completed += 1

            if completed and (completed % max(1, total // 20) < len(edge_records)):
                print(f"  null progress: {completed}/{total} edge-surrogates", flush=True)

    pvals, medians, q95s, maxima = [], [], [], []
    for idx, row in candidate.iterrows():
        values = null_values[int(idx)]
        values = values[np.isfinite(values)]
        observed = float(row.improvement)
        if not len(values):
            pvals.append(np.nan); medians.append(np.nan)
            q95s.append(np.nan); maxima.append(np.nan)
            continue
        pvals.append(float((1 + np.sum(values >= observed)) / (len(values) + 1)))
        medians.append(float(np.median(values)))
        q95s.append(float(np.quantile(values, 0.95)))
        maxima.append(float(np.max(values)))

    candidate["null_p"] = pvals
    candidate["null_median"] = medians
    candidate["null_q95"] = q95s
    candidate["null_max"] = maxima
    candidate["fdr_q"] = bh_fdr(candidate["null_p"])
    candidate["retained"] = candidate["fdr_q"] <= cfg.fdr_alpha
    return candidate
