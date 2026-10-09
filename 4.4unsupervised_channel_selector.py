"""
unsupervised_channel_selector.py

Fast label-free selector for attribution channels.

Changes from the first implementation
-------------------------------------
1. Stability is run-to-consensus rather than every-run-to-every-run.
   Complexity changes from O(R^2) to O(R) per scenario/channel.
2. Sensor ranks and rank evidence are precomputed once as NumPy arrays.
3. Bootstrap is hierarchical: scenarios are sampled first, then runs within
   each sampled scenario.
4. Stability is reported as unavailable (NaN) when every scenario has one run.
   Reliability then uses the geometric mean of the metrics that are estimable.
5. Progress output makes long runs visibly active.

Public API is compatible with run_attribution.py:
    SelectorConfig
    SelectionResult
    select_primary_channel_unsupervised
    print_selection_report
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class SelectorConfig:
    top_k: int = 3
    bootstrap_iterations: int = 500
    random_seed: int = 42
    min_reliability: float = 0.15
    min_win_probability: float = 0.60
    min_win_margin: float = 0.15
    evidence_temperature: float = 1.0
    progress_every: int = 50


@dataclass
class SelectionResult:
    selected_channel: Optional[str]
    abstained: bool
    reason: str
    metrics: pd.DataFrame
    bootstrap_win_probability: pd.Series
    winner_margin: float


@dataclass
class _PreparedScenario:
    # channel -> arrays with shape (runs, sensors)
    scores: Dict[str, np.ndarray]
    ranks: Dict[str, np.ndarray]
    evidence: Dict[str, np.ndarray]


@dataclass
class _PreparedData:
    scenarios: Dict[str, _PreparedScenario]
    channels: List[str]
    sensors: List[str]


def _as_score_series(value: Any) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.copy()
    elif isinstance(value, Mapping):
        series = pd.Series(dict(value))
    else:
        raise TypeError("Each channel result must be a Series or sensor->score mapping.")
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    series = series.dropna().astype(float)
    if series.empty:
        raise ValueError("A channel result contained no finite sensor scores.")
    if series.index.has_duplicates:
        raise ValueError("Sensor names must be unique within a channel result.")
    return series


def _rank_rows_desc(values: np.ndarray) -> np.ndarray:
    """Average descending ranks, row by row. Runs once during preparation."""
    return pd.DataFrame(values).rank(axis=1, method="average", ascending=False).to_numpy(float)


def _prepare(features_by_scenario: Mapping[str, Sequence[Mapping[str, Any]]],
             temperature: float) -> _PreparedData:
    if not features_by_scenario:
        raise ValueError("features_by_scenario is empty.")

    channels: Optional[List[str]] = None
    sensors: Optional[List[str]] = None
    prepared: Dict[str, _PreparedScenario] = {}

    for scenario, runs in features_by_scenario.items():
        if not runs:
            raise ValueError(f"Scenario {scenario!r} has no runs.")

        first_run = runs[0]
        current_channels = sorted(first_run.keys())
        if channels is None:
            channels = current_channels
        elif current_channels != channels:
            raise ValueError("Every run must contain the same channels.")

        first_series = _as_score_series(first_run[current_channels[0]])
        current_sensors = sorted(map(str, first_series.index))
        if sensors is None:
            sensors = current_sensors
        elif current_sensors != sensors:
            raise ValueError("Every scenario must contain the same sensors.")

        channel_scores: Dict[str, List[np.ndarray]] = {c: [] for c in channels}
        for run in runs:
            if sorted(run.keys()) != channels:
                raise ValueError("Every run must contain the same channels.")
            for channel in channels:
                series = _as_score_series(run[channel])
                series.index = series.index.map(str)
                if sorted(series.index) != sensors:
                    raise ValueError("Every run/channel must contain the same sensors.")
                channel_scores[channel].append(series.reindex(sensors).to_numpy(float))

        score_arrays: Dict[str, np.ndarray] = {}
        rank_arrays: Dict[str, np.ndarray] = {}
        evidence_arrays: Dict[str, np.ndarray] = {}
        n_sensors = len(sensors)
        temp = max(float(temperature), EPS)

        for channel in channels:
            scores = np.vstack(channel_scores[channel])
            ranks = _rank_rows_desc(scores)
            evidence = np.log(n_sensors / np.clip(ranks, 1.0, None)) / temp
            score_arrays[channel] = scores
            rank_arrays[channel] = ranks
            evidence_arrays[channel] = evidence

        prepared[str(scenario)] = _PreparedScenario(
            scores=score_arrays, ranks=rank_arrays, evidence=evidence_arrays
        )

    assert channels is not None and sensors is not None
    return _PreparedData(prepared, channels, sensors)


def _row_consensus_stability(ranks: np.ndarray, top_k: int) -> float:
    """Mean run agreement with the scenario consensus; O(runs*sensors)."""
    n_runs, n_sensors = ranks.shape
    if n_runs < 2:
        return float("nan")

    consensus = np.median(ranks, axis=0)

    # Row-wise Pearson correlation of rank vectors, mapped from [-1,1] to [0,1].
    x = ranks - ranks.mean(axis=1, keepdims=True)
    y = consensus - consensus.mean()
    denom = np.sqrt((x * x).sum(axis=1) * np.sum(y * y))
    corr = np.divide(x @ y, denom, out=np.zeros(n_runs), where=denom > EPS)
    rank_agreement = np.clip((corr + 1.0) / 2.0, 0.0, 1.0)

    # Top-k overlap with consensus top-k. Lowest ranks are best.
    k = max(1, min(int(top_k), n_sensors))
    consensus_top = np.argpartition(consensus, k - 1)[:k]
    run_top = np.argpartition(ranks, k - 1, axis=1)[:, :k]
    overlap = np.isin(run_top, consensus_top).sum(axis=1) / k

    return float(np.mean(0.5 * rank_agreement + 0.5 * overlap))


def _posterior_from_evidence(evidence: np.ndarray) -> np.ndarray:
    shifted = evidence - np.max(evidence)
    weights = np.exp(shifted)
    return weights / max(float(weights.sum()), EPS)


def _concentration(p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum()
    if len(p) <= 1:
        return 1.0
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.clip(1.0 - entropy / np.log(len(p)), 0.0, 1.0))


def _separation(p: np.ndarray) -> float:
    if len(p) <= 1:
        return 1.0
    two = np.partition(p, -2)[-2:]
    p2, p1 = float(two[0]), float(two[1])
    return float(np.clip((p1 - p2) / max(p1, EPS), 0.0, 1.0))


def _diversity(winners: Sequence[int], n_sensors: int) -> float:
    if not winners or n_sensors <= 1:
        return 0.0
    counts = np.bincount(np.asarray(winners, dtype=int), minlength=n_sensors)
    probs = counts[counts > 0] / len(winners)
    entropy = -float(np.sum(probs * np.log(probs)))
    maximum = np.log(min(n_sensors, len(winners)))
    return 0.0 if maximum <= EPS else float(np.clip(entropy / maximum, 0.0, 1.0))


def _geometric_available(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    values = np.clip(values, EPS, 1.0)
    return float(np.exp(np.mean(np.log(values))))


def _metrics_from_prepared(prepared: _PreparedData,
                           config: SelectorConfig,
                           scenario_draws: Optional[List[Tuple[str, np.ndarray]]] = None
                           ) -> pd.DataFrame:
    """Compute metrics, optionally for a hierarchical bootstrap sample."""
    if scenario_draws is None:
        scenario_draws = [
            (name, np.arange(next(iter(sc.scores.values())).shape[0]))
            for name, sc in prepared.scenarios.items()
        ]

    rows: Dict[str, Dict[str, float]] = {}
    n_sensors = len(prepared.sensors)

    for channel in prepared.channels:
        stabilities: List[float] = []
        concentrations: List[float] = []
        separations: List[float] = []
        winners: List[int] = []

        for scenario_name, run_indices in scenario_draws:
            sc = prepared.scenarios[scenario_name]
            ranks = sc.ranks[channel][run_indices]
            evidence = sc.evidence[channel][run_indices]

            stability = _row_consensus_stability(ranks, config.top_k)
            if np.isfinite(stability):
                stabilities.append(stability)

            aggregate = np.median(evidence, axis=0)
            posterior = _posterior_from_evidence(aggregate)
            concentrations.append(_concentration(posterior))
            separations.append(_separation(posterior))
            winners.append(int(np.argmax(posterior)))

        stability = float(np.mean(stabilities)) if stabilities else float("nan")
        concentration = float(np.mean(concentrations))
        separation = float(np.mean(separations))
        diversity = _diversity(winners, n_sensors)
        reliability = _geometric_available(
            [stability, concentration, separation, diversity]
        )

        rows[channel] = {
            "stability": stability,
            "concentration": concentration,
            "separation": separation,
            "diversity": diversity,
            "reliability": reliability,
        }

    return pd.DataFrame.from_dict(rows, orient="index").sort_values(
        "reliability", ascending=False
    )


def compute_channel_metrics(
    features_by_scenario: Mapping[str, Sequence[Mapping[str, Any]]],
    config: SelectorConfig = SelectorConfig(),
) -> pd.DataFrame:
    prepared = _prepare(features_by_scenario, config.evidence_temperature)
    return _metrics_from_prepared(prepared, config)


def _hierarchical_draw(prepared: _PreparedData,
                       rng: np.random.Generator) -> List[Tuple[str, np.ndarray]]:
    names = list(prepared.scenarios)
    sampled = rng.choice(names, size=len(names), replace=True)
    draws: List[Tuple[str, np.ndarray]] = []
    for name in sampled:
        name = str(name)
        n_runs = next(iter(prepared.scenarios[name].scores.values())).shape[0]
        run_indices = rng.integers(0, n_runs, size=n_runs)
        draws.append((name, run_indices))
    return draws


def _bootstrap_from_prepared(prepared: _PreparedData,
                             config: SelectorConfig) -> pd.Series:
    rng = np.random.default_rng(config.random_seed)
    wins = pd.Series(0.0, index=prepared.channels, dtype=float)
    iterations = max(1, int(config.bootstrap_iterations))

    print(f"  bootstrap: {iterations} hierarchical resamples", flush=True)
    for iteration in range(1, iterations + 1):
        draws = _hierarchical_draw(prepared, rng)
        metrics = _metrics_from_prepared(prepared, config, draws)
        best = float(metrics["reliability"].max())
        tied = metrics.index[
            np.isclose(metrics["reliability"], best, rtol=1e-10, atol=1e-12)
        ]
        wins.loc[list(tied)] += 1.0 / len(tied)

        every = max(0, int(config.progress_every))
        if every and (iteration % every == 0 or iteration == iterations):
            current = (wins / iteration).sort_values(ascending=False)
            print(
                f"    {iteration:>4}/{iterations}: leader={current.index[0]} "
                f"({current.iloc[0]:.3f})",
                flush=True,
            )

    return (wins / iterations).sort_values(ascending=False)


def bootstrap_channel_wins(
    features_by_scenario: Mapping[str, Sequence[Mapping[str, Any]]],
    config: SelectorConfig = SelectorConfig(),
) -> pd.Series:
    prepared = _prepare(features_by_scenario, config.evidence_temperature)
    return _bootstrap_from_prepared(prepared, config)


def select_primary_channel_unsupervised(
    features_by_scenario: Mapping[str, Sequence[Mapping[str, Any]]],
    config: SelectorConfig = SelectorConfig(),
) -> SelectionResult:
    print("  preparing ranks/evidence once...", flush=True)
    prepared = _prepare(features_by_scenario, config.evidence_temperature)
    metrics = _metrics_from_prepared(prepared, config)
    win_probability = _bootstrap_from_prepared(prepared, config)

    best_channel = str(metrics.index[0])
    best_reliability = float(metrics.iloc[0]["reliability"])
    best_win_probability = float(win_probability.get(best_channel, 0.0))
    sorted_wins = win_probability.sort_values(ascending=False)
    runner_up = float(sorted_wins.iloc[1]) if len(sorted_wins) > 1 else 0.0
    winner_margin = best_win_probability - runner_up

    reasons: List[str] = []
    if best_reliability < config.min_reliability:
        reasons.append(
            f"best reliability {best_reliability:.3f} is below "
            f"{config.min_reliability:.3f}"
        )
    if best_win_probability < config.min_win_probability:
        reasons.append(
            f"bootstrap win probability {best_win_probability:.3f} is below "
            f"{config.min_win_probability:.3f}"
        )
    if winner_margin < config.min_win_margin:
        reasons.append(
            f"winner margin {winner_margin:.3f} is below {config.min_win_margin:.3f}"
        )

    if reasons:
        return SelectionResult(
            None, True, "; ".join(reasons), metrics, win_probability, winner_margin
        )

    reason = (
        f"selected {best_channel}: reliability={best_reliability:.3f}, "
        f"bootstrap win probability={best_win_probability:.3f}, "
        f"margin={winner_margin:.3f}"
    )
    return SelectionResult(
        best_channel, False, reason, metrics, win_probability, winner_margin
    )


def print_selection_report(result: SelectionResult) -> None:
    print("\nUNSUPERVISED CHANNEL AUDIT")
    print("-" * 72)
    shown = result.metrics.copy()
    # Make missing TTS stability explicit rather than printing a fake 0.5.
    print(shown.round(4).to_string(na_rep="not estimable"))
    print("\nBootstrap win probabilities:")
    print(result.bootstrap_win_probability.round(4).to_string())
    print("\nDecision:", "ABSTAIN" if result.abstained else result.selected_channel)
    print("Reason:", result.reason)


__all__ = [
    "SelectorConfig",
    "SelectionResult",
    "compute_channel_metrics",
    "bootstrap_channel_wins",
    "select_primary_channel_unsupervised",
    "print_selection_report",
]
