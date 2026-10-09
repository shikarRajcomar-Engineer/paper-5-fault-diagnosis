#!/usr/bin/env python3
"""
router_v1_diagnostics.py
========================
Diagnostic experiment for Router v1. It does NOT change the attribution method.

Question
--------
Why can an internally reliable channel still be poor at true-origin attribution?

This script measures, for every dataset/channel:
  1. Router-v1 internal metrics: stability, concentration, separation, diversity,
     reliability.
  2. Within-scenario ranking distance: do repeated runs of the SAME scenario agree?
  3. Between-scenario ranking distance: do DIFFERENT scenarios produce different ranks?
  4. Specificity = between / within when repeat data exist.
  5. Labelled evaluation only AFTER the label-free diagnostics are computed:
     median true-origin rank, top-1, top-3 and balanced accuracy.
  6. Spearman relationships between internal metrics and labelled performance.
  7. Leave-one-component-out Router-v1 ablations.

It imports the existing project code and caches. Nothing here trains a model or
modifies run_attribution.py.

Usage
-----
    python router_v1_diagnostics.py --dataset tep
    python router_v1_diagnostics.py --all
    python router_v1_diagnostics.py --all --out Router_V1_Diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

import attribution_core as core
import datasets as ds
import run_attribution_unsupervised as driver
from unsupervised_channel_selector import SelectorConfig, compute_channel_metrics

EPS = 1e-12


def _rank_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Larger channel score = better candidate, hence descending ranks."""
    return frame.rank(axis=1, method="average", ascending=False).astype(float)


def _normalised_spearman_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distance between two sensor-rank vectors in [0,1].
    0 = identical ranking; 1 = perfectly reversed ranking.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return 0.0
    a0 = a - a.mean()
    b0 = b - b.mean()
    den = np.sqrt(np.dot(a0, a0) * np.dot(b0, b0))
    if den <= EPS:
        rho = 1.0 if np.allclose(a, b) else 0.0
    else:
        rho = float(np.dot(a0, b0) / den)
    return float(np.clip((1.0 - rho) / 2.0, 0.0, 1.0))


def ranking_geometry(features: Dict, channels: Sequence[str]) -> pd.DataFrame:
    """
    Compute label-free ranking geometry.

    Within-scenario distance:
        each repeated run versus that scenario's median consensus rank.
    Between-scenario distance:
        all pairs of scenario consensus rankings.
    Specificity:
        between / within. Undefined when no scenario has repeated runs.

    This deliberately uses no ground-truth origin.
    """
    rows = []
    for channel in channels:
        consensus = {}
        within_distances: List[float] = []
        repeat_scenarios = 0

        for scenario, per_channel in features.items():
            frame = per_channel[channel]
            if frame.empty:
                continue
            ranks = _rank_rows(frame)
            med = ranks.median(axis=0)
            consensus[scenario] = med

            if len(ranks) >= 2:
                repeat_scenarios += 1
                med_values = med.to_numpy(float)
                for _, row in ranks.iterrows():
                    within_distances.append(
                        _normalised_spearman_distance(
                            row.to_numpy(float), med_values
                        )
                    )

        between_distances: List[float] = []
        keys = list(consensus)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                common = consensus[keys[i]].index.intersection(
                    consensus[keys[j]].index
                )
                if len(common) < 2:
                    continue
                between_distances.append(
                    _normalised_spearman_distance(
                        consensus[keys[i]].loc[common].to_numpy(float),
                        consensus[keys[j]].loc[common].to_numpy(float),
                    )
                )

        within = (float(np.mean(within_distances))
                  if within_distances else float("nan"))
        between = (float(np.mean(between_distances))
                   if between_distances else float("nan"))
        specificity = (
            between / max(within, EPS)
            if np.isfinite(within) and np.isfinite(between)
            else float("nan")
        )

        rows.append(dict(
            channel=channel,
            within_distance=within,
            between_distance=between,
            specificity=specificity,
            repeat_scenarios=repeat_scenarios,
            n_scenarios=len(consensus),
        ))

    return pd.DataFrame(rows).set_index("channel")


def _rank_of_truth(scores: pd.Series, valid: Sequence[str]) -> float:
    """Best rank among valid origins, with average tie ranks."""
    ranks = scores.rank(method="average", ascending=False)
    values = [float(ranks[v]) for v in valid if v in ranks.index]
    return min(values) if values else float("nan")


def labelled_channel_performance(features: Dict, truth: Dict,
                                 sensors: Sequence[str]) -> pd.DataFrame:
    """
    External evaluation only. Labels enter here, after all label-free metrics
    have already been calculated.
    """
    chance = core.chance_level(truth)
    rows = []

    for channel in core.CHANNELS:
        # Audit median rank from the project's established labelled audit.
        audit_df = core.audit(features, truth)
        channel_audit = audit_df[audit_df["channel"] == channel]
        scenario_median_rank = float(channel_audit["median_rank"].median())

        # Existing final diagnosis machinery with the uniform prior only.
        tiers = {key: "?" for key in features}
        results = driver.evaluate(
            features, truth, sensors, (channel,), "uniform", [], tiers
        )
        if results.empty:
            top1 = top3 = balanced = float("nan")
        else:
            stats = core.summarise(results, chance)
            top1 = float(stats["top1"])
            top3 = float(stats["top3"])
            balanced = float(stats["balanced"])

        rows.append(dict(
            channel=channel,
            labelled_median_origin_rank=scenario_median_rank,
            top1=top1,
            top3=top3,
            balanced_accuracy=balanced,
        ))

    return pd.DataFrame(rows).set_index("channel")


def _geom(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    arr = np.clip(arr, EPS, 1.0)
    return float(np.exp(np.mean(np.log(arr))))


def ablation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Router-v1 component ablation. This is diagnostic only. It does not select a
    replacement formula and does not use labelled accuracy.
    """
    components = ["stability", "concentration", "separation", "diversity"]
    variants = {"full_SCGD": components}
    for omitted in components:
        variants[f"minus_{omitted}"] = [c for c in components if c != omitted]
    for component in components:
        variants[f"only_{component}"] = [component]

    rows = []
    for variant, included in variants.items():
        scores = pd.Series(
            {channel: _geom([metrics.loc[channel, c] for c in included])
             for channel in metrics.index},
            dtype=float,
        )
        winner = scores.idxmax() if scores.notna().any() else None
        for channel, score in scores.items():
            rows.append(dict(
                variant=variant,
                channel=channel,
                score=float(score),
                winner=(channel == winner),
            ))
    return pd.DataFrame(rows)


def spearman_correlations(table: pd.DataFrame) -> pd.DataFrame:
    """Link internal diagnostics to external performance for analysis only."""
    predictors = [
        "stability", "concentration", "separation", "diversity",
        "reliability", "within_distance", "between_distance", "specificity",
    ]
    outcomes = ["balanced_accuracy", "labelled_median_origin_rank"]
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            pair = table[[predictor, outcome]].dropna()
            rho = (float(pair[predictor].corr(pair[outcome], method="spearman"))
                   if len(pair) >= 3 else float("nan"))
            rows.append(dict(
                predictor=predictor,
                outcome=outcome,
                spearman_rho=rho,
                n=len(pair),
            ))
    return pd.DataFrame(rows)


def analyse_dataset(name: str, out_root: str) -> pd.DataFrame:
    config = ds.get(name)
    print("=" * 78)
    print(f"{config.name} - ROUTER V1 DIAGNOSTICS")
    print("=" * 78)

    normal_df, fault_df, _ = driver.load_cache(config)
    sensors = driver.identify_sensors(normal_df, fault_df, config)
    scenarios, truth = driver.build_scenarios(fault_df, sensors, config)
    driver.check_duplicate_scenarios(scenarios)
    mu, sigma, thresholds = driver.fit_baseline(normal_df, sensors, config)
    features = driver.extract_all_features(scenarios, mu, sigma, thresholds)

    print("\nComputing label-free Router-v1 metrics...")
    selector_cfg = SelectorConfig(
        top_k=3,
        bootstrap_iterations=1,  # not required for this diagnostic
        random_seed=42,
        min_reliability=0.0,
        min_win_probability=0.0,
        min_win_margin=0.0,
        progress_every=0,
    )
    internal = compute_channel_metrics(driver.selector_input(features), selector_cfg)

    print("Computing within/between scenario ranking geometry...")
    geometry = ranking_geometry(features, core.CHANNELS)

    # Freeze label-free calculations before opening the evaluation side.
    diagnostic = internal.join(geometry, how="outer")

    print("Computing labelled evaluation AFTER diagnostics...")
    external = labelled_channel_performance(features, truth, sensors)
    table = diagnostic.join(external, how="outer")
    table.insert(0, "dataset", config.name)

    out_dir = os.path.join(out_root, "Router_V1_Diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    safe_name = name.lower()
    table.to_csv(os.path.join(out_dir, f"{safe_name}_channel_diagnostics.csv"))

    abl = ablation_table(internal)
    abl.insert(0, "dataset", config.name)
    abl.to_csv(os.path.join(out_dir, f"{safe_name}_router_v1_ablation.csv"),
               index=False)

    corr = spearman_correlations(table)
    corr.insert(0, "dataset", config.name)
    corr.to_csv(os.path.join(out_dir, f"{safe_name}_correlations.csv"),
                index=False)

    print("\nChannel diagnostic table:")
    cols = [
        "reliability", "within_distance", "between_distance", "specificity",
        "labelled_median_origin_rank", "top1", "top3", "balanced_accuracy",
    ]
    print(table[cols].round(4).to_string(na_rep="not estimable"))

    print("\nRouter-v1 ablation winners (diagnostic, not a new selector):")
    winners = abl[abl["winner"]][["variant", "channel", "score"]]
    print(winners.round(4).to_string(index=False))

    print("\nKey Spearman checks:")
    key_corr = corr[
        (corr["predictor"].isin(["reliability", "specificity"])) &
        (corr["outcome"].isin(["balanced_accuracy",
                               "labelled_median_origin_rank"]))
    ]
    print(key_corr.round(4).to_string(index=False))
    print(f"\nWritten to {out_dir}/")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose why Router v1 rewards or rejects channels."
    )
    parser.add_argument("--dataset", choices=sorted(ds.DATASETS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("pass --dataset <name> or --all")

    names = sorted(ds.DATASETS) if args.all else [args.dataset]
    tables = []
    for name in names:
        tables.append(analyse_dataset(name, args.out))
        print()

    if tables:
        combined = pd.concat(tables, axis=0).reset_index().rename(
            columns={"index": "channel"}
        )
        out_dir = os.path.join(args.out, "Router_V1_Diagnostics")
        combined.to_csv(os.path.join(out_dir, "all_channel_diagnostics.csv"),
                        index=False)
        all_corr = spearman_correlations(combined)
        all_corr.to_csv(os.path.join(out_dir, "all_correlations.csv"), index=False)

        print("=" * 78)
        print("CROSS-DATASET CHECK")
        print("=" * 78)
        print(all_corr[
            (all_corr["predictor"].isin(["reliability", "specificity"])) &
            (all_corr["outcome"].isin(["balanced_accuracy",
                                       "labelled_median_origin_rank"]))
        ].round(4).to_string(index=False))

        with open(os.path.join(out_dir, "README_results.json"), "w") as f:
            json.dump({
                "purpose": "Diagnose Router v1; not tune Router v2",
                "specificity_definition": (
                    "mean between-scenario normalised Spearman rank distance "
                    "/ mean within-scenario run-to-consensus distance"
                ),
                "label_use": (
                    "No labels in internal metrics or ranking geometry. "
                    "Ground truth enters only in the external evaluation columns."
                ),
            }, f, indent=2)


if __name__ == "__main__":
    main()
