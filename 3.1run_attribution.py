"""
run_attribution.py — RUN THE FROZEN METHOD ON ANY DATASET
=========================================================
One driver. Reads a Stage-0 cache, applies the frozen method from
attribution_core.py, and writes results.

    python run_attribution.py --dataset tep
    python run_attribution.py --dataset cstr
    python run_attribution.py --dataset tts

    # all three in one go, with a comparison table at the end
    python run_attribution.py --all

WHAT IT DOES, IN ORDER
    1. load the cache and check its integrity (duplicate scenarios)
    2. fit the normal baseline and the adaptive per-sensor thresholds
    3. build |z| trajectories and extract the five evidence channels
    4. run the inference-free audit and assign attributability tiers
    5. select the primary channel from the audit, and separately evaluate
       the pre-specified channel, so the two are compared on record
    6. attribute with a uniform prior (Model 1) and the leave-one-out
       group prior (Model 2)
    7. report accuracy, balanced accuracy, abstention, calibration and
       selective prediction, then write CSV and JSON

WHAT IT DELIBERATELY DOES NOT DO
    No tuning. No per-dataset special cases. Every plant runs the same code
    with the settings declared in datasets.py, so a difference in results is
    a difference between the plants and not between three scripts that
    drifted apart.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import attribution_core as core
import datasets as ds


# ──────────────────────────────────────────────────────────────────────────
# 1. LOADING THE CACHE
# ──────────────────────────────────────────────────────────────────────────

def load_cache(config: ds.DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                  dict]:
    """Read the two parquet files a Stage-0 builder produced."""
    normal_path = os.path.join(config.cache_dir, "normal_error_df.parquet")
    fault_path = os.path.join(config.cache_dir, "fault_error_df.parquet")
    for path in (normal_path, fault_path):
        if not os.path.exists(path):
            raise SystemExit(
                f"Cache file not found: {path}\n"
                f"  Build it first with the Stage-0 builder for "
                f"{config.name}.")

    normal_df = pd.read_parquet(normal_path)
    fault_df = pd.read_parquet(fault_path)

    metadata = {}
    meta_path = os.path.join(config.cache_dir, "cache_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            metadata = json.load(handle)

    print(f"  normal_error_df {normal_df.shape}")
    print(f"  fault_error_df  {fault_df.shape}")
    if metadata:
        print(f"  built {metadata.get('built_utc', '?')} | "
              f"{metadata.get('representation', '?')}")
    return normal_df, fault_df, metadata


def identify_sensors(normal_df: pd.DataFrame, fault_df: pd.DataFrame,
                     config: ds.DatasetConfig) -> List[str]:
    """
    Work out which columns are sensors, as the intersection of both caches.

    Anything that is a label rather than a measurement is excluded by name.
    Taking the intersection means a column present in one cache but not the
    other cannot silently become a phantom sensor with no baseline.
    """
    label_cols = {config.run_col, config.sample_col, "faultNumber",
                  "true_cause", "amplitude", "fault_type", "Class"}
    fault_sensors = [c for c in fault_df.columns if c not in label_cols]
    normal_sensors = [c for c in normal_df.columns if c not in label_cols]

    sensors = [c for c in fault_sensors if c in normal_sensors]
    dropped = set(fault_sensors) ^ set(normal_sensors)
    if dropped:
        print(f"  WARNING: columns in only one cache, excluded: "
              f"{sorted(dropped)}")
    if not sensors:
        raise SystemExit("No sensor columns common to both caches.")
    print(f"  {len(sensors)} sensors: {sensors[:6]}"
          f"{'...' if len(sensors) > 6 else ''}")
    return sensors


# ──────────────────────────────────────────────────────────────────────────
# 2. BUILDING SCENARIOS
# ──────────────────────────────────────────────────────────────────────────

def scenario_key(row, config: ds.DatasetConfig):
    """One scenario's identity, as a scalar or a tuple of the key columns."""
    if len(config.scenario_cols) == 1:
        return row[config.scenario_cols[0]]
    return tuple(row[c] for c in config.scenario_cols)


def build_scenarios(fault_df: pd.DataFrame, sensors: Sequence[str],
                    config: ds.DatasetConfig
                    ) -> Tuple[Dict, Dict[object, List[str]]]:
    """
    Split the fault cache into scenarios, each holding its runs in order.

    A scenario is the unit the method aggregates over. For CSTR it is an
    origin at an amplitude, so a 5% fault is never pooled with a 20% one
    sharing its origin — severity is evidence, and pooling would let the
    strong case mask the weak one.

    Runs within a scenario are kept separate and sorted by sample, because
    the onset channel measures position within a run.
    """
    missing = [c for c in config.scenario_cols if c not in fault_df.columns]
    if missing:
        raise SystemExit(
            f"Cache is missing scenario column(s) {missing}. "
            f"Found: {list(fault_df.columns)}")

    scenarios: Dict[object, Dict[object, pd.DataFrame]] = {}
    truth: Dict[object, List[str]] = {}

    grouped = fault_df.groupby(config.scenario_cols, sort=True)
    for key, block in grouped:
        if isinstance(key, tuple) and len(config.scenario_cols) == 1:
            key = key[0]

        origins = resolve_truth(key, block, config)
        if origins is None:
            print(f"    scenario {key}: no ground truth, skipped")
            continue
        valid = [o for o in origins if o in sensors]
        if not valid:
            print(f"    scenario {key}: origin(s) {origins} not among the "
                  f"cache sensors, skipped")
            continue

        runs = split_runs(block, sensors, config)
        if not runs:
            print(f"    scenario {key}: no usable runs, skipped")
            continue
        scenarios[key] = runs
        truth[key] = valid

    if not scenarios:
        raise SystemExit("No usable scenarios found in the cache.")
    print(f"  {len(scenarios)} scenarios, "
          f"{sum(len(r) for r in scenarios.values())} runs total")
    return scenarios, truth


def resolve_truth(key, block: pd.DataFrame,
                  config: ds.DatasetConfig) -> Optional[List[str]]:
    """
    Find the valid origin sensors for one scenario.

    Two sources: a column the Stage-0 builder wrote into the cache, or a
    table in datasets.py. A cache column carrying several origins separated
    by "/" is split, which is how the three-tank dual-origin faults survive
    the round trip.
    """
    if config.truth_source == "column":
        values = block["true_cause"].dropna().unique()
        if len(values) == 0:
            return None
        return str(values[0]).split("/")

    if config.truth_table is None:
        return None
    lookup = int(key) if not isinstance(key, tuple) else key
    origins = config.truth_table.get(lookup)
    return list(origins) if origins else None


def split_runs(block: pd.DataFrame, sensors: Sequence[str],
               config: ds.DatasetConfig) -> Dict[object, pd.DataFrame]:
    """Split one scenario into its runs, sorted by sample within each."""
    runs: Dict[object, pd.DataFrame] = {}
    if config.run_col in block.columns:
        groups = block.groupby(config.run_col, sort=True)
    else:
        groups = [(0, block)]

    for run_id, run_block in groups:
        if config.sample_col in run_block.columns:
            run_block = run_block.sort_values(config.sample_col)
        frame = run_block[list(sensors)].reset_index(drop=True)
        if len(frame) > 1:
            runs[run_id] = frame

    if config.max_runs_per_scenario is not None:
        keep = sorted(runs)[:config.max_runs_per_scenario]
        runs = {k: runs[k] for k in keep}
    return runs


def check_duplicate_scenarios(scenarios: Dict) -> None:
    """
    Warn when two scenarios contain identical data.

    Duplicated data invalidates any comparison between the affected
    scenarios and inflates accuracy by counting one case twice. The
    three-tank 7-column files had exactly this problem.
    """
    import hashlib

    prints = {}
    for key, runs in scenarios.items():
        first = runs[sorted(runs)[0]]
        prints[key] = hashlib.md5(
            np.round(first.values[:200], 6).tobytes()).hexdigest()

    seen: Dict[str, List] = {}
    for key, digest in prints.items():
        seen.setdefault(digest, []).append(key)
    for group in seen.values():
        if len(group) > 1:
            print(f"  *** DATA-INTEGRITY WARNING: scenarios {group} contain "
                  f"IDENTICAL data. Any comparison between them is "
                  f"meaningless and accuracy over all of them counts one "
                  f"case more than once. ***")


# ──────────────────────────────────────────────────────────────────────────
# 3. THE PIPELINE
# ──────────────────────────────────────────────────────────────────────────

def fit_baseline(normal_df: pd.DataFrame, sensors: Sequence[str],
                 config: ds.DatasetConfig) -> Tuple[pd.Series, pd.Series,
                                                    pd.Series]:
    """
    Fit the normal baseline and the adaptive per-sensor thresholds.

    The scheme is declared per dataset. Robust (median / scaled MAD) is the
    default everywhere now that all three plants build their errors from
    per-sensor autoencoders; shrunk reproduces the earlier TEP numbers,
    where the errors came from one global autoencoder.
    """
    scheme = core.BASELINE_SCHEMES[config.baseline]
    mu, sigma = scheme(normal_df[list(sensors)])
    normal_z = core.z_trajectory(normal_df[list(sensors)], mu, sigma)
    thresholds = core.sensor_thresholds(normal_z)
    print(f"  baseline: {config.baseline} | thresholds at the "
          f"{100 * core.NORMAL_FA_QUANTILE:.1f}th percentile of normal |z| "
          f"(min {thresholds.min():.2f}, median {thresholds.median():.2f}, "
          f"max {thresholds.max():.2f})")
    return mu, sigma, thresholds


def extract_all_features(scenarios: Dict, mu: pd.Series, sigma: pd.Series,
                         thresholds: pd.Series) -> Dict:
    """Channel features for every scenario, computed once and reused."""
    features = {}
    for key, runs in scenarios.items():
        z_by_run = {rid: core.z_trajectory(rdf, mu, sigma)
                    for rid, rdf in runs.items()}
        features[key] = core.extract_channels(z_by_run, thresholds)
    return features


def evaluate(features: Dict, truth: Dict, sensors: Sequence[str],
             combo: Sequence[str], prior_mode: str,
             group_members: Sequence[str], tiers: Dict) -> pd.DataFrame:
    """
    Attribute every scenario with one channel combination and one prior.

    prior_mode 'uniform' is Model 1; 'group' is Model 2, whose weight is
    learned leave-one-out so the evaluated scenario's own label never enters
    its own prior.
    """
    rows = []
    for key, per_channel in features.items():
        valid = truth[key]
        if prior_mode == "group":
            prior = core.lofo_group_prior(sensors, group_members, key, truth)
        else:
            prior = core.uniform_prior(sensors)

        result = core.diagnose(per_channel, combo, prior)
        if result is None:
            print(f"    scenario {key}: no evidence on any channel, skipped")
            continue
        scored = core.score_scenario(result, valid)
        scored.update(scenario=key, true_cause="/".join(valid),
                      true_class=valid[0], tier=tiers.get(key, "?"))
        rows.append(scored)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# 4. REPORTING
# ──────────────────────────────────────────────────────────────────────────

def print_audit(audit_df: pd.DataFrame, config: ds.DatasetConfig) -> None:
    """Show the audit as a scenario x channel table of true-origin ranks."""
    print("\n  Median rank of the true origin, per channel "
          "(1 = best, NaN = no run produced evidence):")
    pivot = audit_df.pivot_table(index="scenario", columns="channel",
                                 values="median_rank")
    pivot = pivot[[c for c in core.CHANNELS if c in pivot.columns]]
    print(pivot.to_string())

    onset = audit_df[audit_df["channel"] == core.PRESPECIFIED_CHANNEL]
    if not onset.empty:
        print(f"\n  Conditioning on the {core.PRESPECIFIED_CHANNEL} channel "
              f"(low rates mean the ranks above rest on rare events):")
        print(onset[["scenario", "evidence_rate", "true_cross_rate",
                     "n_runs"]].to_string(index=False))


def print_channel_choice(audit_df: pd.DataFrame) -> Tuple[str, dict]:
    """
    Report the audit's channel choice, unconditionally and conditioned on
    evidence, against the pre-specified channel.

    The conditional comparison is the primary one. Discrete channels cannot
    rank a sensor that never crosses, so scenarios where the true origin is
    silent measure the fault's attributability rather than the channel's
    quality. Both are printed so the gap between them is visible.
    """
    with_evidence = core.scenarios_with_evidence(audit_df)
    all_scenarios = sorted(audit_df["scenario"].unique(), key=str)
    n_all, n_ok = len(all_scenarios), len(with_evidence)

    all_medians = core.channel_medians(audit_df)
    print(f"\n  Median of per-scenario median-rank, by channel.")
    print(f"    ALL {n_all} scenarios          : " + "   ".join(
        f"{c}={all_medians[c]:.1f}" if pd.notna(all_medians[c]) else f"{c}=NaN"
        for c in core.CHANNELS))

    if n_ok == 0:
        print("    No scenario had the true origin crossing in at least "
              f"{100 * core.MIN_CROSS_RATE:.0f}% of runs; the conditional\n"
              "    comparison is undefined and the unconditional one is used.")
        auto, medians, tied = core.select_primary_channel(audit_df)
    else:
        ok_medians = core.channel_medians(audit_df, with_evidence)
        print(f"    {n_ok} with true origin crossing : " + "   ".join(
            f"{c}={ok_medians[c]:.1f}" if pd.notna(ok_medians[c]) else f"{c}=NaN"
            for c in core.CHANNELS))
        auto, medians, tied = core.select_primary_channel(audit_df,
                                                          with_evidence)

    if n_ok < n_all:
        excluded = [s for s in all_scenarios if s not in with_evidence]
        print(f"\n    Excluded from the conditional comparison: {excluded}")
        print(f"    In these the true origin crossed in under "
              f"{100 * core.MIN_CROSS_RATE:.0f}% of runs, so the discrete "
              f"channels\n    (onset, persistence) have no evidence by "
              f"construction and their rank there is a\n    tie-group "
              f"artifact rather than a measurement. Their attributability is "
              f"reported\n    by the tiers below, not by the channel "
              f"comparison.")

    if tied:
        print(f"\n    TIE: {', '.join(tied)} are equal at "
              f"{medians[tied[0]]:.1f}. Resolved in favour of the "
              f"pre-specified\n    channel where it is among them; this is "
              f"what pre-specification is for.")

    print(f"\n    audit selects (conditional) : {auto}")
    print(f"    pre-specified               : {core.PRESPECIFIED_CHANNEL}")
    if auto == core.PRESPECIFIED_CHANNEL:
        print("    AGREEMENT - the automatic choice matches the "
              "pre-specification.")
    else:
        print(f"    DISAGREEMENT - report this as a finding for this plant. "
              f"Both are\n    evaluated below; do not retune on the basis of "
              f"which scores better.")

    detail = dict(
        auto_channel=auto,
        tied_channels=tied,
        n_scenarios_all=n_all,
        n_scenarios_with_evidence=n_ok,
        medians_all={c: (None if pd.isna(all_medians[c])
                         else float(all_medians[c])) for c in core.CHANNELS},
        medians_with_evidence=({c: (None if pd.isna(v) else float(v))
                                for c, v in
                                core.channel_medians(audit_df,
                                                     with_evidence).items()}
                               if n_ok else {}),
    )
    return auto, detail


def print_results(results: pd.DataFrame, label: str, chance: float,
                  n_sensors: int, config: ds.DatasetConfig) -> dict:
    """Print the headline numbers and per-tier breakdown for one run."""
    stats = core.summarise(results, chance)
    print(f"\n  [{label}]")
    print(f"    top-1 {100 * stats['top1']:.1f}%   "
          f"top-3 {100 * stats['top3']:.1f}%   "
          f"balanced {100 * stats['balanced']:.1f}%   "
          f"(chance {100 * chance:.1f}%)")
    if stats["n_abstained"]:
        print(f"    abstained {stats['n_abstained']}/{stats['n_scenarios']}"
              f"   coverage {100 * stats['coverage']:.0f}%   "
              f"top-1 on answered {100 * stats['top1_answered']:.1f}%")
    print(f"    diversity {stats['n_distinct']} distinct predictions"
          + (f" (dominant {stats['dominant_sensor']} @ "
             f"{100 * stats['dominant_share']:.0f}%)"
             if stats["dominant_sensor"] else ""))
    if stats["dominant_share"] == stats["dominant_share"] \
            and stats["dominant_share"] > 0.4:
        print(f"    WARNING: one sensor dominates - hub-bias symptom, "
              f"inspect before reporting.")

    for tier in ["A", "B", "C"]:
        sub = results[results["tier"] == tier]
        if len(sub) == 0:
            continue
        print(f"    Tier {tier} (n={len(sub)}): "
              f"top-1 {100 * sub['correct_top1'].mean():.1f}%   "
              f"top-3 {100 * sub['correct_top3'].mean():.1f}%")
    return stats


def print_per_scenario(results: pd.DataFrame,
                       config: ds.DatasetConfig) -> None:
    """One line per scenario, so individual cases can be inspected."""
    print("\n  Per scenario:")
    for row in results.itertuples(index=False):
        mark = "OK " if row.correct_top1 else (
            "top3" if row.correct_top3 else "X  ")
        abstain = "  ABSTAIN" if row.abstained else ""
        note = ""
        if not isinstance(row.scenario, tuple):
            disputed = config.disputed.get(row.scenario)
            if disputed:
                note = f"   DISPUTED: {disputed}"
        print(f"    {config.scenario_label(row.scenario):<42} "
              f"true {row.true_cause:<12} -> {row.top1:<12} "
              f"P={row.top1_prob:.3f}  rank {row.true_rank}  "
              f"{mark}{abstain}{note}")


# ──────────────────────────────────────────────────────────────────────────
# 5. ONE DATASET, END TO END
# ──────────────────────────────────────────────────────────────────────────

def run_dataset(name: str, out_root: str = ".",
                baseline_override: Optional[str] = None) -> dict:
    """Run the frozen method on one dataset and write its results."""
    config = ds.get(name)
    if baseline_override:
        config.baseline = baseline_override

    print("=" * 74)
    print(f"{config.name} — FROZEN ATTRIBUTION METHOD")
    print("=" * 74)

    normal_df, fault_df, metadata = load_cache(config)
    sensors = identify_sensors(normal_df, fault_df, config)
    group_members = ds.resolve_group_members(config, sensors)
    print(f"  {config.group_label}: {len(group_members)} sensor(s)"
          + (f" {group_members}" if len(group_members) <= 8 else ""))
    if not group_members:
        print("  WARNING: the group is empty, so Model 2 falls back to "
              "uniform.")

    print("\nBuilding scenarios...")
    scenarios, truth = build_scenarios(fault_df, sensors, config)
    check_duplicate_scenarios(scenarios)

    print("\nFitting baseline and thresholds...")
    mu, sigma, thresholds = fit_baseline(normal_df, sensors, config)

    print("\nExtracting channels...")
    features = extract_all_features(scenarios, mu, sigma, thresholds)
    print(f"  {len(core.CHANNELS)} channels over {len(features)} scenarios")

    print("\n" + "-" * 74)
    print("INFERENCE-FREE AUDIT")
    print("-" * 74)
    audit_df = core.audit(features, truth)
    print_audit(audit_df, config)
    auto_channel, channel_detail = print_channel_choice(audit_df)

    tiers = core.assign_tiers(audit_df, config.tier_a_max, config.tier_b_max)

    # The crossing rate must come from the channel that SET the tier, not the
    # maximum over channels. magnitude and peak are continuous and never
    # exactly zero, so they always report a 100% crossing rate; quoting that
    # next to a tier decided by onset would hide the very weakness this
    # column exists to expose.
    #
    # Scenarios where every channel is NaN produced no evidence anywhere and
    # have no deciding channel, so they are simply left out of this lookup.
    ranked = audit_df.dropna(subset=["median_rank"])
    tier_channel, cross_rates = {}, {}
    if not ranked.empty:
        best_rows = ranked.loc[ranked.groupby("scenario")["median_rank"].idxmin()]
        tier_channel = dict(zip(best_rows["scenario"], best_rows["channel"]))
        cross_rates = dict(zip(best_rows["scenario"],
                               best_rows["true_cross_rate"]))

    print(f"\n  Attributability tiers (A <= {config.tier_a_max}, "
          f"B <= {config.tier_b_max}). 'crossed' is the fraction of runs in "
          f"which the\n  true origin crossed its own threshold, on the "
          f"channel that set the tier:")
    for tier in ["A", "B", "C"]:
        members = sorted((k for k, t in tiers.items() if t == tier), key=str)
        if not members:
            continue
        shown = ", ".join(
            f"{config.scenario_label(m).split(' (')[0]}"
            f"[{tier_channel.get(m, '?')}, crossed "
            f"{100 * cross_rates.get(m, 0):.0f}%]" for m in members)
        print(f"    Tier {tier}: {shown}")

    weak = [m for m, t in tiers.items()
            if t in ("A", "B") and cross_rates.get(m, 1.0) < 0.5]
    if weak:
        print(f"\n    NOTE: these scenarios were tiered on a channel where "
              f"the true origin crossed\n    in under half of runs: "
              f"{sorted(weak, key=str)}. Their rank rests largely on the\n"
              f"    tied non-crossing group, so treat the tier as weak "
              f"evidence and check whether\n    their predictions actually "
              f"succeed.")
    print("  Tier C cannot be recovered by ANY inference on these features - "
          "a\n  representation or ground-truth limitation, not a method "
          "failure.")

    chance = core.chance_level(truth)
    print("\n" + "-" * 74)
    print(f"ATTRIBUTION   (chance-level balanced accuracy "
          f"{100 * chance:.1f}%)")
    print("-" * 74)

    all_stats = {}
    results_store = {}
    for channel in dict.fromkeys([core.PRESPECIFIED_CHANNEL, auto_channel]):
        for prior_mode, prior_label in [("uniform", "Model 1 uniform"),
                                        ("group", "Model 2 group prior")]:
            label = f"{channel} + {prior_label}"
            results = evaluate(features, truth, sensors, (channel,),
                               prior_mode, group_members, tiers)
            if results.empty:
                continue
            all_stats[label] = print_results(results, label, chance,
                                             len(sensors), config)
            results_store[label] = results

    primary_label = f"{core.PRESPECIFIED_CHANNEL} + Model 2 group prior"
    primary = results_store.get(primary_label)
    if primary is not None:
        print_per_scenario(primary, config)
        print(f"\n  Calibration: corr(top-1 confidence, correct) = "
              f"{core.calibration(primary):.2f}")
        print("  Selective prediction:")
        table = core.selective_prediction(primary, len(sensors))
        for row in table.itertuples(index=False):
            print(f"    tau {row.tau:.4f}: coverage "
                  f"{100 * row.coverage:>3.0f}%   "
                  f"top-1 on answered {100 * row.top1_on_answered:.1f}%")

    out_dir = os.path.join(out_root, f"{config.cache_dir}_results")
    os.makedirs(out_dir, exist_ok=True)
    audit_df.to_csv(os.path.join(out_dir, "audit.csv"), index=False)
    for label, results in results_store.items():
        safe = label.replace(" ", "_").replace("+", "and")
        results.to_csv(os.path.join(out_dir, f"results_{safe}.csv"),
                       index=False)

    summary = dict(
        dataset=config.name,
        run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_sensors=len(sensors),
        n_scenarios=len(features),
        chance_level=chance,
        baseline=config.baseline,
        audit_selected_channel=auto_channel,
        prespecified_channel=core.PRESPECIFIED_CHANNEL,
        channels_agree=auto_channel == core.PRESPECIFIED_CHANNEL,
        channel_comparison=channel_detail,
        tiers={str(k): v for k, v in tiers.items()},
        configurations=all_stats,
        cache_metadata=metadata,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"\n  Results written to {out_dir}/")
    return summary


def print_comparison(summaries: List[dict]) -> None:
    """
    Cross-dataset table — the point of the whole exercise.

    One frozen method, three plants. The comparison is only meaningful
    because every column below was produced by the same code.
    """
    print("\n" + "=" * 74)
    print("CROSS-DATASET COMPARISON (identical frozen method)")
    print("=" * 74)
    print(f"\n  {'dataset':<14} {'sensors':>8} {'scenarios':>10} "
          f"{'w/evidence':>11} {'chance':>8} {'channel':>12} {'agrees':>8}")
    for summary in summaries:
        detail = summary.get('channel_comparison', {})
        print(f"  {summary['dataset']:<14} {summary['n_sensors']:>8} "
              f"{summary['n_scenarios']:>10} "
              f"{detail.get('n_scenarios_with_evidence', '?'):>11} "
              f"{100 * summary['chance_level']:>7.1f}% "
              f"{summary['audit_selected_channel']:>12} "
              f"{str(summary['channels_agree']):>8}")

    print(f"\n  Channel medians on scenarios where the true origin crossed "
          f"(lower = better):")
    print(f"  {'dataset':<14}" + "".join(f"{c:>13}" for c in core.CHANNELS))
    for summary in summaries:
        med = summary.get('channel_comparison', {}).get(
            'medians_with_evidence', {})
        if not med:
            continue
        print(f"  {summary['dataset']:<14}" + "".join(
            f"{med.get(c):>13.1f}" if med.get(c) is not None else f"{'-':>13}"
            for c in core.CHANNELS))

    print(f"\n  {'dataset':<14} {'configuration':<34} {'top-1':>7} "
          f"{'top-3':>7} {'balanced':>9}")
    for summary in summaries:
        for label, stats in summary["configurations"].items():
            print(f"  {summary['dataset']:<14} {label:<34} "
                  f"{100 * stats['top1']:>6.1f}% {100 * stats['top3']:>6.1f}% "
                  f"{100 * stats['balanced']:>8.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen attribution method on a dataset.")
    parser.add_argument("--dataset", choices=sorted(ds.DATASETS),
                        help="which plant to run")
    parser.add_argument("--all", action="store_true",
                        help="run every dataset and compare them")
    parser.add_argument("--baseline", choices=sorted(core.BASELINE_SCHEMES),
                        default=None,
                        help="override the baseline scheme for this run")
    parser.add_argument("--out", default=".", help="where to write results")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("pass --dataset <name> or --all")

    names = sorted(ds.DATASETS) if args.all else [args.dataset]
    summaries = []
    for name in names:
        try:
            summaries.append(run_dataset(name, args.out, args.baseline))
        except SystemExit as exc:
            print(f"\n  {ds.get(name).name} skipped: {exc}\n")
        print()

    if len(summaries) > 1:
        print_comparison(summaries)


if __name__ == "__main__":
    main()