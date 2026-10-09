"""
attribution_core.py — THE FROZEN METHOD (v4.3), DATASET-AGNOSTIC
================================================================
One implementation of the attribution method. TEP, CSTR and the three-tank
system all run THIS code. Nothing here knows which plant it is on.

A change made here changes all three datasets. That is the point: the
scientific claim is that one frozen protocol transfers across plants, and
that claim is only testable if there is literally one copy of the protocol.

WHAT IS FROZEN (do not edit these without re-running every dataset)
-------------------------------------------------------------------
  - the five evidence channels and their hyperparameters
  - the adaptive per-sensor threshold (0.999 quantile of that sensor's own
    normal |z|)
  - the inference-free audit, including the v4.3 no-evidence rule
  - rank evidence with median aggregation across runs
  - the posterior form and the abstention rule

WHAT A DATASET MAY CHOOSE (declared in datasets.py, never here)
---------------------------------------------------------------
  - the baseline scheme (robust or shrunk)
  - which sensors form the "actuator" group for the Model 2 prior
  - the tier thresholds, which scale with the number of candidate sensors
  - the ground truth

THE PIPELINE
------------
    errors -> baseline -> |z| trajectories -> thresholds -> 5 channels
           -> audit -> tiers -> rank evidence -> prior -> posterior
           -> ranked diagnosis
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

# ──────────────────────────────────────────────────────────────────────────
# FROZEN CONSTANTS — DO NOT EDIT
# ──────────────────────────────────────────────────────────────────────────

NORMAL_FA_QUANTILE = 0.999   # per-sensor threshold = this quantile of the
                             # sensor's OWN normal |z|, so every sensor has
                             # the same false-crossing rate under normal
                             # operation. A fixed z=3 line does not have this
                             # property and manufactures hub sensors.
ONSET_SUSTAIN_K = 3          # consecutive samples required to call a crossing
RISE_SMOOTH_WINDOW = 5       # rolling-mean window before the slope estimate
EARLY_WINDOW_FRAC = 0.20     # "early" region for magnitude and rise_rate
MIN_SIGMA = 1e-9

CHANNELS: List[str] = ["magnitude", "peak", "persistence", "onset", "rise_rate"]

# The channel the TEP work pre-specified, evaluated on every plant alongside
# the audit's automatic choice so the two can be compared on record.
PRESPECIFIED_CHANNEL = "onset"


# ──────────────────────────────────────────────────────────────────────────
# STAGE 1 — BASELINES
# ──────────────────────────────────────────────────────────────────────────

def robust_baseline(normal_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Per-sensor median and scaled-MAD sigma of the normal-operation error.

    Transient-resistant: a handful of extreme settling samples cannot move a
    median. This is the default for every plant now that all three build
    their errors from per-sensor autoencoders.
    """
    mu = normal_df.median()
    mad = (normal_df - mu).abs().median()
    sigma = (1.4826 * mad)
    sigma = sigma.where(sigma > 0, normal_df.std())
    return mu, sigma.clip(lower=MIN_SIGMA)


def shrunk_baseline(normal_df: pd.DataFrame,
                    weight: float = 0.5) -> Tuple[pd.Series, pd.Series]:
    """
    Per-sensor mean and std, each sigma shrunk toward the pooled median sigma.

    This was the TEP v4.3 scheme, justified when its errors came from one
    global autoencoder producing many exchangeable per-sensor estimates.
    Kept so those numbers can be reproduced exactly.
    """
    mu = normal_df.mean()
    sigma = normal_df.std().clip(lower=MIN_SIGMA)
    pooled = float(np.median(sigma.values))
    shrunk = np.sqrt((1 - weight) * sigma ** 2 + weight * pooled ** 2)
    return mu, shrunk.clip(lower=MIN_SIGMA)


BASELINE_SCHEMES: Dict[str, Callable] = {
    "robust": robust_baseline,
    "shrunk": shrunk_baseline,
}


def z_trajectory(run_df: pd.DataFrame, mu: pd.Series,
                 sigma: pd.Series) -> pd.DataFrame:
    """|z_t| per sensor per timestep — the raw material for every channel."""
    return ((run_df - mu) / sigma).abs()


def sensor_thresholds(pooled_normal_z: pd.DataFrame,
                      q: float = NORMAL_FA_QUANTILE) -> pd.Series:
    """Adaptive per-sensor threshold: the q-th quantile of pooled normal |z|."""
    return pooled_normal_z.quantile(q)


# ──────────────────────────────────────────────────────────────────────────
# STAGE 2 — THE FIVE EVIDENCE CHANNELS
# Each returns a Series indexed by sensor. Bigger = more evidence of origin.
# ──────────────────────────────────────────────────────────────────────────

def channel_magnitude(zdf: pd.DataFrame, thresholds: pd.Series) -> pd.Series:
    """Mean |z| over the early window — the plain magnitude baseline."""
    n_early = max(1, int(len(zdf) * EARLY_WINDOW_FRAC))
    return zdf.iloc[:n_early].mean()


def channel_peak(zdf: pd.DataFrame, thresholds: pd.Series) -> pd.Series:
    """Largest |z| reached at any point in the run."""
    return zdf.max()


def channel_persistence(zdf: pd.DataFrame, thresholds: pd.Series) -> pd.Series:
    """Fraction of the run spent beyond the sensor's own threshold."""
    return (zdf > thresholds).mean()


def channel_onset(zdf: pd.DataFrame, thresholds: pd.Series) -> pd.Series:
    """
    Earliness of the first SUSTAINED threshold crossing, scored 1 - t/n.

    A sensor that never crosses for ONSET_SUSTAIN_K consecutive samples
    scores exactly 0. Deliberately harsh: an origin that never leaves normal
    is not attributable by this channel, and the audit reports that honestly
    rather than hiding it in a tie.
    """
    n = len(zdf)
    scores = {}
    for sensor in zdf.columns:
        above = (zdf[sensor].values > thresholds[sensor]).astype(int)
        if ONSET_SUSTAIN_K > 1:
            kernel = np.ones(ONSET_SUSTAIN_K, dtype=int)
            sustained = np.convolve(above, kernel, mode="valid") == ONSET_SUSTAIN_K
            idx = int(np.argmax(sustained)) if sustained.any() else -1
        else:
            idx = int(np.argmax(above)) if above.any() else -1
        scores[sensor] = 0.0 if idx < 0 else 1.0 - idx / max(n - 1, 1)
    return pd.Series(scores)


def channel_rise_rate(zdf: pd.DataFrame, thresholds: pd.Series) -> pd.Series:
    """
    Maximum positive slope of the smoothed |z| trajectory in the early window.

    A true origin under closed-loop control typically shows the sharpest
    initial excursion, before controller compensation and propagation blur
    the picture across coupled sensors.
    """
    n = len(zdf)
    n_early = min(max(RISE_SMOOTH_WINDOW + 2,
                      int(n * max(EARLY_WINDOW_FRAC, 0.33))), n)
    scores = {}
    for sensor in zdf.columns:
        smoothed = (pd.Series(zdf[sensor].values[:n_early])
                    .rolling(RISE_SMOOTH_WINDOW, min_periods=1).mean().values)
        grad = np.gradient(smoothed) if len(smoothed) > 1 else np.array([0.0])
        scores[sensor] = float(max(grad.max(), 0.0))
    return pd.Series(scores)


CHANNEL_FUNCS: Dict[str, Callable[[pd.DataFrame, pd.Series], pd.Series]] = {
    "magnitude": channel_magnitude,
    "peak": channel_peak,
    "persistence": channel_persistence,
    "onset": channel_onset,
    "rise_rate": channel_rise_rate,
}


def extract_channels(z_by_run: Dict[object, pd.DataFrame],
                     thresholds: pd.Series) -> Dict[str, pd.DataFrame]:
    """
    Run every channel over every run of one scenario.

    Returns {channel: DataFrame(runs x sensors)}. Everything downstream
    reuses this, so the expensive timestep work happens exactly once.
    """
    out = {}
    for channel in CHANNELS:
        rows = {run_id: CHANNEL_FUNCS[channel](zdf, thresholds)
                for run_id, zdf in z_by_run.items()}
        out[channel] = pd.DataFrame(rows).T if rows else pd.DataFrame()
    return out


# ──────────────────────────────────────────────────────────────────────────
# STAGE 3 — INFERENCE-FREE AUDIT
# Answers "does the information exist?" before "can we extract it?".
# ──────────────────────────────────────────────────────────────────────────

def true_rank_in_run(run_values: pd.Series,
                     valid_origins: Sequence[str]) -> float:
    """
    Rank of the best-placed valid origin in one run, or NaN if no sensor
    produced any evidence at all.

    THREE CASES, and the third is the one that matters:

    1. No sensor produced evidence. The run contributes NOTHING to the rank
       statistics. Under a naive min-rank every sensor would tie at rank 1
       and silently inflate subtle faults (the v4.3 fix).

    2. The true cause crossed. Its ordinary rank among the crossers.

    3. The true cause did NOT cross while other sensors did. It is tied with
       every other non-crossing sensor, so it gets that group's AVERAGE rank:

           n_crossers + (n_non_crossers + 1) / 2

       The earlier v4.3 code gave it n_crossers + 1 — the BEST rank in the
       tied group rather than the expected one. On a plant where typically
       one sensor crosses, a silent true cause therefore scored rank 2 out
       of 52, which reads as near-perfect attributability when in fact the
       sensor produced nothing at all. TEP faults 3, 9 and 15 sat in Tier A
       on that basis while their actual predictions came out at true rank
       48, 50 and 2.

       The average-rank convention is not a new choice: it is the same tie
       handling rank_evidence already uses (method="average"). This makes
       the audit internally consistent rather than introducing a parameter.
    """
    active = run_values > 0
    n_active = int(active.sum())
    if n_active == 0:
        return float("nan")

    n_total = len(run_values)
    n_silent = n_total - n_active
    # Expected rank of any member of the tied non-crossing group.
    silent_rank = n_active + (n_silent + 1) / 2.0

    ranked = run_values[active].rank(ascending=False, method="min")
    candidates = [
        float(ranked[origin]) if run_values.get(origin, 0) > 0 else silent_rank
        for origin in valid_origins if origin in run_values.index
    ]
    return float(min(candidates)) if candidates else float("nan")


def audit(features_by_scenario: Dict[object, Dict[str, pd.DataFrame]],
          truth: Dict[object, Sequence[str]]) -> pd.DataFrame:
    """
    Per scenario x channel: the median rank of the true origin across runs,
    plus the conditioning rates that say how trustworthy that median is.

    evidence_rate  fraction of runs where ANY sensor crossed
    true_cross_rate fraction where a valid origin crossed

    Low rates mean the median rank is conditional on rare events, which is
    exactly what the v4.3 fix exists to expose.
    """
    rows = []
    for key, per_channel in features_by_scenario.items():
        valid = list(truth[key])
        for channel in CHANNELS:
            fdf = per_channel.get(channel)
            if fdf is None or fdf.empty:
                continue
            ranks, n_no_evidence, n_true_active = [], 0, 0
            for _, run_values in fdf.iterrows():
                rank = true_rank_in_run(run_values, valid)
                if np.isnan(rank):
                    n_no_evidence += 1
                    continue
                if any(run_values.get(o, 0) > 0 for o in valid):
                    n_true_active += 1
                ranks.append(rank)
            n_runs = len(fdf)
            ranks_arr = np.array(ranks)
            rows.append(dict(
                scenario=key,
                true_cause="/".join(valid),
                channel=channel,
                median_rank=(float(np.median(ranks_arr)) if len(ranks_arr)
                             else np.nan),
                best_rank=(float(ranks_arr.min()) if len(ranks_arr) else np.nan),
                frac_top1=(float((ranks_arr == 1).mean()) if len(ranks_arr)
                           else 0.0),
                frac_top3=(float((ranks_arr <= 3).mean()) if len(ranks_arr)
                           else 0.0),
                evidence_rate=round(1.0 - n_no_evidence / n_runs, 3),
                true_cross_rate=round(n_true_active / n_runs, 3),
                n_runs=n_runs,
            ))
    return pd.DataFrame(rows)


MIN_CROSS_RATE = 0.5     # a scenario counts as having evidence when the true
                         # origin crossed in at least this fraction of runs
TIE_TOLERANCE = 1e-9     # channel medians this close are reported as tied


def channel_medians(audit_df: pd.DataFrame,
                    scenarios: Optional[Sequence] = None) -> pd.Series:
    """
    Median, across scenarios, of the true origin's median rank per channel.

    Restrict to a subset of scenarios by passing `scenarios`.
    """
    frame = audit_df
    if scenarios is not None:
        frame = frame[frame["scenario"].isin(list(scenarios))]
    if frame.empty:
        return pd.Series(np.nan, index=CHANNELS)
    return frame.groupby("channel")["median_rank"].median().reindex(CHANNELS)


def scenarios_with_evidence(audit_df: pd.DataFrame,
                            min_cross_rate: float = MIN_CROSS_RATE) -> List:
    """
    Scenarios where the true origin actually crossed its own threshold.

    WHY THIS MATTERS FOR COMPARING CHANNELS. onset and persistence are
    DISCRETE: they are exactly zero for a sensor that never crosses. magnitude,
    peak and rise_rate are CONTINUOUS and always return some value. On a
    scenario where the true origin never crosses, the discrete channels have
    no evidence BY CONSTRUCTION and their rank is a tie-group artifact, not a
    measurement.

    Pooling those scenarios into the channel comparison therefore measures one
    thing that could see the origin against another that could not, and the
    comparison says more about the fault than about the channels. The
    conditional comparison is the principled one; both are reported so the
    difference is visible rather than assumed.

    The crossing rate is taken from a discrete channel, since a continuous
    channel's crossing rate is always 1.0 and would select everything.
    """
    onset_rows = audit_df[audit_df["channel"] == PRESPECIFIED_CHANNEL]
    if onset_rows.empty:
        return sorted(audit_df["scenario"].unique(), key=str)
    keep = onset_rows[onset_rows["true_cross_rate"] >= min_cross_rate]
    return keep["scenario"].tolist()


def select_primary_channel(audit_df: pd.DataFrame,
                           scenarios: Optional[Sequence] = None
                           ) -> Tuple[str, pd.Series, List[str]]:
    """
    Audit-directed channel selection, unsupervised with respect to inference.

    Returns the chosen channel, every channel's median, and the full list of
    channels tied for best.

    TIES ARE REPORTED, NOT BROKEN SILENTLY. An earlier version took the first
    minimum, which on TEP resolved an exact 17.0/17.0 tie between persistence
    and onset by list order alone and printed it as a decision. When channels
    tie, the pre-specified channel is preferred — that is what pre-specifying
    is FOR — and the tie itself is surfaced to the caller.
    """
    medians = channel_medians(audit_df, scenarios)
    if medians.isna().all():
        return PRESPECIFIED_CHANNEL, medians, []

    best_value = medians.min()
    tied = [c for c in CHANNELS
            if pd.notna(medians[c]) and abs(medians[c] - best_value) <= TIE_TOLERANCE]
    if len(tied) > 1 and PRESPECIFIED_CHANNEL in tied:
        return PRESPECIFIED_CHANNEL, medians, tied
    return tied[0], medians, (tied if len(tied) > 1 else [])


def assign_tiers(audit_df: pd.DataFrame, tier_a_max: float,
                 tier_b_max: float) -> Dict[object, str]:
    """
    Attributability tier per scenario, from its best median rank over channels.

    Tier A: the information is clearly present. Tier B: marginal. Tier C: not
    attributable from this representation by ANY inference — a statement about
    the data and the ground-truth label, not about the inference layer.
    Thresholds are per-dataset because the candidate-set size differs.
    """
    best = audit_df.groupby("scenario")["median_rank"].min()
    best = best.fillna(np.inf)
    return {key: ("A" if r <= tier_a_max else ("B" if r <= tier_b_max else "C"))
            for key, r in best.items()}


# ──────────────────────────────────────────────────────────────────────────
# STAGE 4 — BAYESIAN ATTRIBUTION
# ──────────────────────────────────────────────────────────────────────────

def rank_evidence(channel_values: pd.DataFrame) -> pd.Series:
    """
    Scale-free rank evidence, aggregated across runs by the MEDIAN.

    Within each run, rank sensors descending and score log(n / rank). Summing
    across channels would be log rank-product fusion (Breitling et al. 2004):
    scale-free, robust to one channel failing, the discrete analogue of the
    paper's weighted geometric mean.

    The MEDIAN across runs is the statistic the audit demonstrates carries the
    signal. A mean lets zero-evidence runs drag a genuine origin below a
    mediocre-but-consistent early crosser.
    """
    n_sensors = channel_values.shape[1]
    ranks = channel_values.rank(axis=1, ascending=False, method="average")
    return np.log(n_sensors / ranks).median(axis=0)


def combine_channels(per_channel: Dict[str, pd.DataFrame],
                     combo: Sequence[str]) -> pd.Series:
    """Sum the rank evidence of several channels — log rank-product fusion."""
    total = None
    for channel in combo:
        fdf = per_channel.get(channel)
        if fdf is None or fdf.empty:
            continue
        evidence = rank_evidence(fdf)
        total = evidence if total is None else total.add(evidence,
                                                         fill_value=0.0)
    return total if total is not None else pd.Series(dtype=float)


def uniform_prior(sensors: Sequence[str]) -> pd.Series:
    """Model 1 prior: every sensor equally likely to be the origin."""
    return pd.Series(1.0 / len(sensors), index=list(sensors))


def lofo_group_prior(sensors: Sequence[str], group_members: Sequence[str],
                     scenario_key: object,
                     truth: Dict[object, Sequence[str]]) -> pd.Series:
    """
    Model 2 prior: one group of sensors (actuators, xmv variables, pumps)
    weighted by how often it holds the origin, learned LEAVE-ONE-OUT.

    The FORM predates the experiments; the WEIGHT is estimated without ever
    seeing the evaluated scenario's label. Applying it leave-one-out is what
    makes it reportable — a weight computed over all scenarios including the
    evaluated one leaks the label into the prior. The three-tank script did
    exactly that; here there is only one implementation, so it cannot happen
    on one plant and not another.
    """
    group = [s for s in sensors if s in group_members]
    rest = [s for s in sensors if s not in group_members]
    if not group or not rest:
        return uniform_prior(sensors)

    others = [origins[0] for key, origins in truth.items()
              if key != scenario_key and origins]
    if not others:
        return uniform_prior(sensors)

    rate_group = (sum(1 for o in others if o in group_members)
                  / len(others)) / len(group)
    rate_rest = (sum(1 for o in others if o not in group_members)
                 / len(others)) / len(rest)
    weight = max(rate_group / rate_rest, 1e-3) if rate_rest > 0 else 1.0
    prior = pd.Series({s: (weight if s in group_members else 1.0)
                       for s in sensors})
    return prior / prior.sum()


def posterior(evidence: pd.Series, prior: pd.Series) -> pd.Series:
    """Normalised posterior over sensors, computed in log space."""
    aligned_prior = prior.reindex(evidence.index).fillna(1e-300)
    log_unnorm = evidence.values + np.log(aligned_prior.values + 1e-300)
    return pd.Series(np.exp(log_unnorm - logsumexp(log_unnorm)),
                     index=evidence.index)


def is_abstention(post: pd.Series, tol: float = 1e-9) -> bool:
    """
    True when the posterior is numerically uniform.

    Reporting these as ABSTAIN rather than as an arbitrary tie-ordered sensor
    is the honest reading: the system found no evidence, and a tie-break would
    manufacture a prediction out of nothing. Applied on every plant.
    """
    if len(post) == 0:
        return True
    return bool(float(post.max()) <= 1.0 / len(post) + tol)


def diagnose(per_channel: Dict[str, pd.DataFrame], combo: Sequence[str],
             prior: pd.Series, top_k: int = 5) -> Optional[dict]:
    """
    Full attribution for one scenario: evidence, posterior, ranked shortlist.

    Returns None when no channel produced any evidence at all, which the
    caller reports rather than scoring as a wrong answer.
    """
    evidence = combine_channels(per_channel, combo)
    if evidence.empty:
        return None
    post = posterior(evidence, prior)
    ranked = post.sort_values(ascending=False)
    return dict(
        posterior=post,
        ranked=ranked.index.tolist(),
        top_k=ranked.index[:top_k].tolist(),
        top1=ranked.index[0],
        top1_prob=float(ranked.iloc[0]),
        abstained=is_abstention(post),
        entropy=normalised_entropy(post),
    )


def normalised_entropy(post: pd.Series) -> float:
    """Posterior entropy divided by its maximum — 1.0 means fully uniform."""
    values = post.values[post.values > 0]
    if len(values) == 0:
        return float("nan")
    return float(-(values * np.log(values)).sum() / np.log(len(post)))


# ──────────────────────────────────────────────────────────────────────────
# STAGE 5 — SCORING
# ──────────────────────────────────────────────────────────────────────────

def score_scenario(result: dict, valid_origins: Sequence[str]) -> dict:
    """
    Score one diagnosis against the ground truth.

    A scenario with several physically valid origins counts as correct if any
    of them is named, which is the only defensible reading when the fault
    equation genuinely couples two variables.
    """
    top_k = result["top_k"]
    return dict(
        top1=result["top1"],
        top1_prob=result["top1_prob"],
        abstained=result["abstained"],
        entropy=result["entropy"],
        correct_top1=bool(top_k and top_k[0] in valid_origins),
        correct_top3=bool(any(o in top_k[:3] for o in valid_origins)),
        correct_top5=bool(any(o in top_k[:5] for o in valid_origins)),
        true_rank=min((result["ranked"].index(o) + 1
                       for o in valid_origins if o in result["ranked"]),
                      default=np.nan),
        true_prob=float(max((result["posterior"].get(o, 0.0)
                             for o in valid_origins), default=np.nan)),
    )


def balanced_accuracy(records: Iterable[dict]) -> float:
    """
    Per-class recall averaged over distinct true-origin classes.

    The required selection metric: plain top-1 accuracy rewards a model that
    always names whichever sensor happens to cause the most faults.
    """
    by_class = defaultdict(list)
    for row in records:
        by_class[row["true_class"]].append(bool(row["correct_top1"]))
    if not by_class:
        return float("nan")
    return float(np.mean([np.mean(v) for v in by_class.values()]))


def chance_level(truth: Dict[object, Sequence[str]]) -> float:
    """Balanced-accuracy chance level: one over the number of distinct classes."""
    classes = {origins[0] for origins in truth.values() if origins}
    return 1.0 / len(classes) if classes else float("nan")


def summarise(results_df: pd.DataFrame, chance: float) -> dict:
    """Headline numbers for one configuration."""
    answered = results_df[~results_df["abstained"]]
    n_abstain = int(results_df["abstained"].sum())
    stats = dict(
        n_scenarios=len(results_df),
        top1=float(results_df["correct_top1"].mean()),
        top3=float(results_df["correct_top3"].mean()),
        top5=float(results_df["correct_top5"].mean()),
        balanced=balanced_accuracy(results_df.to_dict("records")),
        chance=chance,
        n_abstained=n_abstain,
        coverage=1.0 - n_abstain / len(results_df) if len(results_df) else 0.0,
        top1_answered=(float(answered["correct_top1"].mean())
                       if len(answered) else float("nan")),
        n_distinct=int(answered["top1"].nunique()) if len(answered) else 0,
    )
    if len(answered):
        dominant, count = Counter(answered["top1"]).most_common(1)[0]
        stats["dominant_sensor"] = dominant
        stats["dominant_share"] = count / len(answered)
    else:
        stats["dominant_sensor"] = None
        stats["dominant_share"] = float("nan")
    return stats


def calibration(results_df: pd.DataFrame) -> float:
    """
    Point-biserial correlation between top-1 confidence and correctness.

    The "knows when it doesn't know" property: the posterior should
    concentrate when it is right and stay flat when evidence is absent.
    """
    conf = results_df["top1_prob"].values
    correct = results_df["correct_top1"].astype(float).values
    if len(set(correct)) < 2 or np.std(conf) == 0:
        return float("nan")
    return float(np.corrcoef(conf, correct)[0, 1])


def selective_prediction(results_df: pd.DataFrame,
                         n_sensors: int) -> pd.DataFrame:
    """
    Accuracy when answering only above a confidence threshold.

    Reports coverage against accuracy-on-answered, which is the deployment
    property: a shortlist you can trust on the cases you choose to act on.
    """
    uniform = 1.0 / n_sensors
    rows = []
    for tau in [uniform, 2 * uniform, 4 * uniform, 0.10]:
        answered = results_df[results_df["top1_prob"] >= tau]
        rows.append(dict(
            tau=tau,
            coverage=len(answered) / len(results_df) if len(results_df) else 0.0,
            top1_on_answered=(float(answered["correct_top1"].mean())
                              if len(answered) else float("nan")),
            top3_on_answered=(float(answered["correct_top3"].mean())
                              if len(answered) else float("nan")),
        ))
    return pd.DataFrame(rows)