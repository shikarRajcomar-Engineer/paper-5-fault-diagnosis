"""
datasets.py — WHAT DIFFERS BETWEEN THE PLANTS
=============================================
Every plant-specific fact lives here and nowhere else. attribution_core.py
holds the method and knows nothing about TEP, CSTR or the three-tank system;
run_attribution.py drives it and reads its settings from this file.

If you find yourself writing "if dataset == ..." anywhere else, the fact
belongs in this file instead.

WHAT A DATASET DECLARES
-----------------------
  cache_dir        where the Stage-0 parquet files live
  scenario_cols    which cache columns identify one scenario
  truth_source     'column' (the cache carries true_cause) or 'table'
  truth_table      scenario -> list of valid origin sensors, when 'table'
  group_members    the sensor group the Model 2 prior up-weights
  tier_a_max       best median rank for tier A
  tier_b_max       best median rank for tier B
  baseline         'robust' (default) or 'shrunk' (reproduces TEP v4.3)

WHY TIER THRESHOLDS DIFFER
--------------------------
They scale with the candidate-set size. Rank 5 out of 52 sensors is the top
10%; rank 5 out of 5 is last place. Using one number everywhere would make
the tiers mean opposite things on different plants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ──────────────────────────────────────────────────────────────────────────
# GROUND TRUTH TABLES
# ──────────────────────────────────────────────────────────────────────────

# TEP — corrected against the published literature. Fault 1's root cause is
# xmeas_4, not xmeas_1; an earlier ChatGPT-sourced table had several of these
# wrong. Faults 16-20 have no agreed sensor-level origin and are omitted.
TEP_TRUTH: Dict[int, List[str]] = {
    1: ["xmeas_4"], 2: ["xmeas_4"], 3: ["xmeas_23"], 4: ["xmv_10"],
    5: ["xmeas_22"], 6: ["xmv_3"], 7: ["xmeas_4"], 8: ["xmeas_4"],
    9: ["xmeas_23"], 10: ["xmeas_4"], 11: ["xmv_10"], 12: ["xmv_11"],
    13: ["xmeas_9"], 14: ["xmv_10"], 15: ["xmv_11"], 21: ["xmeas_9"],
}

TEP_DESCRIPTIONS: Dict[int, str] = {
    1: "A/C feed ratio step", 2: "B composition step",
    3: "D feed temperature step", 4: "Reactor cooling water inlet temp",
    5: "Condenser cooling water inlet temp", 6: "A feed loss",
    7: "C header pressure loss", 8: "A,B,C feed composition variation",
    9: "D feed temperature random variation",
    10: "C feed temperature random variation",
    11: "Reactor cooling water valve random",
    12: "Condenser cooling water valve random",
    13: "Reaction kinetics slow drift",
    14: "Reactor cooling water valve drift",
    15: "Condenser cooling water valve drift", 21: "Valve position fixed",
}

# Three-tank — derived from the fault equations in the repo README. f1 enters
# both the h1 and h2 equations; f10 couples tanks 1 and 3 through Q13; f12
# couples tanks 3 and 2 through Q32. Those three genuinely have two valid
# origins and are scored as correct if either is named.
#
# fault 6 IS DISPUTED: the truth below says h3, but h3 shifts 1.0 units and
# varies 3.9x after onset while Q2 shifts 6.0 against a predicted 7.2 and
# varies 23.5x. The README lists faults 4-6 as "Sensor" faults without naming
# the sensor, so h3 was always an assumption. Left as h3 so results are not
# built on an unverified change; what the method says about fault 6 is
# independent evidence for revisiting it.
TTS_TRUTH: Dict[int, List[str]] = {
    1: ["h1", "h2"], 2: ["h1"], 3: ["h2"],
    4: ["h1"], 5: ["h2"], 6: ["h3"],
    7: ["h1"], 8: ["h2"], 9: ["h3"],
    10: ["h1", "h3"], 11: ["h2"], 12: ["h2", "h3"],
}

TTS_DISPUTED: Dict[int, str] = {
    6: "ground truth says h3; the data points at Q2",
}

TTS_DESCRIPTIONS: Dict[int, str] = {
    1: "Actuator, sinusoid on both inlets", 2: "Actuator ramp on Q1 path",
    3: "Actuator ramp on Q2 path", 4: "Sensor ramp (negative)",
    5: "Sensor ramp (positive)", 6: "Sensor ramp (positive, larger)",
    7: "Tank 1 leak", 8: "Tank 2 leak", 9: "Tank 3 leak",
    10: "Pipe 1-3 blockage", 11: "Tank 2 outflow blockage",
    12: "Pipe 3-2 blockage",
}


# ──────────────────────────────────────────────────────────────────────────
# DATASET DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """Everything the runner needs to know about one plant."""

    name: str
    cache_dir: str
    scenario_cols: List[str]
    truth_source: str                      # 'column' or 'table'
    group_members: List[str]               # up-weighted by the Model 2 prior
    group_label: str                       # what that group is called
    tier_a_max: float
    tier_b_max: float
    baseline: str = "robust"
    truth_table: Optional[Dict] = None
    descriptions: Dict = field(default_factory=dict)
    disputed: Dict = field(default_factory=dict)
    run_col: str = "simulationRun"
    sample_col: str = "sample"
    max_runs_per_scenario: Optional[int] = None

    def scenario_label(self, key) -> str:
        """Human-readable name for one scenario, used in reports."""
        if isinstance(key, tuple):
            return "|".join(str(k) for k in key)
        description = self.descriptions.get(key)
        return f"{key} ({description})" if description else str(key)


DATASETS: Dict[str, DatasetConfig] = {

    # ──────────────────────────────────────────────────────────────────
    # TENNESSEE EASTMAN
    # 52 sensors, ~500 runs per fault. One scenario per fault number.
    # The Model 2 prior up-weights the manipulated variables (xmv), whose
    # form predates these experiments.
    # ──────────────────────────────────────────────────────────────────
    "tep": DatasetConfig(
        name="TEP",
        cache_dir="TEP_cache",
        scenario_cols=["faultNumber"],
        truth_source="table",
        truth_table=TEP_TRUTH,
        descriptions=TEP_DESCRIPTIONS,
        # Resolved at load time against the sensors actually in the cache,
        # because the group is defined by a naming pattern, not a fixed list.
        group_members=["__PREFIX__xmv"],
        group_label="manipulated variables (xmv)",
        # 52 candidates: rank 5 is the top 10%, rank 15 the top 29%.
        tier_a_max=5,
        tier_b_max=15,
        baseline="robust",
        max_runs_per_scenario=100,
    ),

    # ──────────────────────────────────────────────────────────────────
    # CSTR
    # 7 sensors, one file per simulation. A scenario is an origin at an
    # amplitude, so a low-severity fault is never pooled with a
    # high-severity one sharing its origin.
    # ──────────────────────────────────────────────────────────────────
    "cstr": DatasetConfig(
        name="CSTR",
        cache_dir="CSTR_cache",
        scenario_cols=["true_cause", "amplitude"],
        truth_source="column",
        group_members=["Qc", "Tci"],
        group_label="actuator-side variables",
        # 7 candidates: rank 2 is the top 29%, rank 3 the top 43%.
        tier_a_max=2,
        tier_b_max=3,
        baseline="robust",
    ),

    # ──────────────────────────────────────────────────────────────────
    # THREE-TANK (DTS200)
    # 5 sensors, one run per fault. Three faults have two valid origins.
    # ──────────────────────────────────────────────────────────────────
    "tts": DatasetConfig(
        name="three_tank",
        cache_dir="TTS_cache",
        scenario_cols=["faultNumber"],
        truth_source="table",
        truth_table=TTS_TRUTH,
        descriptions=TTS_DESCRIPTIONS,
        disputed=TTS_DISPUTED,
        group_members=["Q1", "Q2"],
        group_label="pump actuators",
        # 5 candidates: rank 2 is the top 40%, rank 3 the top 60%.
        tier_a_max=2,
        tier_b_max=3,
        baseline="robust",
    ),
}


def resolve_group_members(config: DatasetConfig,
                          sensors: Sequence[str]) -> List[str]:
    """
    Turn a group definition into the actual sensor names present in the cache.

    TEP declares its group by prefix because the list is long and the exact
    sensor set depends on what the cache holds. The other plants name their
    members outright. Either way the result is checked against the cache, so
    a typo becomes an empty group and a visible warning rather than a silent
    fallback to a uniform prior.
    """
    members = []
    for entry in config.group_members:
        if entry.startswith("__PREFIX__"):
            prefix = entry.replace("__PREFIX__", "")
            members.extend([s for s in sensors if s.startswith(prefix)])
        elif entry in sensors:
            members.append(entry)
    return sorted(set(members))


def get(name: str) -> DatasetConfig:
    """Look up one dataset, with a helpful error listing the alternatives."""
    key = name.lower()
    if key not in DATASETS:
        raise SystemExit(
            f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    return DATASETS[key]