"""
===============================================================================
V2.5.2 — FAULT 15 SUB-THRESHOLD DIAGNOSTIC
===============================================================================

Research question:
Does the known Fault 15 origin (xmv_11) contain a consistent reconstruction-
error change that remains below the frozen V2.5 abnormality threshold?

This experiment DOES NOT modify V2.5.

V2.5.1 established:
    - xmv_11 sustained abnormal onset: 0 / 500 runs
    - median V2.5 rank: 52

V2.5.2 therefore examines the CONTINUOUS reconstruction-error trajectory
instead of only asking whether the 99th-percentile threshold is crossed.

Frozen V2.5 settings:
    Normal threshold quantile = 0.99
    Sustained onset           = 3 samples

No:
    - new attribution score
    - new threshold
    - fault-specific tuning
    - ranking modification

Ground truth is used only to identify xmv_11 for diagnostic investigation.
===============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

BASE = Path(
    "/content/drive/MyDrive/PHD/"
    "Router_V2_Structural_Attribution"
)

NORMAL_FILE = (
    BASE /
    "TEP_cache" /
    "normal_error_df.parquet"
)

FAULT_FILE = (
    BASE /
    "TEP_cache" /
    "fault_error_df.parquet"
)

OUTPUT_DIR = (
    BASE /
    "V2_5_2_Fault15_Subthreshold_Diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FAULT = 15
TRUE_ORIGIN = "xmv_11"

QUANTILE = 0.99


# =============================================================================
# 2. LOAD ONLY THE DATA WE NEED
# =============================================================================

print("=" * 100)
print("V2.5.2 — FAULT 15 SUB-THRESHOLD DIAGNOSTIC")
print("=" * 100)

print("\nLoading xmv_11 normal reconstruction errors...")

normal = pd.read_parquet(
    NORMAL_FILE,
    columns=[TRUE_ORIGIN]
)

print(
    f"Normal rows: {len(normal):,}"
)


print("\nLoading Fault 15 xmv_11 reconstruction errors...")

fault = pd.read_parquet(
    FAULT_FILE,
    columns=[
        "faultNumber",
        "simulationRun",
        "sample",
        TRUE_ORIGIN
    ]
)

fault = fault[
    fault["faultNumber"] == FAULT
].copy()

fault = fault.sort_values(
    [
        "simulationRun",
        "sample"
    ]
)

print(
    f"Fault 15 rows: {len(fault):,}"
)

print(
    f"Fault 15 runs: "
    f"{fault['simulationRun'].nunique()}"
)


# =============================================================================
# 3. FROZEN V2.5 THRESHOLD
# =============================================================================

normal_values = (
    normal[TRUE_ORIGIN]
    .dropna()
)

threshold = float(
    normal_values.quantile(
        QUANTILE
    )
)


print("\n" + "=" * 100)
print("1. NORMAL xmv_11 RECONSTRUCTION ERROR")
print("=" * 100)

print(
    f"\nMedian : "
    f"{normal_values.median():.10f}"
)

print(
    f"Mean   : "
    f"{normal_values.mean():.10f}"
)

print(
    f"Std    : "
    f"{normal_values.std():.10f}"
)

print(
    f"95%    : "
    f"{normal_values.quantile(0.95):.10f}"
)

print(
    f"99%    : "
    f"{normal_values.quantile(0.99):.10f}"
)

print(
    f"Max    : "
    f"{normal_values.max():.10f}"
)

print(
    f"\nFrozen V2.5 threshold: "
    f"{threshold:.10f}"
)


# =============================================================================
# 4. NORMALISE FAULT ERROR BY THE FROZEN THRESHOLD
#
# Ratio:
#
#       1.0 = exactly at V2.5 threshold
#       0.5 = half the threshold
#       2.0 = twice the threshold
#
# This does NOT change the attribution method.
# =============================================================================

fault[
    "threshold_ratio"
] = (
    fault[TRUE_ORIGIN]
    /
    threshold
)


# =============================================================================
# 5. SAMPLE-BY-SAMPLE SUMMARY ACROSS ALL 500 RUNS
#
# This allows us to see whether xmv_11 changes consistently with time.
# =============================================================================

trajectory = (
    fault
    .groupby(
        "sample"
    )
    .agg(
        median_error=(
            TRUE_ORIGIN,
            "median"
        ),

        mean_error=(
            TRUE_ORIGIN,
            "mean"
        ),

        q25_error=(
            TRUE_ORIGIN,
            lambda x:
                x.quantile(0.25)
        ),

        q75_error=(
            TRUE_ORIGIN,
            lambda x:
                x.quantile(0.75)
        ),

        median_ratio=(
            "threshold_ratio",
            "median"
        ),

        q25_ratio=(
            "threshold_ratio",
            lambda x:
                x.quantile(0.25)
        ),

        q75_ratio=(
            "threshold_ratio",
            lambda x:
                x.quantile(0.75)
        ),

        maximum_ratio=(
            "threshold_ratio",
            "max"
        )
    )
    .reset_index()
)


# =============================================================================
# 6. FIND THE STRONGEST MEDIAN RESPONSE
#
# Diagnostic only.
# We are NOT using this to create a new ranking.
# =============================================================================

peak_idx = (
    trajectory[
        "median_ratio"
    ].idxmax()
)

peak_row = (
    trajectory.loc[
        peak_idx
    ]
)


print("\n" + "=" * 100)
print("2. STRONGEST MEDIAN FAULT-15 RESPONSE")
print("=" * 100)


print(
    f"\nSample with highest median response: "
    f"{peak_row['sample']:.0f}"
)

print(
    f"Median reconstruction error: "
    f"{peak_row['median_error']:.10f}"
)

print(
    f"Median threshold ratio: "
    f"{peak_row['median_ratio']:.4f}x"
)

print(
    f"25–75% ratio range: "
    f"{peak_row['q25_ratio']:.4f}x "
    f"to "
    f"{peak_row['q75_ratio']:.4f}x"
)


# =============================================================================
# 7. HOW CLOSE DOES EACH RUN EVER GET TO THE THRESHOLD?
# =============================================================================

per_run = (
    fault
    .groupby(
        "simulationRun"
    )
    .agg(
        max_error=(
            TRUE_ORIGIN,
            "max"
        ),

        max_ratio=(
            "threshold_ratio",
            "max"
        ),

        median_error=(
            TRUE_ORIGIN,
            "median"
        ),

        median_ratio=(
            "threshold_ratio",
            "median"
        )
    )
    .reset_index()
)


print("\n" + "=" * 100)
print("3. MAXIMUM xmv_11 RESPONSE PER RUN")
print("=" * 100)


print(
    f"\nMedian of run maximum ratios : "
    f"{per_run['max_ratio'].median():.4f}x"
)

print(
    f"25% quantile                : "
    f"{per_run['max_ratio'].quantile(0.25):.4f}x"
)

print(
    f"75% quantile                : "
    f"{per_run['max_ratio'].quantile(0.75):.4f}x"
)

print(
    f"Minimum run maximum         : "
    f"{per_run['max_ratio'].min():.4f}x"
)

print(
    f"Maximum run maximum         : "
    f"{per_run['max_ratio'].max():.4f}x"
)


for level in [
    0.25,
    0.50,
    0.75,
    0.90,
    1.00
]:

    count = int(
        (
            per_run[
                "max_ratio"
            ] >= level
        ).sum()
    )

    percentage = (
        100
        *
        count
        /
        len(per_run)
    )

    print(
        f"Runs ever reaching >= "
        f"{level:.2f}x threshold: "
        f"{count}/"
        f"{len(per_run)} "
        f"({percentage:.2f}%)"
    )


# =============================================================================
# 8. COMPARE EARLY AND LATE PORTIONS OF EACH RUN
#
# IMPORTANT:
# We are not assuming a known fault-start sample here.
#
# Instead, compare the first 20% of each run with the final 80%.
#
# This is a descriptive diagnostic only.
# =============================================================================

change_rows = []


for run_id, run in fault.groupby(
    "simulationRun",
    sort=True
):

    run = (
        run
        .sort_values("sample")
        .reset_index(drop=True)
    )

    n = len(run)

    split = max(
        1,
        int(
            0.20 * n
        )
    )

    early = (
        run.iloc[:split][
            "threshold_ratio"
        ]
    )

    later = (
        run.iloc[split:][
            "threshold_ratio"
        ]
    )


    early_median = float(
        early.median()
    )

    later_median = float(
        later.median()
    )


    if early_median > 0:

        fold_change = (
            later_median
            /
            early_median
        )

    else:

        fold_change = np.nan


    change_rows.append({

        "simulationRun":
            run_id,

        "early_median_ratio":
            early_median,

        "later_median_ratio":
            later_median,

        "later_minus_early":
            later_median
            -
            early_median,

        "later_div_early":
            fold_change
    })


change_df = pd.DataFrame(
    change_rows
)


print("\n" + "=" * 100)
print("4. EARLY 20% VS LATER 80%")
print("=" * 100)


print(
    f"\nMedian early ratio : "
    f"{change_df['early_median_ratio'].median():.4f}x"
)

print(
    f"Median later ratio : "
    f"{change_df['later_median_ratio'].median():.4f}x"
)

print(
    f"Median absolute increase: "
    f"{change_df['later_minus_early'].median():.4f}"
)

print(
    f"Median fold-change: "
    f"{change_df['later_div_early'].median():.4f}x"
)


increased = (
    change_df[
        "later_minus_early"
    ] > 0
)


print(
    f"\nRuns where later median > early median: "
    f"{increased.sum()}/"
    f"{len(change_df)} "
    f"({100 * increased.mean():.2f}%)"
)


# =============================================================================
# 9. SHOW SELECTED POINTS OF THE MEDIAN TRAJECTORY
#
# This is just to make the output easy for us to inspect.
# =============================================================================

print("\n" + "=" * 100)
print("5. MEDIAN xmv_11 TRAJECTORY — SELECTED SAMPLES")
print("=" * 100)


available_samples = (
    trajectory[
        "sample"
    ]
    .astype(int)
    .tolist()
)


requested_samples = [
    1,
    10,
    20,
    21,
    25,
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500
]


selected_samples = [
    sample
    for sample in requested_samples
    if sample in available_samples
]


selected = trajectory[
    trajectory[
        "sample"
    ].isin(
        selected_samples
    )
][
    [
        "sample",
        "median_error",
        "median_ratio",
        "q25_ratio",
        "q75_ratio"
    ]
]


print(
    "\n" +
    selected.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# =============================================================================
# 10. SAVE EVERYTHING
# =============================================================================

trajectory_file = (
    OUTPUT_DIR /
    "v2_5_2_fault15_xmv11_trajectory.csv"
)

per_run_file = (
    OUTPUT_DIR /
    "v2_5_2_fault15_xmv11_per_run.csv"
)

change_file = (
    OUTPUT_DIR /
    "v2_5_2_fault15_xmv11_change.csv"
)


trajectory.to_csv(
    trajectory_file,
    index=False
)

per_run.to_csv(
    per_run_file,
    index=False
)

change_df.to_csv(
    change_file,
    index=False
)


print("\n" + "=" * 100)
print("V2.5.2 COMPLETE")
print("=" * 100)


print("\nSaved:")

print(
    trajectory_file
)

print(
    per_run_file
)

print(
    change_file
)


print(
    """
INTERPRETATION:

CASE A
------
xmv_11 shows a strong and consistent increase during Fault 15,
but remains below 1.0x threshold.

Meaning:
Useful information exists in the continuous reconstruction error,
but V2.5's hard threshold removes it.

CASE B
------
xmv_11 remains essentially unchanged throughout Fault 15.

Meaning:
The individual AE reconstruction error itself provides little
observable evidence that xmv_11 is the physical origin.

CASE C
------
xmv_11 changes only weakly or inconsistently.

Meaning:
Some information may exist, but it is unlikely that a simple
threshold adjustment would solve the attribution problem.

IMPORTANT:
Do not change the 0.99 threshold based on this result.
This experiment is diagnostic only.
"""
)