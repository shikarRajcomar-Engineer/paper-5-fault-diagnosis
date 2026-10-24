"""
===============================================================================
V2.5.3 — FAULT 15: NORMAL VS FAULT SPIKE DIAGNOSTIC
===============================================================================

Research question:
Are the isolated xmv_11 reconstruction-error spikes observed during Fault 15
different from the spike behaviour observed during normal operation?

Previous findings
-----------------
V2.5.1:
    xmv_11 sustained abnormal onset = 0 / 500 Fault-15 runs.

V2.5.2:
    xmv_11 crosses the frozen threshold at least once in 496 / 500 runs,
    but does not show a sustained reconstruction-error shift.

V2.5.3:
    Compare the spike behaviour against healthy operation.

IMPORTANT
---------
This is a diagnostic experiment only.

It does NOT:
    - modify V2.5
    - change the threshold
    - change K
    - create a new ranking
    - use a new attribution score

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
    "V2_5_3_Fault15_Normal_vs_Fault_Spikes"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SENSOR = "xmv_11"
FAULT = 15

QUANTILE = 0.99

EXPECTED_RUN_LENGTH = 500


# =============================================================================
# 2. LOAD NORMAL DATA
# =============================================================================

print("=" * 100)
print("V2.5.3 — FAULT 15 NORMAL VS FAULT SPIKE DIAGNOSTIC")
print("=" * 100)


print("\nLoading normal xmv_11 reconstruction errors...")


normal = pd.read_parquet(
    NORMAL_FILE,
    columns=[SENSOR]
)


print(
    f"Normal rows: "
    f"{len(normal):,}"
)


# =============================================================================
# 3. RECONSTRUCT NORMAL RUN IDs
#
# The cached normal-error file contains 250,000 rows:
#
#       500 runs × 500 samples
#
# The rows were generated from the normal TEP cache in run order.
#
# We reconstruct run/sample identifiers ONLY to compare run-level
# spike behaviour.
# =============================================================================

if len(normal) % EXPECTED_RUN_LENGTH != 0:

    raise RuntimeError(
        "Normal cache length is not divisible by the expected "
        f"run length ({EXPECTED_RUN_LENGTH}). "
        "Do not continue until the normal run structure is verified."
    )


n_normal_runs = (
    len(normal)
    //
    EXPECTED_RUN_LENGTH
)


normal = normal.copy()


normal[
    "simulationRun"
] = (
    np.repeat(
        np.arange(
            1,
            n_normal_runs + 1
        ),
        EXPECTED_RUN_LENGTH
    )
)


normal[
    "sample"
] = (
    np.tile(
        np.arange(
            1,
            EXPECTED_RUN_LENGTH + 1
        ),
        n_normal_runs
    )
)


print(
    f"Reconstructed normal runs: "
    f"{n_normal_runs}"
)


# =============================================================================
# 4. LOAD FAULT 15
# =============================================================================

print(
    "\nLoading Fault 15 xmv_11 reconstruction errors..."
)


fault = pd.read_parquet(

    FAULT_FILE,

    columns=[
        "faultNumber",
        "simulationRun",
        "sample",
        SENSOR
    ]
)


fault = fault[
    fault[
        "faultNumber"
    ] == FAULT
].copy()


fault = fault.sort_values(
    [
        "simulationRun",
        "sample"
    ]
)


print(
    f"Fault 15 rows: "
    f"{len(fault):,}"
)


print(
    f"Fault 15 runs: "
    f"{fault['simulationRun'].nunique()}"
)


# =============================================================================
# 5. FROZEN V2.5 THRESHOLD
#
# IMPORTANT:
# Same 99th-percentile normal threshold used by V2.5.
# =============================================================================

threshold = float(

    normal[
        SENSOR
    ].quantile(
        QUANTILE
    )
)


print(
    "\nFrozen V2.5 threshold:"
)

print(
    f"{SENSOR}: "
    f"{threshold:.10f}"
)


# =============================================================================
# 6. THRESHOLD-NORMALISED ERROR
# =============================================================================

normal[
    "threshold_ratio"
] = (
    normal[SENSOR]
    /
    threshold
)


fault[
    "threshold_ratio"
] = (
    fault[SENSOR]
    /
    threshold
)


# =============================================================================
# 7. RUN-LEVEL SPIKE FEATURES
#
# These are DESCRIPTIVE FEATURES ONLY.
#
# For each run we calculate:
#
#   maximum threshold ratio
#   number of samples > threshold
#   fraction of samples > threshold
#   longest consecutive sequence > threshold
#
# This does not create a new attribution score.
# =============================================================================

def longest_consecutive_true(mask):

    mask = np.asarray(
        mask,
        dtype=bool
    )

    longest = 0
    current = 0

    for value in mask:

        if value:

            current += 1

            longest = max(
                longest,
                current
            )

        else:

            current = 0

    return longest


def summarize_runs(
    df,
    group_name
):

    rows = []


    for run_id, run in df.groupby(
        "simulationRun",
        sort=True
    ):

        run = run.sort_values(
            "sample"
        )

        ratios = (
            run[
                "threshold_ratio"
            ].to_numpy()
        )

        above = (
            ratios > 1.0
        )


        max_idx = int(
            np.argmax(
                ratios
            )
        )


        rows.append({

            "group":
                group_name,

            "simulationRun":
                int(run_id),

            "max_ratio":
                float(
                    np.max(
                        ratios
                    )
                ),

            "max_sample":
                int(
                    run.iloc[
                        max_idx
                    ][
                        "sample"
                    ]
                ),

            "samples_above_threshold":
                int(
                    np.sum(
                        above
                    )
                ),

            "fraction_above_threshold":
                float(
                    np.mean(
                        above
                    )
                ),

            "longest_above_threshold_run":
                int(
                    longest_consecutive_true(
                        above
                    )
                ),

            "median_ratio":
                float(
                    np.median(
                        ratios
                    )
                ),

            "q95_ratio":
                float(
                    np.quantile(
                        ratios,
                        0.95
                    )
                ),

            "q99_ratio":
                float(
                    np.quantile(
                        ratios,
                        0.99
                    )
                )
        })


    return pd.DataFrame(
        rows
    )


normal_summary = summarize_runs(
    normal,
    "Normal"
)


fault_summary = summarize_runs(
    fault,
    "Fault15"
)


# =============================================================================
# 8. BASIC COMPARISON
# =============================================================================

print("\n" + "=" * 100)
print("1. MAXIMUM SPIKE PER RUN")
print("=" * 100)


for name, df in [

    ("Normal", normal_summary),

    ("Fault 15", fault_summary)

]:

    print(
        f"\n{name}"
    )

    print(
        f"  Median maximum : "
        f"{df['max_ratio'].median():.4f}x"
    )

    print(
        f"  25% quantile   : "
        f"{df['max_ratio'].quantile(0.25):.4f}x"
    )

    print(
        f"  75% quantile   : "
        f"{df['max_ratio'].quantile(0.75):.4f}x"
    )

    print(
        f"  95% quantile   : "
        f"{df['max_ratio'].quantile(0.95):.4f}x"
    )

    print(
        f"  Maximum        : "
        f"{df['max_ratio'].max():.4f}x"
    )


# =============================================================================
# 9. HOW OFTEN DOES EACH RUN CROSS THE V2.5 THRESHOLD?
# =============================================================================

print("\n" + "=" * 100)
print("2. RUNS WITH AT LEAST ONE THRESHOLD CROSSING")
print("=" * 100)


for name, df in [

    ("Normal", normal_summary),

    ("Fault 15", fault_summary)

]:

    crossed = (
        df[
            "samples_above_threshold"
        ] > 0
    )

    print(
        f"\n{name}: "
        f"{crossed.sum()}/"
        f"{len(df)} "
        f"({100 * crossed.mean():.2f}%)"
    )


# =============================================================================
# 10. NUMBER OF SPIKES PER RUN
# =============================================================================

print("\n" + "=" * 100)
print("3. NUMBER OF ABOVE-THRESHOLD SAMPLES PER RUN")
print("=" * 100)


for name, df in [

    ("Normal", normal_summary),

    ("Fault 15", fault_summary)

]:

    print(
        f"\n{name}"
    )

    print(
        f"  Median : "
        f"{df['samples_above_threshold'].median():.2f}"
    )

    print(
        f"  Mean   : "
        f"{df['samples_above_threshold'].mean():.2f}"
    )

    print(
        f"  75%    : "
        f"{df['samples_above_threshold'].quantile(0.75):.2f}"
    )

    print(
        f"  95%    : "
        f"{df['samples_above_threshold'].quantile(0.95):.2f}"
    )

    print(
        f"  Max    : "
        f"{df['samples_above_threshold'].max()}"
    )


# =============================================================================
# 11. LONGEST CONSECUTIVE CROSSING
#
# This directly helps explain why K=3 never fires for Fault 15.
# =============================================================================

print("\n" + "=" * 100)
print("4. LONGEST CONSECUTIVE ABOVE-THRESHOLD SEQUENCE")
print("=" * 100)


for name, df in [

    ("Normal", normal_summary),

    ("Fault 15", fault_summary)

]:

    print(
        f"\n{name}"
    )

    print(
        f"  Median : "
        f"{df['longest_above_threshold_run'].median():.2f}"
    )

    print(
        f"  Mean   : "
        f"{df['longest_above_threshold_run'].mean():.2f}"
    )

    print(
        f"  95%    : "
        f"{df['longest_above_threshold_run'].quantile(0.95):.2f}"
    )

    print(
        f"  Max    : "
        f"{df['longest_above_threshold_run'].max()}"
    )


    for k in [
        1,
        2,
        3
    ]:

        count = (
            df[
                "longest_above_threshold_run"
            ] >= k
        ).sum()

        print(
            f"  Runs with sequence >= {k}: "
            f"{count}/"
            f"{len(df)} "
            f"({100 * count / len(df):.2f}%)"
        )


# =============================================================================
# 12. CONTINUOUS DISTRIBUTION COMPARISON
#
# If Fault 15 is genuinely changing xmv_11, we may also see a shift
# in these run-level continuous summaries.
# =============================================================================

print("\n" + "=" * 100)
print("5. CONTINUOUS ERROR DISTRIBUTION")
print("=" * 100)


for metric in [

    "median_ratio",
    "q95_ratio",
    "q99_ratio"

]:

    print(
        f"\n{metric}"
    )

    print(
        f"  Normal median   : "
        f"{normal_summary[metric].median():.4f}"
    )

    print(
        f"  Fault 15 median : "
        f"{fault_summary[metric].median():.4f}"
    )


# =============================================================================
# 13. SIMPLE EFFECT-SIZE DIAGNOSTIC
#
# Cliff's delta compares the distributions without assuming normality.
#
# Interpretation:
#     +1 -> every Fault15 value > every normal value
#      0 -> strong overlap
#     -1 -> every Fault15 value < every normal value
#
# Diagnostic only.
# =============================================================================

def cliffs_delta(
    fault_values,
    normal_values
):

    fault_values = np.asarray(
        fault_values
    )

    normal_values = np.asarray(
        normal_values
    )


    greater = 0
    lower = 0


    for value in fault_values:

        greater += np.sum(
            value > normal_values
        )

        lower += np.sum(
            value < normal_values
        )


    total = (
        len(fault_values)
        *
        len(normal_values)
    )


    return (
        greater - lower
    ) / total


print("\n" + "=" * 100)
print("6. FAULT-15 VS NORMAL EFFECT SIZE")
print("=" * 100)


metrics = [

    "max_ratio",

    "samples_above_threshold",

    "longest_above_threshold_run",

    "median_ratio",

    "q95_ratio",

    "q99_ratio"
]


effect_rows = []


for metric in metrics:

    delta = cliffs_delta(

        fault_summary[
            metric
        ].to_numpy(),

        normal_summary[
            metric
        ].to_numpy()
    )


    effect_rows.append({

        "metric":
            metric,

        "cliffs_delta":
            delta
    })


effect_df = pd.DataFrame(
    effect_rows
)


print(
    "\n" +
    effect_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# =============================================================================
# 14. WHERE DO MAXIMUM SPIKES OCCUR?
#
# If spike locations are essentially random in both groups,
# that is useful information too.
# =============================================================================

print("\n" + "=" * 100)
print("7. LOCATION OF MAXIMUM SPIKE")
print("=" * 100)


for name, df in [

    ("Normal", normal_summary),

    ("Fault 15", fault_summary)

]:

    print(
        f"\n{name}"
    )

    print(
        f"  Median sample : "
        f"{df['max_sample'].median():.1f}"
    )

    print(
        f"  Q25 sample    : "
        f"{df['max_sample'].quantile(0.25):.1f}"
    )

    print(
        f"  Q75 sample    : "
        f"{df['max_sample'].quantile(0.75):.1f}"
    )


# =============================================================================
# 15. SAVE RESULTS
# =============================================================================

normal_file = (
    OUTPUT_DIR /
    "v2_5_3_normal_xmv11_spikes.csv"
)

fault_file = (
    OUTPUT_DIR /
    "v2_5_3_fault15_xmv11_spikes.csv"
)

effect_file = (
    OUTPUT_DIR /
    "v2_5_3_normal_vs_fault15_effect_sizes.csv"
)


normal_summary.to_csv(
    normal_file,
    index=False
)

fault_summary.to_csv(
    fault_file,
    index=False
)

effect_df.to_csv(
    effect_file,
    index=False
)


print("\n" + "=" * 100)
print("V2.5.3 COMPLETE")
print("=" * 100)


print("\nSaved:")

print(
    normal_file
)

print(
    fault_file
)

print(
    effect_file
)


print(
    """
HOW TO INTERPRET THIS EXPERIMENT
--------------------------------

POSSIBILITY A
Normal and Fault 15 have very similar spike behaviour.

Meaning:
The isolated xmv_11 spikes are probably normal AE variability.
There is little useful Fault-15 attribution information in them.

POSSIBILITY B
Fault 15 has substantially more/larger isolated spikes than normal.

Meaning:
Fault information exists in xmv_11, but V2.5's sustained-onset
representation does not capture that type of evidence.

POSSIBILITY C
Some spike properties differ while others overlap.

Meaning:
Fault 15 may produce a weak or intermittent signature.
We would investigate that scientifically before considering any
new attribution method.

IMPORTANT:
No V2.5 parameters should be changed from this diagnostic.
"""
)