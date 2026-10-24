"""
===============================================================================
V2.5.1 — FAULT 15 LEADING-EDGE DIAGNOSTIC
===============================================================================

Research question:
Why does frozen V2.5 rank the known Fault 15 origin (xmv_11) so poorly?

IMPORTANT:
- V2.5 is NOT modified.
- No parameters are tuned.
- No new attribution score is created.
- Ground truth is used only to identify the sensor we want to investigate.

Frozen V2.5:
    Normal threshold = 99th percentile
    Sustained onset  = 3 consecutive samples
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

NORMAL_FILE = BASE / "TEP_cache" / "normal_error_df.parquet"
FAULT_FILE  = BASE / "TEP_cache" / "fault_error_df.parquet"

OUTPUT_DIR = BASE / "V2_5_1_Fault15_Diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAULT = 15
TRUE_ORIGIN = "xmv_11"

QUANTILE = 0.99
K = 3

SENSORS = (
    [f"xmeas_{i}" for i in range(1, 42)]
    +
    [f"xmv_{i}" for i in range(1, 12)]
)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

print("=" * 100)
print("V2.5.1 — FAULT 15 DIAGNOSTIC")
print("=" * 100)

print("\nLoading normal reconstruction errors...")

normal = pd.read_parquet(
    NORMAL_FILE,
    columns=SENSORS
)

print(f"Normal shape: {normal.shape}")


print("\nLoading Fault 15 reconstruction errors...")

fault = pd.read_parquet(
    FAULT_FILE,
    columns=[
        "faultNumber",
        "simulationRun",
        "sample",
        *SENSORS
    ]
)

fault = fault[
    fault["faultNumber"] == FAULT
].copy()

print(f"Fault 15 shape: {fault.shape}")
print(
    f"Runs: "
    f"{fault['simulationRun'].nunique()}"
)


# =============================================================================
# 3. RECREATE FROZEN V2.5 THRESHOLDS
# =============================================================================

thresholds = {
    sensor: float(
        normal[sensor].quantile(QUANTILE)
    )
    for sensor in SENSORS
}

print("\nFrozen V2.5 threshold for true origin:")

print(
    f"{TRUE_ORIGIN}: "
    f"{thresholds[TRUE_ORIGIN]:.10f}"
)

del normal


# =============================================================================
# 4. SUSTAINED ONSET FUNCTION
# =============================================================================

def sustained_onset(values, threshold, k=3):

    above = values > threshold

    if len(above) < k:
        return None

    sustained = np.convolve(
        above.astype(np.int8),
        np.ones(k, dtype=np.int8),
        mode="valid"
    )

    locations = np.flatnonzero(
        sustained == k
    )

    if len(locations) == 0:
        return None

    return int(locations[0])


# =============================================================================
# 5. ANALYSE EVERY FAULT 15 RUN
# =============================================================================

run_results = []
ranking_rows = []


for run_id, run in fault.groupby(
    "simulationRun",
    sort=True
):

    run = (
        run
        .sort_values("sample")
        .reset_index(drop=True)
    )

    samples = run["sample"].to_numpy()

    sensor_results = []


    for sensor in SENSORS:

        values = run[sensor].to_numpy()

        threshold = thresholds[sensor]

        onset_idx = sustained_onset(
            values,
            threshold,
            K
        )


        if onset_idx is None:

            onset_sample = np.inf
            strength = 0.0
            detected = False

        else:

            onset_sample = float(
                samples[onset_idx]
            )

            strength = float(
                values[onset_idx]
                /
                threshold
            )

            detected = True


        sensor_results.append({

            "sensor":
                sensor,

            "onset":
                onset_sample,

            "onset_strength":
                strength,

            "detected":
                detected
        })


    ranking = pd.DataFrame(
        sensor_results
    )


    # -------------------------------------------------------------------------
    # EXACT FROZEN V2.5 RANKING
    # -------------------------------------------------------------------------

    ranking = (
        ranking
        .sort_values(
            [
                "onset",
                "onset_strength"
            ],
            ascending=[
                True,
                False
            ],
            kind="mergesort"
        )
        .reset_index(drop=True)
    )


    ranking["rank"] = (
        np.arange(
            1,
            len(ranking) + 1
        )
    )


    ranking["simulationRun"] = run_id


    true_row = ranking[
        ranking["sensor"] == TRUE_ORIGIN
    ].iloc[0]


    winner = ranking.iloc[0]


    number_before_origin = int(
        (
            ranking["rank"]
            <
            true_row["rank"]
        ).sum()
    )


    run_results.append({

        "simulationRun":
            run_id,

        "true_rank":
            int(true_row["rank"]),

        "true_onset":
            true_row["onset"],

        "true_strength":
            true_row["onset_strength"],

        "true_detected":
            true_row["detected"],

        "winner":
            winner["sensor"],

        "winner_onset":
            winner["onset"],

        "winner_strength":
            winner["onset_strength"],

        "sensors_before_origin":
            number_before_origin
    })


    ranking_rows.append(
        ranking
    )


run_results = pd.DataFrame(
    run_results
)

all_rankings = pd.concat(
    ranking_rows,
    ignore_index=True
)


# =============================================================================
# 6. TRUE ORIGIN SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("1. TRUE ORIGIN — xmv_11")
print("=" * 100)


detected_count = int(
    run_results[
        "true_detected"
    ].sum()
)


print(
    f"\nxmv_11 sustainably abnormal in: "
    f"{detected_count}/"
    f"{len(run_results)} runs "
    f"({100 * detected_count / len(run_results):.2f}%)"
)


finite_onsets = run_results.loc[
    np.isfinite(
        run_results["true_onset"]
    ),
    "true_onset"
]


if len(finite_onsets) > 0:

    print(
        f"Median xmv_11 onset: "
        f"{finite_onsets.median():.1f}"
    )

    print(
        f"Earliest xmv_11 onset: "
        f"{finite_onsets.min():.1f}"
    )

    print(
        f"Latest xmv_11 onset: "
        f"{finite_onsets.max():.1f}"
    )

else:

    print(
        "xmv_11 never has a sustained "
        "abnormal onset."
    )


detected_strength = run_results.loc[
    run_results["true_detected"],
    "true_strength"
]


if len(detected_strength) > 0:

    print(
        f"Median xmv_11 onset strength: "
        f"{detected_strength.median():.2f}x threshold"
    )


print(
    f"\nMedian V2.5 rank: "
    f"{run_results['true_rank'].median():.1f}"
)

print(
    f"Mean V2.5 rank: "
    f"{run_results['true_rank'].mean():.2f}"
)


print(
    f"Median sensors ahead of xmv_11: "
    f"{run_results['sensors_before_origin'].median():.1f}"
)


# =============================================================================
# 7. WHICH SENSORS WIN FAULT 15?
# =============================================================================

print("\n" + "=" * 100)
print("2. MOST COMMON V2.5 WINNERS")
print("=" * 100)


winner_counts = (
    run_results[
        "winner"
    ]
    .value_counts()
    .rename_axis("sensor")
    .reset_index(name="wins")
)


winner_counts["percentage"] = (
    100
    *
    winner_counts["wins"]
    /
    len(run_results)
)


print(
    "\n" +
    winner_counts
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# =============================================================================
# 8. WHICH SENSORS CONSISTENTLY RANK AHEAD OF xmv_11?
# =============================================================================

ahead_counts = {
    sensor: 0
    for sensor in SENSORS
    if sensor != TRUE_ORIGIN
}


for run_id, ranking in all_rankings.groupby(
    "simulationRun"
):

    true_rank = int(
        ranking.loc[
            ranking["sensor"] == TRUE_ORIGIN,
            "rank"
        ].iloc[0]
    )

    ahead = ranking[
        ranking["rank"] < true_rank
    ]["sensor"]

    for sensor in ahead:

        ahead_counts[sensor] += 1


ahead_df = pd.DataFrame({

    "sensor":
        list(ahead_counts.keys()),

    "runs_ahead_of_xmv11":
        list(ahead_counts.values())
})


ahead_df["percentage"] = (
    100
    *
    ahead_df["runs_ahead_of_xmv11"]
    /
    len(run_results)
)


ahead_df = (
    ahead_df
    .sort_values(
        "runs_ahead_of_xmv11",
        ascending=False
    )
)


print("\n" + "=" * 100)
print("3. SENSORS MOST OFTEN AHEAD OF xmv_11")
print("=" * 100)


print(
    "\n" +
    ahead_df
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# =============================================================================
# 9. RUN 1 — LOOK AT THE LEADING EDGE DIRECTLY
# =============================================================================

run1 = (
    fault[
        fault["simulationRun"] == 1
    ]
    .sort_values("sample")
    .reset_index(drop=True)
)


run1_ranking = all_rankings[
    all_rankings["simulationRun"] == 1
].sort_values("rank")


print("\n" + "=" * 100)
print("4. RUN 1 — TOP 10 SENSORS")
print("=" * 100)


print(
    "\n" +
    run1_ranking[
        [
            "rank",
            "sensor",
            "onset",
            "onset_strength"
        ]
    ]
    .head(10)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


true_run1 = run1_ranking[
    run1_ranking["sensor"] == TRUE_ORIGIN
].iloc[0]


print(
    f"\nRun 1 xmv_11:"
)

print(
    f"  Rank     : "
    f"{int(true_run1['rank'])}"
)

print(
    f"  Onset    : "
    f"{true_run1['onset']}"
)

print(
    f"  Strength : "
    f"{true_run1['onset_strength']:.3f}x"
)


# =============================================================================
# 10. SAVE V2.5.1 RESULTS
# =============================================================================

run_file = (
    OUTPUT_DIR /
    "v2_5_1_fault15_run_summary.csv"
)

winner_file = (
    OUTPUT_DIR /
    "v2_5_1_fault15_winners.csv"
)

ahead_file = (
    OUTPUT_DIR /
    "v2_5_1_fault15_sensors_ahead.csv"
)

ranking_file = (
    OUTPUT_DIR /
    "v2_5_1_fault15_rankings.parquet"
)


run_results.to_csv(
    run_file,
    index=False
)

winner_counts.to_csv(
    winner_file,
    index=False
)

ahead_df.to_csv(
    ahead_file,
    index=False
)

all_rankings.to_parquet(
    ranking_file,
    index=False
)


print("\n" + "=" * 100)
print("V2.5.1 DIAGNOSTIC COMPLETE")
print("=" * 100)


print("\nSaved:")

print(run_file)
print(winner_file)
print(ahead_file)
print(ranking_file)


print(
    """
WHAT WE WANT TO LEARN:

1. Does xmv_11 actually become abnormal?

2. If it does, WHEN does it become abnormal?

3. How many sensors have already reacted before xmv_11?

4. Are the same sensors repeatedly appearing first?

Do NOT change V2.5 based on this output.

V2.5.1 is diagnostic evidence only.
"""
)