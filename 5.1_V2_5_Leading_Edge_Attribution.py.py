"""
===============================================================================
V2.5 — LEADING-EDGE SENSOR ATTRIBUTION
===============================================================================

Purpose
-------
Rank candidate fault-origin sensors using the earliest sustained abnormal
reconstruction-error response.

The attribution method is label-free at inference.

For each sensor:

    1. Estimate a sensor-specific abnormality threshold from normal
       autoencoder reconstruction errors.

    2. Find the first sample at which reconstruction error remains above
       that threshold for K consecutive samples.

    3. Rank sensors primarily by earliest sustained onset.

    4. When sensors have the same onset, rank them by the strength of the
       reconstruction error relative to their own normal threshold.

Frozen V2.5 configuration
-------------------------
Normal threshold quantile : 0.99
Sustained onset length    : 3 samples

Ranking:
    earlier onset        -> better
    tied onset           -> stronger threshold-normalised response wins
    never abnormal       -> placed after sensors with detected onset

IMPORTANT
---------
Ground-truth fault origins are NOT used to construct the sensor ranking.

The TRUE_CAUSES dictionary is used ONLY after attribution has completed,
for evaluation of Top-k accuracy and true-origin rank.

No:
    - supervised classifier
    - learned fusion weights
    - fault-specific thresholds
    - structural-network weighting
    - root-cause labels during attribution

This file freezes the V2.5 method before further failure analysis.

===============================================================================
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

BASE = Path(
    "/content/drive/MyDrive/PHD/"
    "Router_V2_Structural_Attribution"
)

NORMAL_CACHE = (
    BASE /
    "TEP_cache" /
    "normal_error_df.parquet"
)

FAULT_CACHE = (
    BASE /
    "TEP_cache" /
    "fault_error_df.parquet"
)

OUTPUT_DIR = (
    BASE /
    "V2_5_Leading_Edge"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------------------------------------------------------
# FROZEN V2.5 SETTINGS
# -----------------------------------------------------------------------------

NORMAL_QUANTILE = 0.99

SUSTAIN_K = 3


# -----------------------------------------------------------------------------
# TEP SENSOR LIST
# -----------------------------------------------------------------------------

SENSORS = (
    [f"xmeas_{i}" for i in range(1, 42)]
    +
    [f"xmv_{i}" for i in range(1, 12)]
)


# -----------------------------------------------------------------------------
# CONFIRMED TEP ROOT CAUSES
#
# IMPORTANT:
# These labels are used ONLY for final evaluation.
# They are never passed to the attribution algorithm.
# -----------------------------------------------------------------------------

TRUE_CAUSES = {

    1:  "xmeas_4",
    2:  "xmeas_4",
    3:  "xmeas_23",
    4:  "xmv_10",
    5:  "xmeas_22",
    6:  "xmv_3",
    7:  "xmeas_4",
    8:  "xmeas_4",
    9:  "xmeas_23",
    10: "xmeas_4",
    11: "xmv_10",
    12: "xmv_11",
    13: "xmeas_9",
    14: "xmv_10",
    15: "xmv_11",
}


KNOWN_FAULTS = sorted(
    TRUE_CAUSES.keys()
)


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def calculate_thresholds(
    normal_df,
    sensors,
    quantile
):
    """
    Calculate one reconstruction-error threshold per sensor
    using fault-free data only.
    """

    thresholds = {}

    for sensor in sensors:

        values = (
            normal_df[sensor]
            .dropna()
            .to_numpy()
        )

        thresholds[sensor] = float(
            np.quantile(
                values,
                quantile
            )
        )

    return thresholds


# -----------------------------------------------------------------------------


def find_sustained_onset(
    values,
    threshold,
    k
):
    """
    Return the ARRAY INDEX of the first point beginning a sequence
    of k consecutive reconstruction errors above threshold.

    Returns None if no sustained abnormality is detected.
    """

    above = (
        values > threshold
    )

    if len(above) < k:

        return None

    # Convolution counts the number of True values
    # inside each K-sample window.

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

    return int(
        locations[0]
    )


# -----------------------------------------------------------------------------


def attribute_run(
    run_df,
    sensors,
    thresholds,
    sustain_k
):
    """
    Perform V2.5 attribution for ONE process run.

    This function has no access to the true fault origin.

    Returns a ranked DataFrame containing all sensors.
    """

    run_df = (
        run_df
        .sort_values("sample")
        .reset_index(drop=True)
    )

    samples = (
        run_df["sample"]
        .to_numpy()
    )

    sensor_rows = []


    for sensor in sensors:

        values = (
            run_df[sensor]
            .to_numpy()
        )

        threshold = (
            thresholds[sensor]
        )


        # ---------------------------------------------------------------------
        # Find sustained abnormal onset
        # ---------------------------------------------------------------------

        onset_idx = (
            find_sustained_onset(
                values=values,
                threshold=threshold,
                k=sustain_k
            )
        )


        # ---------------------------------------------------------------------
        # Sensor never becomes sustainably abnormal
        # ---------------------------------------------------------------------

        if onset_idx is None:

            onset_sample = np.inf

            onset_strength = 0.0

            abnormal_detected = False


        # ---------------------------------------------------------------------
        # Sustained abnormality detected
        # ---------------------------------------------------------------------

        else:

            onset_sample = float(
                samples[onset_idx]
            )

            onset_strength = float(
                values[onset_idx]
                /
                threshold
            )

            abnormal_detected = True


        sensor_rows.append({

            "sensor":
                sensor,

            "threshold":
                threshold,

            "abnormal_detected":
                abnormal_detected,

            "onset":
                onset_sample,

            "onset_strength":
                onset_strength,
        })


    ranking = pd.DataFrame(
        sensor_rows
    )


    # =========================================================================
    # FROZEN V2.5 RANKING RULE
    #
    # Primary:
    #       earliest sustained abnormal onset
    #
    # Secondary:
    #       strongest threshold-normalised reconstruction error
    #       when onset is identical
    # =========================================================================

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


    ranking[
        "v2_5_rank"
    ] = (
        np.arange(
            1,
            len(ranking) + 1
        )
    )


    return ranking


# -----------------------------------------------------------------------------


def calculate_metrics(
    evaluation_df
):
    """
    Calculate root-cause ranking metrics.
    """

    return {

        "Top1_%":
            100 * (
                evaluation_df[
                    "true_rank"
                ] <= 1
            ).mean(),

        "Top3_%":
            100 * (
                evaluation_df[
                    "true_rank"
                ] <= 3
            ).mean(),

        "Top5_%":
            100 * (
                evaluation_df[
                    "true_rank"
                ] <= 5
            ).mean(),

        "Top10_%":
            100 * (
                evaluation_df[
                    "true_rank"
                ] <= 10
            ).mean(),

        "Median_rank":
            evaluation_df[
                "true_rank"
            ].median(),

        "Mean_rank":
            evaluation_df[
                "true_rank"
            ].mean(),
    }


# =============================================================================
# 3. LOAD NORMAL DATA
# =============================================================================

print(
    "=" * 100
)

print(
    "V2.5 — LEADING-EDGE SENSOR ATTRIBUTION"
)

print(
    "=" * 100
)


print(
    "\nLoading normal reconstruction-error cache..."
)


normal_df = pd.read_parquet(
    NORMAL_CACHE,
    columns=SENSORS
)


print(
    f"Normal cache shape: "
    f"{normal_df.shape}"
)


# =============================================================================
# 4. BUILD NORMAL THRESHOLDS
#
# NO FAULT DATA IS USED HERE.
# =============================================================================

print(
    "\nCalculating frozen sensor-specific thresholds..."
)


thresholds = calculate_thresholds(

    normal_df=normal_df,

    sensors=SENSORS,

    quantile=NORMAL_QUANTILE
)


threshold_df = pd.DataFrame({

    "sensor":
        list(
            thresholds.keys()
        ),

    "threshold":
        list(
            thresholds.values()
        )
})


threshold_file = (
    OUTPUT_DIR /
    "v2_5_normal_thresholds.csv"
)


threshold_df.to_csv(
    threshold_file,
    index=False
)


print(
    f"Thresholds calculated: "
    f"{len(thresholds)}"
)


# Normal data no longer required.

del normal_df


# =============================================================================
# 5. LOAD FAULT RECONSTRUCTION ERRORS
# =============================================================================

print(
    "\nLoading fault reconstruction-error cache..."
)


fault_df = pd.read_parquet(

    FAULT_CACHE,

    columns=[

        "faultNumber",

        "simulationRun",

        "sample",

        *SENSORS
    ]
)


# Keep only faults with confirmed evaluation labels.

fault_df = fault_df[

    fault_df[
        "faultNumber"
    ].isin(
        KNOWN_FAULTS
    )

].copy()


run_index = (

    fault_df[
        [
            "faultNumber",
            "simulationRun"
        ]
    ]

    .drop_duplicates()

    .sort_values(
        [
            "faultNumber",
            "simulationRun"
        ]
    )
)


total_runs = len(
    run_index
)


print(
    f"Known faults: "
    f"{len(KNOWN_FAULTS)}"
)

print(
    f"Evaluation runs: "
    f"{total_runs:,}"
)

print(
    f"Sensors per run: "
    f"{len(SENSORS)}"
)


# =============================================================================
# 6. V2.5 ATTRIBUTION
#
# IMPORTANT:
# TRUE_CAUSES IS NOT USED ANYWHERE IN THIS SECTION.
# =============================================================================

print(
    "\nRunning V2.5 attribution..."
)


all_rankings = []

processed = 0


for (
    fault_number,
    simulation_run
), run_df in fault_df.groupby(
    [
        "faultNumber",
        "simulationRun"
    ],
    sort=True
):


    ranking = attribute_run(

        run_df=run_df,

        sensors=SENSORS,

        thresholds=thresholds,

        sustain_k=SUSTAIN_K
    )


    ranking.insert(
        0,
        "simulationRun",
        simulation_run
    )

    ranking.insert(
        0,
        "faultNumber",
        int(fault_number)
    )


    all_rankings.append(
        ranking
    )


    processed += 1


    if (
        processed % 500 == 0
        or
        processed == total_runs
    ):

        print(
            f"Processed "
            f"{processed:,} / "
            f"{total_runs:,} runs"
        )


rankings_df = pd.concat(
    all_rankings,
    ignore_index=True
)


# =============================================================================
# 7. SAVE LABEL-FREE ATTRIBUTION OUTPUT FIRST
#
# At this point V2.5 attribution is complete.
#
# This file contains the output of the attribution algorithm
# BEFORE ground-truth root causes are used for evaluation.
# =============================================================================

rankings_file = (

    OUTPUT_DIR /
    "v2_5_sensor_rankings.parquet"
)


rankings_df.to_parquet(
    rankings_file,
    index=False
)


print(
    "\nLabel-free attribution complete."
)

print(
    f"Rankings saved:\n"
    f"{rankings_file}"
)


# =============================================================================
# 8. EVALUATION ONLY
#
# Ground truth enters the workflow HERE for the first time.
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "EVALUATION — GROUND TRUTH USED FROM THIS POINT ONLY"
)

print(
    "=" * 100
)


evaluation_rows = []


for (
    fault_number,
    simulation_run
), run_ranking in rankings_df.groupby(
    [
        "faultNumber",
        "simulationRun"
    ]
):


    fault_number = int(
        fault_number
    )


    true_sensor = (
        TRUE_CAUSES[
            fault_number
        ]
    )


    true_row = run_ranking[

        run_ranking[
            "sensor"
        ] == true_sensor

    ]


    if len(true_row) != 1:

        raise RuntimeError(

            f"Expected exactly one row for "
            f"true sensor {true_sensor}, "
            f"Fault {fault_number}, "
            f"Run {simulation_run}."
        )


    true_row = (
        true_row.iloc[0]
    )


    evaluation_rows.append({

        "faultNumber":
            fault_number,

        "simulationRun":
            simulation_run,

        "true_sensor":
            true_sensor,

        "true_rank":
            int(
                true_row[
                    "v2_5_rank"
                ]
            ),

        "true_onset":
            true_row[
                "onset"
            ],

        "true_onset_strength":
            true_row[
                "onset_strength"
            ],
    })


evaluation_df = pd.DataFrame(
    evaluation_rows
)


# =============================================================================
# 9. OVERALL PERFORMANCE
# =============================================================================

overall = calculate_metrics(
    evaluation_df
)


print(
    "\n" + "=" * 100
)

print(
    "V2.5 OVERALL PERFORMANCE"
)

print(
    "=" * 100
)


print(
    f"\nTop-1  : "
    f"{overall['Top1_%']:.2f}%"
)

print(
    f"Top-3  : "
    f"{overall['Top3_%']:.2f}%"
)

print(
    f"Top-5  : "
    f"{overall['Top5_%']:.2f}%"
)

print(
    f"Top-10 : "
    f"{overall['Top10_%']:.2f}%"
)

print(
    f"\nMedian rank : "
    f"{overall['Median_rank']:.2f}"
)

print(
    f"Mean rank   : "
    f"{overall['Mean_rank']:.2f}"
)


# =============================================================================
# 10. PER-FAULT PERFORMANCE
# =============================================================================

per_fault_rows = []


for fault_number in KNOWN_FAULTS:


    temp = evaluation_df[

        evaluation_df[
            "faultNumber"
        ] == fault_number
    ]


    metrics = calculate_metrics(
        temp
    )


    per_fault_rows.append({

        "fault":
            fault_number,

        "true_sensor":
            TRUE_CAUSES[
                fault_number
            ],

        **metrics
    })


per_fault_df = pd.DataFrame(
    per_fault_rows
)


print(
    "\n" + "=" * 100
)

print(
    "V2.5 PERFORMANCE BY FAULT"
)

print(
    "=" * 100
)


print(

    per_fault_df.to_string(

        index=False,

        float_format=lambda x:
            f"{x:.2f}"
    )
)


# =============================================================================
# 11. SUMMARY STATISTICS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "V2.5 FAULT-LEVEL SUMMARY"
)

print(
    "=" * 100
)


print(
    "\nFaults with Top-1 >= 50%: "
    f"{(per_fault_df['Top1_%'] >= 50).sum()}"
    f"/{len(per_fault_df)}"
)


print(
    "Faults with Top-3 >= 50%: "
    f"{(per_fault_df['Top3_%'] >= 50).sum()}"
    f"/{len(per_fault_df)}"
)


print(
    "Faults with Top-3 >= 80%: "
    f"{(per_fault_df['Top3_%'] >= 80).sum()}"
    f"/{len(per_fault_df)}"
)


print(
    "Faults with median rank <= 3: "
    f"{(per_fault_df['Median_rank'] <= 3).sum()}"
    f"/{len(per_fault_df)}"
)


# =============================================================================
# 12. SAVE EVALUATION RESULTS
# =============================================================================

evaluation_file = (

    OUTPUT_DIR /
    "v2_5_evaluation_by_run.csv"
)


per_fault_file = (

    OUTPUT_DIR /
    "v2_5_per_fault_results.csv"
)


evaluation_df.to_csv(
    evaluation_file,
    index=False
)


per_fault_df.to_csv(
    per_fault_file,
    index=False
)


# =============================================================================
# 13. SAVE METHOD METADATA
#
# This gives us a permanent record of exactly what V2.5 means.
# =============================================================================

metadata = {

    "method":
        "V2.5 Leading-Edge Sensor Attribution",

    "normal_threshold_quantile":
        NORMAL_QUANTILE,

    "sustained_onset_samples":
        SUSTAIN_K,

    "number_of_sensors":
        len(SENSORS),

    "ranking_primary":
        "earliest sustained abnormal onset",

    "ranking_secondary":
        "threshold-normalised reconstruction error at onset",

    "attribution_uses_root_cause_labels":
        False,

    "structural_network_used":
        False,

    "learned_weights":
        False,

    "fault_specific_parameters":
        False,

    "evaluation_faults":
        KNOWN_FAULTS,

    "evaluation_only_root_causes":
        TRUE_CAUSES,
}


metadata_file = (

    OUTPUT_DIR /
    "v2_5_method_metadata.json"
)


with open(
    metadata_file,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# =============================================================================
# 14. FINAL OUTPUT
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "V2.5 COMPLETE"
)

print(
    "=" * 100
)


print(
    "\nFrozen method:"
)

print(
    "  Normal threshold = "
    "99th percentile"
)

print(
    "  Sustained onset = "
    "3 consecutive samples"
)

print(
    "  Primary ranking = "
    "earliest onset"
)

print(
    "  Tie-break = "
    "strongest threshold-normalised "
    "response at onset"
)


print(
    "\nSaved files:"
)

print(
    f"  {threshold_file}"
)

print(
    f"  {rankings_file}"
)

print(
    f"  {evaluation_file}"
)

print(
    f"  {per_fault_file}"
)

print(
    f"  {metadata_file}"
)


print(
    "\nIMPORTANT:"
)

print(
    "V2.5 attribution is now frozen."
)

print(
    "Further analysis of difficult faults should "
    "not modify these results."
)