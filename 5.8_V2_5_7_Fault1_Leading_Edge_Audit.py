"""V2.5.7 — Fault 1 leading-edge and propagation audit.

Purpose: explain why origin xmeas_4 is abnormal in every Fault-1 run
but ranks poorly under frozen V2.5. Diagnostic only; no new ranking,
training, causal inference, or threshold tuning.

Compare Fault-1 onset order against healthy-run false alarms. Report
sensors preceding the known origin, their onset timing, and how often
their own sustained detections occur during healthy operation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BASE = Path("/content/drive/MyDrive/PHD/Router_V2_Structural_Attribution")
NORMAL = BASE / "TEP_cache/normal_error_df.parquet"
FAULT = BASE / "TEP_cache/fault_error_df.parquet"
RAW_NORMAL = Path("/content/drive/MyDrive/PHD/Paper5_Datasets/TEP_FaultFree_Training.parquet")
OUT = BASE / "V2_5_7_Fault1_Leading_Edge_Audit"
SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
ORIGIN = "xmeas_4"
K = 3
Q = 0.99


def to_runs(frame, label):
    frame = frame.sort_values(["simulationRun", "sample"], kind="stable")
    sizes = frame.groupby("simulationRun").size()
    assert len(sizes) == 500 and (sizes == 500).all(), f"{label}: invalid run sizes"
    expected = np.tile(np.arange(1, 501), 500)
    assert np.array_equal(frame["sample"].to_numpy(), expected), f"{label}: invalid sample order"
    return frame["simulationRun"].drop_duplicates().to_numpy(), (
        frame[SENSORS].to_numpy(dtype=np.float32).reshape(500, 500, 52)
    )


def sustained_onsets(data, thresholds):
    above = data > thresholds[None, None, :]
    sustained = above[:, :-2, :] & above[:, 1:-1, :] & above[:, 2:, :]
    valid = sustained.any(axis=1)
    first = sustained.argmax(axis=1).astype(float) + 1
    first[~valid] = np.inf
    return first


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 86, flush=True)
    print("V2.5.7 — FAULT 1 LEADING-EDGE AND PROPAGATION AUDIT", flush=True)
    print("=" * 86, flush=True)

    normal_errors = pd.read_parquet(NORMAL, columns=SENSORS)
    normal_ids = pd.read_parquet(RAW_NORMAL, columns=["simulationRun", "sample"])
    assert len(normal_ids) == len(normal_errors) == 250000
    normal = pd.concat(
        [normal_ids.reset_index(drop=True), normal_errors.reset_index(drop=True)], axis=1
    )
    _, norm_a = to_runs(normal, "Healthy")
    thresholds = np.quantile(norm_a, Q, axis=(0, 1))
    norm_onsets = sustained_onsets(norm_a, thresholds)
    healthy_detection = np.isfinite(norm_onsets).mean(axis=0)
    del normal, normal_errors, normal_ids, norm_a

    dataset = ds.dataset(str(FAULT), format="parquet")
    fault_frame = dataset.to_table(
        filter=ds.field("faultNumber") == 1,
        columns=["simulationRun", "sample"] + SENSORS
    ).to_pandas()
    run_ids, fault_a = to_runs(fault_frame, "Fault 1")
    onsets = sustained_onsets(fault_a, thresholds)
    origin_i = SENSORS.index(ORIGIN)
    origin_t = onsets[:, origin_i]
    origin_seen = np.isfinite(origin_t)

    ahead = np.isfinite(onsets) & (onsets < origin_t[:, None])
    tied = np.isfinite(onsets) & (onsets == origin_t[:, None])
    lead = np.where(ahead, origin_t[:, None] - onsets, np.nan)
    detected = np.isfinite(onsets)
    # Per-run sustained onset counts are an observation, not a causal edge.
    rows = []
    for i, sensor in enumerate(SENSORS):
        if i == origin_i:
            continue
        ahead_i = ahead[:, i]
        rows.append({
            "sensor": sensor,
            "healthy_sustained_detection_rate": float(healthy_detection[i]),
            "fault1_sustained_detection_rate": float(detected[:, i].mean()),
            "fault1_minus_healthy_detection_rate": float(
                detected[:, i].mean() - healthy_detection[i]
            ),
            "fraction_ahead_of_origin": float(ahead_i.mean()),
            "fraction_tied_with_origin": float(tied[:, i].mean()),
            "median_samples_ahead_when_ahead": (
                float(np.median(lead[ahead_i, i])) if ahead_i.any() else np.nan
            ),
            "fault1_median_onset_if_detected": (
                float(np.median(onsets[detected[:, i], i])) if detected[:, i].any() else np.nan
            )
        })
    competitors = pd.DataFrame(rows).sort_values(
        ["fraction_ahead_of_origin", "fault1_minus_healthy_detection_rate"],
        ascending=[False, False]
    )

    run_summary = pd.DataFrame({
        "simulationRun": run_ids,
        "origin": ORIGIN,
        "origin_detected": origin_seen,
        "origin_onset": np.where(origin_seen, origin_t, np.nan),
        "number_of_sensors_ahead": ahead.sum(axis=1),
        "number_of_other_sensors_tied": tied.sum(axis=1) - origin_seen.astype(int),
        "earliest_detected_onset": np.where(
            np.isfinite(onsets.min(axis=1)), onsets.min(axis=1), np.nan
        )
    })

    competitor_path = OUT / "v2_5_7_fault1_sensors_ahead.csv"
    run_path = OUT / "v2_5_7_fault1_run_summary.csv"
    competitors.to_csv(competitor_path, index=False)
    run_summary.to_csv(run_path, index=False)

    print(f"Fault-1 runs: {len(run_ids)}", flush=True)
    print(f"Origin {ORIGIN}: normal sustained rate {healthy_detection[origin_i]:.1%}, "
          f"Fault-1 sustained rate {origin_seen.mean():.1%}", flush=True)
    print(f"Origin median onset (detected runs): "
          f"{np.median(origin_t[origin_seen]) if origin_seen.any() else float('nan'):.1f}",
          flush=True)
    print(f"Median other sensors already abnormal before origin: "
          f"{run_summary['number_of_sensors_ahead'].median():.1f}", flush=True)
    print(f"Origin earliest (including ties): "
          f"{np.mean(origin_seen & (origin_t == onsets.min(axis=1))):.1%}", flush=True)

    print("\nTOP 15 SENSORS THAT PRECEDE THE TRUE ORIGIN", flush=True)
    fields = ["sensor", "fraction_ahead_of_origin",
              "fault1_sustained_detection_rate",
              "healthy_sustained_detection_rate",
              "fault1_minus_healthy_detection_rate",
              "median_samples_ahead_when_ahead"]
    print(competitors[fields].head(15).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ), flush=True)
    print(f"\nSaved: {competitor_path}\nSaved: {run_path}", flush=True)
    print("Diagnostic only. Earlier abnormality does not establish physical causality.",
          flush=True)
    print("V2.5.7 COMPLETE — frozen V2.5 unchanged.", flush=True)


if __name__ == "__main__":
    main()
