"""Create independent TEP testing AE-error cache from FROZEN training assets.

A small smoke test is the default: fault 1, simulationRuns 1..5.
No training, no scaler fitting, no threshold changes. Outputs go to a
SEPARATE testing folder, never to the original TEP_cache.

Smoke test:
    python -u 6.4_TEP_Testing_Error_Cache.py
Full known-origin set (after reviewing smoke output):
    python -u 6.4_TEP_Testing_Error_Cache.py --full

The final testing cache is fault-sharded so completed faults are resumable
and no 9.6-million-row DataFrame needs to be held in memory.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
IDS = ["faultNumber", "simulationRun", "sample"]
ROOT = Path("/content/drive/MyDrive")
RAW = ROOT / "PHD/Paper5_Datasets/TEP_Faulty_Testing.parquet"
MODEL_DIR = ROOT / "Paper 5/TEP"
TRAIN_CACHE = ROOT / "PHD/Router_V2_Structural_Attribution/TEP_cache"
OUTPUT_BASE = ROOT / "PHD/Router_V2_Structural_Attribution"
SCHEMA = pa.schema(
    [pa.field(c, pa.int16()) for c in IDS]
    + [pa.field(s, pa.float32()) for s in SENSORS]
)
NORMAL_QUANTILE = 0.99
SUSTAIN_K = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_loading():
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    models = []
    for i, s in enumerate(SENSORS, 1):
        path = MODEL_DIR / f"autoencoder_{s}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        model = tf.keras.models.load_model(str(path), compile=False)
        if model.input_shape[-1] != 1 or model.output_shape[-1] != 1:
            raise ValueError(f"{s}: expected single-input, single-output AE")
        models.append(model)
        if i % 13 == 0:
            print(f"Loaded {i}/52 saved models", flush=True)
    return models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true",
                   help="All 15 known-origin faults, 500 runs each")
    p.add_argument("--fault", type=int, default=1,
                   help="Fault ID for smoke mode, default 1")
    p.add_argument("--runs", type=int, default=5,
                   help="Runs for smoke mode, default 5")
    p.add_argument("--batch", type=int, default=8192)
    args = p.parse_args()
    if args.batch < 1 or args.runs < 1 or args.runs > 500:
        p.error("batch must be positive, runs must be between 1 and 500")
    if args.fault < 1 or args.fault > 20:
        p.error("fault must be between 1 and 20")
    faults = list(range(1, 16)) if args.full else [args.fault]
    n_runs = 500 if args.full else args.runs
    folder = OUTPUT_BASE / (
        "TEP_testing_error_cache_frozen_v2_5"
        if args.full else "TEP_testing_error_cache_smoke"
    )
    folder.mkdir(parents=True, exist_ok=True)
    if not RAW.is_file():
        raise FileNotFoundError(RAW)
    scaler_path = TRAIN_CACHE / "minmax_scaler.pkl"
    normal_cache = TRAIN_CACHE / "normal_error_df.parquet"
    for path in (scaler_path, normal_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    scaler = joblib.load(scaler_path)
    if getattr(scaler, "n_features_in_", None) != 52:
        raise ValueError("Training scaler does not have 52 input columns")
    if list(getattr(scaler, "feature_names_in_", SENSORS)) != SENSORS:
        raise ValueError("Training scaler feature order is incompatible")
    print("=" * 80, flush=True)
    print("TEP TESTING AE ERRORS — FROZEN MODELS AND TRAINING SCALER", flush=True)
    print("=" * 80, flush=True)
    print(f"MODE: {'FULL 15 known faults' if args.full else 'SMOKE TEST'}", flush=True)
    print(f"Faults: {faults} | runs per fault: {n_runs} | samples/run: 960", flush=True)
    print(f"Output: {folder}", flush=True)
    print("Never refitting scaler. Never changing thresholds.", flush=True)

    metadata_path = folder / "cache_metadata.json"
    metadata = {
        "source": str(RAW),
        "test_only": True,
        "source_samples_per_run": 960,
        "faults": faults,
        "runs_per_fault": n_runs,
        "sensors": SENSORS,
        "scaler_path": str(scaler_path),
        "scaler_sha256": sha256(scaler_path),
        "model_dir": str(MODEL_DIR),
        "model_sha256": {
            s: sha256(MODEL_DIR / f"autoencoder_{s}.h5") for s in SENSORS
        },
        "error_metric": "absolute error, scaled space",
        "normal_threshold_source": str(normal_cache),
        "v2_5_quantile_unchanged": NORMAL_QUANTILE,
        "v2_5_sustain_unchanged": SUSTAIN_K,
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old != metadata:
            raise RuntimeError(
                "Existing testing metadata conflicts with current assets. "
                "Refusing to mix different models, scalers or sources."
            )
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2))

    # Dataset scanner applies filter at source; reads neither all columns
    # into one DataFrame nor all 9.6m test observations simultaneously.
    data = ds.dataset(str(RAW), format="parquet")
    if any(c not in data.schema.names for c in IDS + SENSORS):
        raise ValueError("Missing testing columns")
    pending = [f for f in faults if not (folder / f"fault_{f:02d}_errors.parquet").exists()]
    if not pending:
        print("All requested fault files already exist; nothing overwritten.", flush=True)
        return
    models = model_loading()
    for fault in pending:
        out = folder / f"fault_{fault:02d}_errors.parquet"
        part = out.with_suffix(".parquet.incomplete")
        if part.exists():
            part.unlink()
        filt = (
            (ds.field("faultNumber") == fault)
            & (ds.field("simulationRun") <= n_runs)
        )
        scanner = data.scanner(columns=IDS + SENSORS, filter=filt,
                               batch_size=args.batch, use_threads=True)
        seen = np.zeros((n_runs, 960), dtype=np.uint8)
        count = 0
        writer = None
        try:
            for batch in scanner.to_batches():
                raw = batch.to_pandas()
                if raw.empty:
                    continue
                if raw[SENSORS].isna().any().any():
                    raise ValueError(f"Fault {fault}: missing sensor readings")
                r = raw["simulationRun"].to_numpy(dtype=np.int64)
                t = raw["sample"].to_numpy(dtype=np.int64)
                f = raw["faultNumber"].to_numpy(dtype=np.int64)
                if not np.all(f == fault) or np.any(r < 1) or np.any(r > n_runs) or (
                    np.any(t < 1) or np.any(t > 960)
                ):
                    raise ValueError(f"Fault {fault}: unexpected run/sample IDs")
                np.add.at(seen, (r-1, t-1), 1)
                if seen.max() > 1:
                    raise ValueError(f"Fault {fault}: repeated run/sample")
                scaled = scaler.transform(raw[SENSORS].to_numpy()).astype(np.float32)
                errors = np.empty_like(scaled, dtype=np.float32)
                for i, model in enumerate(models):
                    x = scaled[:, i:i+1]
                    # Direct inference avoids Keras .predict() retracing overhead.
                    recon = np.asarray(model(x, training=False)).reshape(-1)
                    errors[:, i] = np.abs(x[:, 0] - recon)
                result = pd.DataFrame(errors, columns=SENSORS)
                for name, vals in zip(IDS, [f, r, t]):
                    result[name] = vals.astype(np.int16)
                table = pa.Table.from_pandas(
                    result[IDS + SENSORS], schema=SCHEMA,
                    preserve_index=False, safe=True
                )
                if writer is None:
                    writer = pq.ParquetWriter(str(part), SCHEMA,
                                              compression="snappy")
                writer.write_table(table)
                count += len(raw)
                del batch, raw, scaled, errors, result, table
                if count % (args.batch * 10) < args.batch:
                    print(f"  fault {fault:02d}: {count:,}/"
                          f"{n_runs * 960:,} rows", flush=True)
            expected = n_runs * 960
            if count != expected or not np.all(seen == 1):
                raise RuntimeError(
                    f"Fault {fault}: incomplete run grid {count:,}/{expected:,}"
                )
            if writer is None:
                raise RuntimeError(f"Fault {fault}: no batches returned")
            writer.close()
            writer = None
            pf = pq.ParquetFile(part)
            if pf.metadata.num_rows != expected or pf.schema_arrow != SCHEMA:
                raise RuntimeError(f"Fault {fault}: output metadata mismatch")
            part.replace(out)
            print(f"VERIFIED fault {fault:02d}: {count:,} rows -> "
                  f"{out.name} ({out.stat().st_size/1048576:.1f} MiB)", flush=True)
        finally:
            if writer is not None:
                writer.close()
            if part.exists():
                part.unlink()
        gc.collect()
    print("=" * 80, flush=True)
    print("TESTING ERROR CACHE COMPLETE — NO TRAINING ASSETS CHANGED", flush=True)
    print(f"New fault files: {folder}", flush=True)


if __name__ == "__main__":
    main()
