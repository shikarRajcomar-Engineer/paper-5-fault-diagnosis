"""TEP frozen-model testing preflight (read-only).

Checks:
  * saved training scaler and 52 model files exist
  * legacy H5 model compatibility
  * predictions exactly reproduce cached normal-training AE errors
No scaler fitting, training, or testing cache creation.
"""
from pathlib import Path
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import h5py
import joblib
import numpy as np
import pandas as pd

SENSORS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]
ROOT = Path("/content/drive/MyDrive")
BASE = ROOT / "PHD/Router_V2_Structural_Attribution/TEP_cache"
RAW = ROOT / "PHD/Paper5_Datasets/TEP_FaultFree_Training.parquet"
MODELS = ROOT / "Paper 5/TEP"


def load_legacy(path):
    """Use legacy tf-keras if installed, otherwise standard Keras H5 loader.

    Do not silently substitute another architecture or weights.
    """
    try:
        import tf_keras
        return tf_keras.models.load_model(str(path), compile=False), "tf_keras"
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as e:
        print(f"Legacy loader failed for {path.name}: {type(e).__name__}: {e}", flush=True)
    import tensorflow as tf
    try:
        return tf.keras.models.load_model(str(path), compile=False), "tf.keras"
    except Exception as e:
        raise RuntimeError(
            f"Cannot deserialize {path.name} safely. "
            f"Need verified model reconstruction from original H5 configuration. "
            f"Error: {type(e).__name__}: {e}"
        ) from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS))
    ap.add_argument("--reference-rows", type=int, default=256)
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    print("=" * 80, flush=True)
    print("TEP TESTING PREFLIGHT — FROZEN TRAINING MODELS AND SCALER", flush=True)
    print("=" * 80, flush=True)
    scaler_path = BASE / "minmax_scaler.pkl"
    reference_cache = BASE / "normal_error_df.parquet"
    for path in (model_dir, scaler_path, RAW, reference_cache):
        if not path.exists():
            raise FileNotFoundError(f"Required frozen input missing: {path}")
        print(f"Found {path}", flush=True)
    models = [model_dir / f"autoencoder_{s}.h5" for s in SENSORS]
    missing = [str(p) for p in models if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} saved models missing; examples: {missing[:5]}")
    print("All 52 saved H5 files exist.", flush=True)

    scaler = joblib.load(scaler_path)
    if getattr(scaler, "n_features_in_", None) != 52:
        raise ValueError("Saved scaler does not expect 52 features")
    raw = pd.read_parquet(RAW, columns=SENSORS).iloc[:args.reference_rows].copy()
    cached = pd.read_parquet(reference_cache, columns=SENSORS).iloc[:args.reference_rows].copy()
    assert len(raw) == len(cached) == args.reference_rows
    scaled = scaler.transform(raw[SENSORS].to_numpy()).astype(np.float32)
    print("Loaded frozen MinMaxScaler; fitting is disabled.", flush=True)

    metadata_file = BASE / "cache_metadata.json"
    if metadata_file.exists():
        info=json.loads(metadata_file.read_text())
        print(f"Training-cache error metric: {info.get('error_metric')}",flush=True)
        if info.get('error_metric') not in (
            "absolute error, scaled space", None
        ):
            raise ValueError("Training cache error metric is not scaled absolute error")

    summaries=[]
    for i, (sensor, path) in enumerate(zip(SENSORS, models)):
        model, loader = load_legacy(path)
        if model.input_shape[-1] != 1 or model.output_shape[-1] != 1:
            raise ValueError(f"{sensor}: unexpected model input/output dimensions")
        x = scaled[:, i:i+1]
        predicted = np.asarray(model.predict(x, batch_size=256, verbose=0)).reshape(-1)
        error = np.abs(x.reshape(-1) - predicted)
        expected = cached[sensor].to_numpy(dtype=np.float32)
        difference = np.abs(error - expected)
        max_error = float(difference.max())
        summaries.append(max_error)
        print(f"[{i+1:02d}/52] {sensor:<10} {loader:<9} "
              f"max cache disagreement={max_error:.3e}", flush=True)
        if not np.isfinite(max_error) or max_error > 1e-5:
            raise RuntimeError(
                f"STOP: {sensor} fails training-cache reproduction. "
                "Do not process testing data until inputs or loader match."
            )
        del model
    print("=" * 80, flush=True)
    print(f"PREFLIGHT PASSED: all 52 sensors; worst difference={max(summaries):.3e}",
          flush=True)
    print("Frozen test-cache creation is safe to attempt. No files changed.",flush=True)


if __name__ == "__main__":
    main()
