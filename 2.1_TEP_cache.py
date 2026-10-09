"""
build_cache_tep.py — STAGE 0 FOR THE TENNESSEE EASTMAN PROCESS
==============================================================
Loads the 52 saved per-sensor autoencoders from TEP_Models/ and writes the
reconstruction-error cache to TEP_cache/. Trains nothing.

WHAT THESE MODELS EXPECT
------------------------
Verified by inspection: every TEP autoencoder takes ONE input and returns
ONE output. They were trained on a single scaled sensor column, NOT on the
(original, rolling_mean, rolling_std) features used by the CSTR and
three-tank models. There is therefore no rolling window anywhere in this
file, and no feature engineering.

SCALING
-------
No scalers were saved alongside these weights, so one MinMaxScaler is
fitted on the fault-free training data and then SAVED to the cache folder.
MinMaxScaler works per column, so a single fitted object holds an
independent range for each sensor. Every later block is transformed with
that saved scaler — never refitted, because refitting per file would give
each block its own reference frame and make the errors incomparable.

ERROR METRIC
------------
Absolute error between the scaled input and its reconstruction. With a
single feature there is nothing to average over. Use --inverse-transform to
express the error in physical units instead.

OUTPUT (all in TEP_cache/)
--------------------------
    normal_error_df.parquet   sensors + simulationRun
    fault_error_df.parquet    sensors + faultNumber + simulationRun + sample
    minmax_scaler.pkl         the fitted scaler, for reuse
    cache_metadata.json       settings, model hashes, timestamp

USAGE
-----
    pip install pyreadr        # needed to read .RData

Run it with no arguments at all (or press Run in VS Code) if the data sits
beside this file under its standard names:

    python build_cache_tep.py

The first run converts each RData file to parquet; later runs pick up those
copies automatically and skip the slow R reader. Override anything as needed:

    python build_cache_tep.py --fault TEP_Faulty_Testing.RData --max-runs 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────

SENSORS: List[str] = ([f"xmeas_{i}" for i in range(1, 42)]
                      + [f"xmv_{i}" for i in range(1, 12)])

FAULT_COL = "faultNumber"
RUN_COL = "simulationRun"
SAMPLE_COL = "sample"

PREDICT_BATCH = 8192      # these models are tiny; the data is not


def model_path(model_dir: str, sensor: str) -> str:
    """Where one sensor's saved autoencoder lives."""
    return os.path.join(model_dir, f"autoencoder_{sensor}.h5")


# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────

def read_rdata(path: str, object_name: Optional[str] = None) -> pd.DataFrame:
    """
    Read a data frame out of an .RData or .rds file.

    The Rieth et al. TEP release ships as RData, where one file can hold
    several named objects. When the name is not given we take the largest
    data frame, which is always the observation table rather than a small
    accompanying object.
    """
    try:
        import pyreadr
    except ImportError:
        raise SystemExit(
            "Reading .RData needs pyreadr. Install it with:\n"
            "    pip install pyreadr\n"
            "Alternatively convert the file to CSV in R:\n"
            "    load('TEP_Faulty_Training.RData')\n"
            "    write.csv(faulty_training, 'TEP_Faulty_Training.csv', "
            "row.names=FALSE)")

    result = pyreadr.read_r(path)
    if not result:
        raise ValueError(f"{path} contains no readable R objects.")
    names = list(result.keys())
    if object_name is not None:
        if object_name not in result:
            raise KeyError(f"'{object_name}' not in {path}. Available: {names}")
        chosen = object_name
    else:
        chosen = max(names, key=lambda k: len(result[k]))
    if len(names) > 1:
        print(f"  {path} holds {names}; using '{chosen}'")
    return result[chosen]


def prefer_parquet(path: str) -> str:
    """
    Swap an RData path for its parquet copy when one already exists.

    Reading RData is slow and holds the whole table in memory as doubles. If
    a previous run already converted the file, using that copy is both
    faster and lighter, with identical contents.
    """
    if not path.lower().endswith((".rdata", ".rda", ".rds")):
        return path
    converted = os.path.splitext(path)[0] + ".parquet"
    if os.path.exists(converted):
        print(f"  using existing parquet copy instead of {path}")
        return converted
    return path


def read_table(path: str, object_name: Optional[str] = None) -> pd.DataFrame:
    """Read a TEP table from RData, csv, parquet or xlsx."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    lower = path.lower()
    if lower.endswith((".rdata", ".rda", ".rds")):
        return read_rdata(path, object_name)
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bring R's column naming into line with the names used here.

    R converts dots and cases inconsistently across exports, so 'xmeas.1',
    'XMEAS_1' and 'xmeas_1' can all appear for the same variable. Renaming
    once here means the rest of the file only ever sees one spelling.
    """
    renamed = {}
    for column in df.columns:
        clean = str(column).strip().replace(".", "_").lower()
        if clean in ("faultnumber", "fault_number"):
            clean = FAULT_COL
        elif clean in ("simulationrun", "simulation_run"):
            clean = RUN_COL
        elif clean == "sample":
            clean = SAMPLE_COL
        renamed[column] = clean
    df = df.rename(columns=renamed)

    # R stores everything as doubles, so faultNumber arrives as 1.0 rather
    # than 1. Left alone that propagates into the cache and into every later
    # lookup keyed by fault number, where 1.0 will not match 1.
    for column in (FAULT_COL, RUN_COL, SAMPLE_COL):
        if column in df.columns:
            df[column] = df[column].astype("int64")
    return df


def save_converted(df: pd.DataFrame, source: str) -> None:
    """
    Write a parquet copy beside the RData file.

    Reading RData is slow and memory-hungry; doing it once and pointing
    later runs at the parquet avoids paying that cost repeatedly.
    """
    target = os.path.splitext(source)[0] + ".parquet"
    df.to_parquet(target, index=False)
    print(f"  converted copy -> {target}")


def validate_columns(df: pd.DataFrame, path: str, need_fault: bool) -> None:
    """Fail early and specifically if expected columns are absent."""
    missing = [s for s in SENSORS if s not in df.columns]
    if missing:
        raise KeyError(
            f"{path} is missing {len(missing)} sensor column(s), e.g. "
            f"{missing[:5]}.\n  Found: {list(df.columns)[:12]}...")
    required = [RUN_COL] + ([FAULT_COL] if need_fault else [])
    for column in required:
        if column not in df.columns:
            raise KeyError(
                f"{path} has no '{column}' column. Run structure is required: "
                f"the attribution method aggregates across runs with a median, "
                f"which is meaningless on concatenated data.")


# ──────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────

def load_models(model_dir: str) -> Dict[str, object]:
    """
    Load all 52 autoencoders once, CPU-only and uncompiled.

    compile=False skips optimizer deserialisation, which is what breaks
    legacy .h5 files under newer Keras versions.
    """
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    try:
        import tf_keras
        loader = lambda p: tf_keras.models.load_model(p, compile=False)
        print("  (using tf-keras legacy loader)")
    except ImportError:
        loader = lambda p: tf.keras.models.load_model(p, compile=False)

    models = {}
    for sensor in SENSORS:
        path = model_path(model_dir, sensor)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model for '{sensor}': {path}")
        models[sensor] = loader(path)
    print(f"  loaded {len(models)} autoencoders from {model_dir}")
    return models


def assert_single_input(models: Dict[str, object]) -> None:
    """
    Confirm every model takes exactly one input feature.

    If any model expects more, it was trained on a different representation
    and this builder would silently produce meaningless errors.
    """
    for sensor, model in models.items():
        shape = model.input_shape
        n_in = shape[-1] if isinstance(shape, tuple) else shape[0][-1]
        if n_in != 1:
            raise ValueError(
                f"Model for '{sensor}' expects {n_in} inputs, not 1. "
                f"These weights were trained on a different representation — "
                f"do not proceed.")
    print(f"  input check passed: all {len(models)} models take a single column")


# ──────────────────────────────────────────────────────────────────────────
# SCALING
# ──────────────────────────────────────────────────────────────────────────

def fit_scaler(normal_df: pd.DataFrame):
    """Fit one MinMaxScaler on fault-free data across all sensor columns."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(normal_df[SENSORS].values)
    print(f"  fitted MinMaxScaler on {len(normal_df)} fault-free rows")
    return scaler


def save_scaler(scaler, out_dir: str) -> str:
    """Persist the scaler so this cache can be reproduced or extended."""
    import joblib

    path = os.path.join(out_dir, "minmax_scaler.pkl")
    joblib.dump(scaler, path)
    print(f"  scaler -> {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION ERROR
# ──────────────────────────────────────────────────────────────────────────

def errors_for_block(df: pd.DataFrame, models: Dict[str, object], scaler,
                     inverse: bool) -> pd.DataFrame:
    """
    Per-sensor reconstruction error for one block of rows.

    Each sensor's scaled column is pushed through its own autoencoder and the
    absolute difference is kept. Results are float32 to halve the memory
    footprint of a dataset this size.
    """
    scaled = scaler.transform(df[SENSORS].values).astype(np.float32)
    errors = np.empty_like(scaled)
    for index, sensor in enumerate(SENSORS):
        column = scaled[:, index].reshape(-1, 1)
        predicted = models[sensor].predict(column, batch_size=PREDICT_BATCH,
                                           verbose=0)
        errors[:, index] = np.abs(column - np.asarray(predicted)).ravel()
    if inverse:
        # Undo the scaling so the error carries physical units. The offset
        # cancels in a difference, so only the range matters.
        span = (scaler.data_max_ - scaler.data_min_).astype(np.float32)
        errors = errors * span
    return pd.DataFrame(errors, columns=SENSORS, index=df.index)


def build_normal_errors(normal_df: pd.DataFrame, models, scaler,
                        inverse: bool) -> pd.DataFrame:
    """Errors on fault-free operation — the baseline every z-score uses."""
    print("\nBuilding NORMAL error cache...")
    errors = errors_for_block(normal_df, models, scaler, inverse)
    errors[RUN_COL] = normal_df[RUN_COL].values
    print(f"  {errors.shape[0]} rows, {normal_df[RUN_COL].nunique()} runs")
    return errors.reset_index(drop=True)


def build_fault_errors(fault_df: pd.DataFrame, models, scaler, inverse: bool,
                       max_runs: Optional[int]) -> pd.DataFrame:
    """
    Errors on fault data, one fault at a time.

    Processing per fault keeps peak memory to a single fault's worth of rows
    and makes progress visible. max_runs caps how many simulation runs are
    used per fault — the lever to pull if the machine runs out of memory.
    """
    print("\nBuilding FAULT error cache...")
    blocks = []
    for fault_number in sorted(fault_df[FAULT_COL].unique()):
        if fault_number == 0:
            continue
        subset = fault_df[fault_df[FAULT_COL] == fault_number]
        if max_runs is not None:
            keep = sorted(subset[RUN_COL].unique())[:max_runs]
            subset = subset[subset[RUN_COL].isin(keep)]
        errors = errors_for_block(subset, models, scaler, inverse)
        for column in (FAULT_COL, RUN_COL, SAMPLE_COL):
            if column in subset.columns:
                errors[column] = subset[column].values
        blocks.append(errors.reset_index(drop=True))
        print(f"    fault {fault_number:>3}: {len(errors):>7} rows, "
              f"{subset[RUN_COL].nunique():>4} runs")
    if not blocks:
        raise SystemExit("No fault rows found — every faultNumber was 0.")
    return pd.concat(blocks, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────

def file_digest(path: str) -> str:
    """Short content hash, so the cache records which weights built it."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def write_metadata(out_dir: str, model_dir: str, sources: Dict[str, str],
                   settings: dict) -> None:
    """Record everything needed to reproduce this cache."""
    metadata = dict(
        dataset="TEP",
        built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_sensors=len(SENSORS),
        sensors=SENSORS,
        representation="single scaled column per sensor (no rolling features)",
        scaler="MinMaxScaler fitted on fault-free data, saved as minmax_scaler.pkl",
        error_metric=settings["error_metric"],
        source_files=sources,
        max_runs_per_fault=settings["max_runs"],
        faults_present=settings["faults_present"],
        model_digests={s: file_digest(model_path(model_dir, s))
                       for s in SENSORS
                       if os.path.exists(model_path(model_dir, s))},
    )
    path = os.path.join(out_dir, "cache_metadata.json")
    with open(path, "w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    print(f"  metadata -> {path}")


def save_cache(out_dir: str, normal_errors: pd.DataFrame,
               fault_errors: pd.DataFrame) -> None:
    """Write the two parquet files the attribution pipeline reads."""
    normal_path = os.path.join(out_dir, "normal_error_df.parquet")
    fault_path = os.path.join(out_dir, "fault_error_df.parquet")
    normal_errors.to_parquet(normal_path, index=False)
    fault_errors.to_parquet(fault_path, index=False)
    print(f"  normal_error_df {normal_errors.shape} -> {normal_path}")
    print(f"  fault_error_df  {fault_errors.shape} -> {fault_path}")


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the TEP error cache from saved weights.")
    parser.add_argument("--normal", default="TEP_FaultFree_Training.RData",
                        help="fault-free data (.RData, .csv, .parquet, .xlsx)")
    parser.add_argument("--fault", default="TEP_Faulty_Training.RData",
                        help="faulty data (.RData, .csv, .parquet, .xlsx)")
    parser.add_argument("--normal-object", default=None,
                        help="name of the R object inside the normal RData file")
    parser.add_argument("--fault-object", default=None,
                        help="name of the R object inside the faulty RData file")
    parser.add_argument("--no-save-parquet", dest="save_parquet",
                        action="store_false",
                        help="skip writing parquet copies beside the RData files")
    parser.add_argument("--models", default="TEP_Models")
    parser.add_argument("--out", default="TEP_cache")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="cap simulation runs per fault (memory relief)")
    parser.add_argument("--inverse-transform", action="store_true",
                        help="express errors in physical units instead of scaled")
    args = parser.parse_args()

    print("=" * 70)
    print("TEP STAGE 0 — reconstruction-error cache from SAVED weights")
    print("=" * 70)

    os.makedirs(args.out, exist_ok=True)

    normal_source = prefer_parquet(args.normal)
    normal_df = normalise_column_names(read_table(normal_source,
                                                  args.normal_object))
    if FAULT_COL in normal_df.columns:
        normal_df = normal_df[normal_df[FAULT_COL] == 0]
    validate_columns(normal_df, normal_source, need_fault=False)
    print(f"  normal data: {normal_df.shape} from {normal_source}")
    if args.save_parquet and normal_source == args.normal:
        save_converted(normal_df, args.normal)

    fault_source = prefer_parquet(args.fault)
    fault_df = normalise_column_names(read_table(fault_source,
                                                 args.fault_object))
    validate_columns(fault_df, fault_source, need_fault=True)
    print(f"  fault data : {fault_df.shape} from {fault_source}")
    if args.save_parquet and fault_source == args.fault:
        save_converted(fault_df, args.fault)

    models = load_models(args.models)
    assert_single_input(models)

    scaler = fit_scaler(normal_df)
    save_scaler(scaler, args.out)

    normal_errors = build_normal_errors(normal_df, models, scaler,
                                        args.inverse_transform)
    fault_errors = build_fault_errors(fault_df, models, scaler,
                                      args.inverse_transform, args.max_runs)

    print()
    save_cache(args.out, normal_errors, fault_errors)
    write_metadata(
        args.out, args.models,
        {"normal": args.normal, "fault": args.fault},
        {"error_metric": ("absolute error, physical units"
                          if args.inverse_transform
                          else "absolute error, scaled space"),
         "max_runs": args.max_runs,
         "faults_present": sorted(int(f) for f in
                                  fault_errors[FAULT_COL].unique())})

    print(f"\nDone. Cache written to {args.out}/")


if __name__ == "__main__":
    main()