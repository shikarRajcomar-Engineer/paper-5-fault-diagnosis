"""
CSTR_cache.py — STAGE 0 FOR THE CSTR
====================================
Loads the 7 saved per-sensor autoencoders from CSTR_Models/ and writes the
reconstruction-error cache to CSTR_cache/. Trains nothing.

INPUTS
    CSTR_FaultFree_TrainingData.xlsx    normal baseline
    CSTR_Faults/*.xlsx                  one file per fault run

OUTPUTS (all in CSTR_cache/)
    normal_error_df.parquet   sensors + simulationRun
    fault_error_df.parquet    sensors + simulationRun + true_cause
                              + amplitude + sample
    cache_metadata.json       settings, model hashes, timestamp

WHAT THE MODELS EXPECT
    3 inputs per sensor: original, rolling_mean, rolling_std (window 10).
    TEP is different - its models take 1 raw column. That difference lives
    here in Stage 0. Both plants still hand the attribution method the same
    thing: an error matrix plus run labels.

SCALERS
    AE_scaler_feature_Stdscalar*.pkl are used with .transform() ONLY, never
    refitted. Refitting per file gives every run its own reference frame,
    which is the scale drift noted in the CSTR calibration work.

RUN IT
    python CSTR_cache.py
    python CSTR_cache.py --faults-dir CSTR_Faults --normal CSTR_FaultFree_TrainingData.xlsx
"""

import argparse
import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------------------------------------

SENSORS = ['Ci', 'Ti', 'T', 'Qc', 'Tci', 'Tc', 'C']

ROLLING_WINDOW = 10          # must match how the weights were trained
ROLLING_MIN_PERIODS = 5

SENSOR_COL_SLICE = slice(2, 9)      # fallback: columns 2-8 of the raw sheet
CLASS_COL = 'Class'
FEATURE_NAMES = ['original', 'rolling_mean', 'rolling_std']

DEFAULT_NORMAL = 'CSTR_FaultFree_TrainingData.xlsx'
DEFAULT_FAULTS_DIR = 'CSTR_Faults'

# Tsp is the temperature setpoint; its faults present as a T origin.
NAME_ALIASES = {'tsp': 'T'}


def model_file(model_dir, index):
    """Path to one sensor's saved autoencoder."""
    return os.path.join(model_dir, f'AE_model_feature_Stdscalar{index}.h5')


def scaler_file(model_dir, index):
    """Path to one sensor's saved scaler."""
    return os.path.join(model_dir, f'AE_scaler_feature_Stdscalar{index}.pkl')


# ----------------------------------------------------------------------------------------------------------
# 1. Startup checks
# ----------------------------------------------------------------------------------------------------------

def check_dependencies():
    """
    Confirm every package is importable before any slow work starts.
    Loading models and Excel files takes minutes; finding out afterwards
    that sklearn is missing wastes all of it.
    """
    required = {'sklearn': 'scikit-learn', 'joblib': 'joblib',
                'pyarrow': 'pyarrow', 'openpyxl': 'openpyxl',
                'tensorflow': 'tensorflow'}
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        raise SystemExit(
            f'Missing required package(s): {", ".join(missing)}\n'
            f'Install them with:\n    pip install {" ".join(missing)}\n'
            f'Use the same interpreter that runs this script.')


# ----------------------------------------------------------------------------------------------------------
# 2. Reading Excel sheets
# ----------------------------------------------------------------------------------------------------------

def extract_sensor_columns(raw, filename):
    """
    Pull the 7 sensor columns out of a raw sheet.

    Named columns are preferred: if the sheet already has headers called
    Ci, Ti, T and so on, those are used directly. Only when the names are
    absent do we fall back to positions 2-8. Guessing by position when the
    names are available is how a different column order in one file ends up
    silently scrambling which sensor is which.
    """
    if all(s in raw.columns for s in SENSORS):
        return raw[SENSORS].copy()

    data = raw.iloc[:, SENSOR_COL_SLICE].copy()
    if data.shape[1] != len(SENSORS):
        raise ValueError(
            f'expected {len(SENSORS)} sensor columns at positions 2-8, '
            f'found {data.shape[1]} in {filename}')
    data.columns = SENSORS
    return data


def read_sheet(path):
    """
    Read one Excel run and return its sensor columns plus any Class label.

    Files without a Class column are treated as entirely fault data, which
    is the right reading for a folder where every file IS a fault run.
    """
    raw = pd.read_excel(path, engine='openpyxl')
    data = extract_sensor_columns(raw, os.path.basename(path))
    data[CLASS_COL] = raw[CLASS_COL].values if CLASS_COL in raw.columns else 1
    return data


# ----------------------------------------------------------------------------------------------------------
# 3. Working out what each fault file contains
# ----------------------------------------------------------------------------------------------------------

def infer_true_cause(filename):
    """
    Work out the origin sensor from the filename.

    Filenames start with the faulted variable: 'Ci amp 20 Fault 1',
    'Qc_SensorBias'. Anything that does not resolve to a known sensor
    returns None and gets reported rather than guessed at.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    first = re.split(r'[\s_]+', stem)[0].strip().lower()
    if first in NAME_ALIASES:
        return NAME_ALIASES[first]
    for sensor in SENSORS:
        if first == sensor.lower():
            return sensor
    return None


def parse_amplitude(filename):
    """Recover the fault amplitude from the filename, for scenario grouping."""
    match = re.search(r'amp\s*(\d+)', filename, flags=re.IGNORECASE)
    if match:
        return f'amp{match.group(1)}'
    if 'sensorbias' in filename.lower():
        return 'bias'
    return 'unspecified'


def discover_fault_files(faults_dir):
    """
    Find every fault run in the folder as (path, true_cause) pairs.

    Files whose origin cannot be read from the name are listed, not
    silently dropped into an unlabelled scenario, so you can rename them
    or add an alias.
    """
    if not os.path.isdir(faults_dir):
        raise SystemExit(
            f'Fault folder not found: {faults_dir}\n'
            f'  Point at it with --faults-dir <folder>.')

    candidates = sorted(glob.glob(os.path.join(faults_dir, '*.xlsx'))
                        + glob.glob(os.path.join(faults_dir, '*.xls')))
    if not candidates:
        raise SystemExit(f'No Excel files found in {faults_dir}.')

    resolved, unresolved = [], []
    for path in candidates:
        cause = infer_true_cause(path)
        if cause is None:
            unresolved.append(os.path.basename(path))
        else:
            resolved.append((path, cause))

    print(f'  found {len(resolved)} fault file(s) in {faults_dir}')
    if unresolved:
        print(f'  UNRECOGNISED ORIGIN, skipped {len(unresolved)}: '
              f'{unresolved[:8]}{"..." if len(unresolved) > 8 else ""}')
    if not resolved:
        raise SystemExit('No fault file names could be resolved to a sensor.')
    return resolved


# ----------------------------------------------------------------------------------------------------------
# 4. Loading the saved models and scalers
# ----------------------------------------------------------------------------------------------------------

def load_models(model_dir):
    """Load all 7 autoencoders once, CPU only and uncompiled."""
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass

    try:
        import tf_keras
        loader = lambda p: tf_keras.models.load_model(p, compile=False)
        print('  (using tf-keras legacy loader)')
    except ImportError:
        loader = lambda p: tf.keras.models.load_model(p, compile=False)

    models = {}
    for index, sensor in enumerate(SENSORS):
        path = model_file(model_dir, index)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model for '{sensor}': {path}")
        models[sensor] = loader(path)
    print(f'  loaded {len(models)} autoencoders from {model_dir}')
    return models


def assert_three_inputs(models):
    """
    Confirm every model takes 3 features.
    A model expecting a different number was trained on another
    representation, and this builder would produce meaningless errors.
    """
    for sensor, model in models.items():
        shape = model.input_shape
        n_in = shape[-1] if isinstance(shape, tuple) else shape[0][-1]
        if n_in != len(FEATURE_NAMES):
            raise ValueError(
                f"Model for '{sensor}' expects {n_in} inputs, not "
                f'{len(FEATURE_NAMES)}. Do not proceed.')
    print(f'  input check passed: all {len(models)} models take '
          f'{len(FEATURE_NAMES)} features')


def load_scalers(model_dir):
    """Load the per-sensor scalers saved during model construction."""
    import joblib

    scalers = {}
    for index, sensor in enumerate(SENSORS):
        path = scaler_file(model_dir, index)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing scaler for '{sensor}': {path}")
        scalers[sensor] = joblib.load(path)
    print(f'  loaded {len(scalers)} scalers')
    return scalers


# ----------------------------------------------------------------------------------------------------------
# 5. Feature construction and reconstruction error
# ----------------------------------------------------------------------------------------------------------

def create_features(sensor_data):
    """
    Build the 3-feature representation the autoencoders expect.
    Rows where the rolling statistics are undefined are dropped, so the
    result is shorter than the input; the caller keeps the surviving index
    to realign labels.
    """
    features_df = pd.DataFrame({
        'original': sensor_data,
        'rolling_mean': sensor_data.rolling(
            ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean(),
        'rolling_std': sensor_data.rolling(
            ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).std(),
    })
    return features_df.dropna()


def reconstruction_error(model, scaler, features_df):
    """
    Per-timestep MAE between input and reconstruction, in original units.
    Both sides are inverse-transformed first so the error carries physical
    meaning rather than scaler units.
    """
    scaled = scaler.transform(features_df.values)
    predicted = model.predict(scaled, verbose=0)
    return np.mean(np.abs(scaler.inverse_transform(predicted)
                          - scaler.inverse_transform(scaled)), axis=1)


def errors_for_sheet(path, models, scalers):
    """
    Reconstruction errors for one Excel run, Class label realigned.
    Every sensor loses the same leading rows to the rolling window, so the
    first sensor's surviving index is reused for the labels.
    """
    data = read_sheet(path)
    errors, kept_index = {}, None
    for sensor in SENSORS:
        features_df = create_features(data[sensor])
        if kept_index is None:
            kept_index = features_df.index
        errors[sensor] = reconstruction_error(models[sensor],
                                              scalers[sensor], features_df)
    frame = pd.DataFrame(errors, index=kept_index).astype(np.float32)
    frame[CLASS_COL] = data.loc[kept_index, CLASS_COL].values
    return frame


# ----------------------------------------------------------------------------------------------------------
# 6. Building the two halves of the cache
# ----------------------------------------------------------------------------------------------------------

def build_normal_cache(normal_path, models, scalers):
    """
    Errors on the fault-free training file - the baseline every z-score
    uses. Any Class==1 rows in this file are dropped: a baseline built
    from fault data is not a baseline.
    """
    print('\nBuilding NORMAL error cache...')
    errors = errors_for_sheet(normal_path, models, scalers)

    n_before = len(errors)
    errors = errors[errors[CLASS_COL] == 0].drop(columns=[CLASS_COL]).copy()
    if len(errors) < n_before:
        print(f'  dropped {n_before - len(errors)} non-normal row(s) '
              f'from the fault-free file')
    if errors.empty:
        raise SystemExit(f'{normal_path} produced no normal rows.')

    errors['simulationRun'] = os.path.basename(normal_path)
    print(f'  {len(errors)} rows from {os.path.basename(normal_path)}')
    return errors.reset_index(drop=True)


def build_fault_cache(fault_files, models, scalers):
    """
    Errors on every fault run, tagged with origin, amplitude and source
    file so scenarios can be reassembled downstream.
    """
    print('\nBuilding FAULT error cache...')
    blocks = []
    for path, true_cause in fault_files:
        name = os.path.basename(path)
        try:
            errors = errors_for_sheet(path, models, scalers)
        except (KeyError, ValueError) as exc:
            print(f'    SKIPPED {name}: {exc}')
            continue

        fault = errors[errors[CLASS_COL] == 1].drop(columns=[CLASS_COL]).copy()
        if fault.empty:
            print(f'    SKIPPED {name}: no fault rows')
            continue

        fault['simulationRun'] = name
        fault['true_cause'] = true_cause
        fault['amplitude'] = parse_amplitude(name)
        fault['sample'] = np.arange(len(fault))
        blocks.append(fault.reset_index(drop=True))
        print(f'    {name:<34} {len(fault):>5} rows   origin {true_cause}')

    if not blocks:
        raise SystemExit('No fault file produced usable rows.')
    return pd.concat(blocks, ignore_index=True)


# ----------------------------------------------------------------------------------------------------------
# 7. Saving the cache
# ----------------------------------------------------------------------------------------------------------

def file_digest(path):
    """Short content hash, so the cache records which weights built it."""
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def save_cache(out_dir, normal_errors, fault_errors):
    """Write the two parquet files the attribution pipeline reads."""
    normal_path = os.path.join(out_dir, 'normal_error_df.parquet')
    fault_path = os.path.join(out_dir, 'fault_error_df.parquet')
    normal_errors.to_parquet(normal_path, index=False)
    fault_errors.to_parquet(fault_path, index=False)
    print(f'  normal_error_df {normal_errors.shape} -> {normal_path}')
    print(f'  fault_error_df  {fault_errors.shape} -> {fault_path}')


def write_metadata(out_dir, model_dir, sources, fault_errors):
    """Record everything needed to reproduce this cache."""
    scenarios = sorted({f'{cause}|{amp}' for cause, amp
                        in zip(fault_errors['true_cause'],
                               fault_errors['amplitude'])})
    metadata = dict(
        dataset='CSTR',
        built_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        n_sensors=len(SENSORS),
        sensors=SENSORS,
        representation=f'{FEATURE_NAMES}, window {ROLLING_WINDOW}, '
                       f'min_periods {ROLLING_MIN_PERIODS}',
        scaler='saved scalers, transform-only',
        error_metric='MAE over features, inverse-transformed',
        source_files=sources,
        n_fault_runs=int(fault_errors['simulationRun'].nunique()),
        scenarios=scenarios,
        origins_covered=sorted(fault_errors['true_cause'].unique().tolist()),
        model_digests={s: file_digest(model_file(model_dir, i))
                       for i, s in enumerate(SENSORS)
                       if os.path.exists(model_file(model_dir, i))},
    )
    path = os.path.join(out_dir, 'cache_metadata.json')
    with open(path, 'w') as handle:
        json.dump(metadata, handle, indent=2, default=str)
    print(f'  metadata -> {path}')


# ----------------------------------------------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build the CSTR error cache from saved weights.')
    parser.add_argument('--normal', default=DEFAULT_NORMAL,
                        help='fault-free training Excel file')
    parser.add_argument('--faults-dir', default=DEFAULT_FAULTS_DIR,
                        help='folder holding the fault Excel files')
    parser.add_argument('--models', default='CSTR_Models')
    parser.add_argument('--out', default='CSTR_cache')
    args = parser.parse_args()

    print('=' * 70)
    print('CSTR STAGE 0 - reconstruction-error cache from SAVED weights')
    print('=' * 70)

    check_dependencies()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.normal):
        raise SystemExit(f'Fault-free file not found: {args.normal}')
    print(f'  normal file: {args.normal}')
    fault_files = discover_fault_files(args.faults_dir)

    models = load_models(args.models)
    assert_three_inputs(models)
    scalers = load_scalers(args.models)

    normal_errors = build_normal_cache(args.normal, models, scalers)
    fault_errors = build_fault_cache(fault_files, models, scalers)

    print()
    save_cache(args.out, normal_errors, fault_errors)
    write_metadata(args.out, args.models,
                   {'normal': args.normal, 'faults_dir': args.faults_dir},
                   fault_errors)

    print(f'\nDone. Cache written to {args.out}/')
    print(f'  {len(fault_errors["simulationRun"].unique())} fault runs, '
          f'{fault_errors["true_cause"].nunique()} distinct origins')


if __name__ == '__main__':
    main()