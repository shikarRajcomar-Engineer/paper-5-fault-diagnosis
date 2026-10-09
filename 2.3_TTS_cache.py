"""
TTS_cache.py — STAGE 0 FOR THE THREE-TANK SYSTEM (DTS200)
=========================================================
Loads the 5 saved per-sensor autoencoders from TTS_Models/ and writes the
reconstruction-error cache to TTS_cache/. Trains nothing.

INPUTS (the DESIGNED files, not the 7-column ones)
    TTS_Dataset/[designed_train].mat   normal (16008 x 5)
    TTS_Dataset/[designed_test].mat    fault01..fault12 (2001 x 5 each)

WHY THE DESIGNED FILES
    The 7-column copies carry two redundant columns and, worse, fault01
    and fault02 are byte-identical there - so that copy holds 11 distinct
    faults, not 12. The designed files are 5 clean columns in exactly the
    order the models expect, with all 12 faults distinct.

COLUMNS
    [h1, h2, h3, Q1, Q2] at positions 0-4. No columns are dropped.
    Confirmed two ways: the pre-fault means match what each saved scaler
    was fitted on to within 0.06%, and the two clean process faults are
    unambiguous - fault07 (tank 1 leak) moves only h1, fault08 (tank 2
    leak) moves only h2.

WHAT THE MODELS EXPECT
    3 inputs per sensor: original, rolling_mean, rolling_std. The window
    is verified against the saved scalers at startup rather than trusted,
    because feeding a model features built at the wrong window produces
    errors that measure the mismatch instead of the fault.

OUTPUTS (all in TTS_cache/)
    normal_error_df.parquet   sensors + simulationRun
    fault_error_df.parquet    sensors + faultNumber + simulationRun
                              + sample + true_cause + fault_type
    cache_metadata.json       settings, model hashes, timestamp

RUN IT
    python TTS_cache.py
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------------------------------------------

SENSORS = ['h1', 'h2', 'h3', 'Q1', 'Q2']
KEEP_COLS = [0, 1, 2, 3, 4]        # the designed files need no dropping

ROLLING_WINDOW = 20                # verified against the scalers at startup
ROLLING_MIN_PERIODS = 10
CANDIDATE_WINDOWS = [5, 10, 15, 20, 25, 30]

FAULT_ONSET = 200                  # samples before this index are normal
FEATURE_NAMES = ['original', 'rolling_mean', 'rolling_std']

DEFAULT_DATA_DIR = 'TTS_Dataset'
TRAIN_FILE = '[designed_train].mat'
TEST_FILE = '[designed_test].mat'

# ----------------------------------------------------------------------------------------------------------
# GROUND TRUTH - EDIT HERE IF IT CHANGES
# ----------------------------------------------------------------------------------------------------------
# Derived from the fault equations in the repo README. f1 enters both the
# h1 and h2 equations, so it has two valid origins; f10 couples tanks 1
# and 3 through Q13; f12 couples tanks 3 and 2 through Q32.
#
# fault06 IS DISPUTED. The truth below says h3, but the data disagrees:
# h3 shifts 1.0 units and varies 3.9x after onset, while Q2 shifts 6.0
# units against a predicted 7.2 and varies 23.5x. Two independent measures
# both point at Q2. The README lists faults 4-6 as "Sensor" faults without
# naming the sensor, so h3 was always an assumption. Left as h3 for now so
# the cache does not bake in an unverified change; revisit once attribution
# has run and can be used as independent evidence.
FAULT_TRUTH = {
    1: ['h1', 'h2'], 2: ['h1'], 3: ['h2'],
    4: ['h1'], 5: ['h2'], 6: ['h3'],
    7: ['h1'], 8: ['h2'], 9: ['h3'],
    10: ['h1', 'h3'], 11: ['h2'], 12: ['h2', 'h3'],
}

FAULT_TYPE = {
    1: 'Actuator', 2: 'Actuator', 3: 'Actuator',
    4: 'Sensor', 5: 'Sensor', 6: 'Sensor',
    7: 'Process', 8: 'Process', 9: 'Process',
    10: 'Process', 11: 'Process', 12: 'Process',
}

DISPUTED_FAULTS = {6: 'ground truth says h3; the data points at Q2'}


def model_file(model_dir, index):
    """Path to one sensor's saved autoencoder."""
    return os.path.join(model_dir, f'TTS_model_{index}.h5')


def scaler_file(model_dir, index):
    """Path to one sensor's saved scaler."""
    return os.path.join(model_dir, f'TTS_scaler_{index}.pkl')


# ----------------------------------------------------------------------------------------------------------
# 1. Startup checks
# ----------------------------------------------------------------------------------------------------------

def check_dependencies():
    """
    Confirm every package is importable before any slow work starts.
    Loading models takes time; finding out afterwards that something is
    missing wastes all of it.
    """
    required = {'sklearn': 'scikit-learn', 'joblib': 'joblib',
                'pyarrow': 'pyarrow', 'scipy': 'scipy',
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
# 2. Loading the .mat data
# ----------------------------------------------------------------------------------------------------------

def load_normal(data_dir):
    """Load the normal-operation array from the designed training file."""
    from scipy.io import loadmat

    path = os.path.join(data_dir, TRAIN_FILE)
    if not os.path.exists(path):
        raise SystemExit(f'Not found: {path}')
    raw = loadmat(path)
    key = 'normal' if 'normal' in raw else max(
        (k for k in raw if not k.startswith('__')),
        key=lambda k: np.asarray(raw[k]).size)
    array = np.asarray(raw[key]).astype(float)
    if array.shape[1] < len(KEEP_COLS):
        raise SystemExit(
            f'{TRAIN_FILE} has {array.shape[1]} columns, fewer than the '
            f'{len(KEEP_COLS)} expected.')
    print(f'  normal data: {array.shape} (object "{key}") from {TRAIN_FILE}')
    return pd.DataFrame(array[:, KEEP_COLS], columns=SENSORS)


def load_faults(data_dir):
    """Load every fault array from the designed test file, keyed by number."""
    from scipy.io import loadmat

    path = os.path.join(data_dir, TEST_FILE)
    if not os.path.exists(path):
        raise SystemExit(f'Not found: {path}')
    raw = loadmat(path)
    faults = {}
    for key, value in raw.items():
        if not key.startswith('fault'):
            continue
        number = int(key.replace('fault', ''))
        array = np.asarray(value).astype(float)
        faults[number] = pd.DataFrame(array[:, KEEP_COLS], columns=SENSORS)
    if not faults:
        raise SystemExit(f'No fault arrays in {TEST_FILE}.')
    print(f'  fault data : {len(faults)} fault array(s) from {TEST_FILE}')
    return faults


# ----------------------------------------------------------------------------------------------------------
# 3. Feature construction
# ----------------------------------------------------------------------------------------------------------

def create_features(sensor_data, window=ROLLING_WINDOW,
                    min_periods=ROLLING_MIN_PERIODS):
    """
    Build the 3-feature representation the autoencoders expect.
    Rows where the rolling statistics are undefined are dropped, so the
    result is shorter than the input; the caller keeps the surviving index
    to realign sample numbers.
    """
    features_df = pd.DataFrame({
        'original': sensor_data,
        'rolling_mean': sensor_data.rolling(window,
                                            min_periods=min_periods).mean(),
        'rolling_std': sensor_data.rolling(window,
                                           min_periods=min_periods).std(),
    })
    return features_df.dropna()


# ----------------------------------------------------------------------------------------------------------
# 4. Verifying the window against the saved scalers
# ----------------------------------------------------------------------------------------------------------

def verify_window(normal_df, scalers):
    """
    Confirm the configured rolling window is the one the models were
    trained with.

    Each scaler stores the mean of the rolling_std feature, and that number
    depends on the window: a 10-sample window and a 20-sample window give
    measurably different standard deviations on the same signal. Rebuilding
    the features at several candidate windows and seeing which reproduces
    the stored value identifies the right one.

    Getting this wrong is silent and total: every sensor would reconstruct
    badly, and the errors would measure the mismatch rather than any fault.
    """
    print('\n  Verifying the rolling window against the saved scalers...')

    stored = {}
    for index, sensor in enumerate(SENSORS):
        mean = getattr(scalers[sensor], 'mean_', None)
        if mean is not None and len(mean) >= 3:
            stored[sensor] = float(mean[2])
    if not stored:
        print('    scalers hold no fitted means; skipping the check')
        return

    results = []
    for window in CANDIDATE_WINDOWS:
        min_periods = max(1, window // 2)
        errors = []
        for sensor, target in stored.items():
            recomputed = create_features(normal_df[sensor], window,
                                         min_periods)['rolling_std'].mean()
            errors.append(abs(recomputed - target) / (abs(target) + 1e-9))
        results.append((float(np.mean(errors)), window, min_periods))

    results.sort()
    best_error, best_window, best_min = results[0]
    for error, window, _min_periods in results:
        marker = '  <- best' if window == best_window else ''
        current = '  (configured)' if window == ROLLING_WINDOW else ''
        print(f'    window {window:>3}: relative error '
              f'{error:.4f}{marker}{current}')

    if best_window == ROLLING_WINDOW and best_error < 0.02:
        print(f'    CONFIRMED: window {ROLLING_WINDOW} is correct '
              f'(error {best_error:.4f})')
    elif best_error < 0.02:
        raise SystemExit(
            f'\n  WINDOW MISMATCH.\n'
            f'  This script is configured for window {ROLLING_WINDOW}, but '
            f'the scalers say {best_window} (min_periods {best_min}).\n'
            f'  Building a cache at the wrong window produces meaningless '
            f'errors.\n'
            f'  Change ROLLING_WINDOW to {best_window} and '
            f'ROLLING_MIN_PERIODS to {best_min}, then re-run.')
    else:
        print(f'    INCONCLUSIVE: best is window {best_window} at error '
              f'{best_error:.4f}, which is a poor match either way.')
        print(f'    Continuing with the configured window '
              f'{ROLLING_WINDOW}, but treat the cache with caution.')


# ----------------------------------------------------------------------------------------------------------
# 5. Loading the saved models and scalers
# ----------------------------------------------------------------------------------------------------------

def load_models(model_dir):
    """
    Load the 5 autoencoders, CPU only and uncompiled.

    TTS_model_5 and TTS_model_6 exist but are duplicate re-trainings of
    models 3 and 4 on the same signals, so they are not loaded.
    """
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
    representation and this builder would produce meaningless errors.
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
    """Load the per-sensor scalers saved alongside the models."""
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
# 6. Reconstruction error
# ----------------------------------------------------------------------------------------------------------

def reconstruction_error(model, scaler, features_df, inverse=True):
    """
    Per-timestep MAE between input and reconstruction.

    inverse=True weights each of the 3 features by its own standard
    deviation, matching thre_tank_frozen.py and the CSTR builder.
    inverse=False measures in scaled space instead. The cache metadata
    records which was used.
    """
    scaled = scaler.transform(features_df.values)
    predicted = model.predict(scaled, verbose=0)
    if inverse:
        return np.mean(np.abs(scaler.inverse_transform(predicted)
                              - scaler.inverse_transform(scaled)), axis=1)
    return np.mean(np.abs(predicted - scaled), axis=1)


def errors_for_block(data, models, scalers, inverse=True):
    """
    Reconstruction errors for one continuous block of samples.

    Every sensor loses the same leading rows to the rolling window, so the
    first sensor's surviving index is reused. That index is the original
    sample number, which matters: the attribution method measures how early
    each sensor crosses its threshold, so a shifted sample number moves the
    apparent onset.
    """
    errors, kept_index = {}, None
    for sensor in SENSORS:
        features_df = create_features(data[sensor])
        if kept_index is None:
            kept_index = features_df.index
        errors[sensor] = reconstruction_error(models[sensor], scalers[sensor],
                                              features_df, inverse)
    return pd.DataFrame(errors, index=kept_index).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------
# 7. Building the two halves of the cache
# ----------------------------------------------------------------------------------------------------------

def build_normal_cache(normal_df, models, scalers, inverse=True):
    """
    Errors on the designed training file - the baseline every z-score uses.

    The training file is one continuous normal run, so the rolling window
    never crosses a boundary and no per-run splitting is needed.
    """
    print('\nBuilding NORMAL error cache...')
    errors = errors_for_block(normal_df, models, scalers, inverse)
    errors['simulationRun'] = 'designed_train'
    print(f'  {len(errors)} rows from {TRAIN_FILE}')
    return errors.reset_index(drop=True)


def build_fault_cache(faults, models, scalers, inverse=True):
    """
    Errors on every fault run, split at the onset sample.

    Features are built over the whole 2001-sample run so the window is
    warm by the time the fault starts, then only the post-onset rows are
    kept. Building features from sample 200 onward would leave the first
    rows of the fault segment with a cold window, exactly where the
    attribution method looks hardest.
    """
    print('\nBuilding FAULT error cache...')
    blocks = []
    for number in sorted(faults):
        errors = errors_for_block(faults[number], models, scalers, inverse)
        fault = errors[errors.index >= FAULT_ONSET].copy()
        if fault.empty:
            print(f'    SKIPPED fault {number}: no post-onset rows')
            continue

        fault['faultNumber'] = number
        fault['simulationRun'] = f'fault{number:02d}'
        fault['sample'] = fault.index.values
        fault['true_cause'] = '/'.join(FAULT_TRUTH.get(number, ['unknown']))
        fault['fault_type'] = FAULT_TYPE.get(number, 'unknown')

        note = f'   DISPUTED: {DISPUTED_FAULTS[number]}' \
            if number in DISPUTED_FAULTS else ''
        print(f'    fault {number:>2} ({FAULT_TYPE.get(number, "?"):<8}) '
              f'{len(fault):>5} rows   origin '
              f'{"/".join(FAULT_TRUTH.get(number, ["?"])):<8}{note}')
        blocks.append(fault.reset_index(drop=True))

    if not blocks:
        raise SystemExit('No fault produced usable rows.')
    return pd.concat(blocks, ignore_index=True)


# ----------------------------------------------------------------------------------------------------------
# 8. Saving the cache
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


def write_metadata(out_dir, model_dir, data_dir, fault_errors, error_metric):
    """Record everything needed to reproduce this cache."""
    metadata = dict(
        dataset='three_tank_DTS200',
        built_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        n_sensors=len(SENSORS),
        sensors=SENSORS,
        keep_cols=KEEP_COLS,
        source='designed files (5 clean columns, 12 distinct faults)',
        representation=f'{FEATURE_NAMES}, window {ROLLING_WINDOW}, '
                       f'min_periods {ROLLING_MIN_PERIODS}',
        scaler='saved scalers, transform-only',
        error_metric=error_metric,
        fault_onset_sample=FAULT_ONSET,
        data_dir=data_dir,
        n_faults=int(fault_errors['faultNumber'].nunique()),
        ground_truth={str(k): v for k, v in FAULT_TRUTH.items()},
        disputed_ground_truth={str(k): v for k, v in DISPUTED_FAULTS.items()},
        model_digests={s: file_digest(model_file(model_dir, i))
                       for i, s in enumerate(SENSORS)
                       if os.path.exists(model_file(model_dir, i))},
    )
    path = os.path.join(out_dir, 'cache_metadata.json')
    with open(path, 'w') as handle:
        json.dump(metadata, handle, indent=2, default=str)
    print(f'  metadata -> {path}')


# ----------------------------------------------------------------------------------------------------------
# 9. Main
# ----------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build the three-tank error cache from saved weights.')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--models', default='TTS_Models')
    parser.add_argument('--out', default='TTS_cache')
    parser.add_argument('--scaled-error', action='store_true',
                        help='compute MAE in scaled space instead of '
                             'inverse-transforming first')
    args = parser.parse_args()
    inverse = not args.scaled_error

    print('=' * 74)
    print('THREE-TANK STAGE 0 - reconstruction-error cache from SAVED weights')
    print('=' * 74)

    check_dependencies()
    os.makedirs(args.out, exist_ok=True)

    normal_df = load_normal(args.data_dir)
    faults = load_faults(args.data_dir)

    models = load_models(args.models)
    assert_three_inputs(models)
    scalers = load_scalers(args.models)
    verify_window(normal_df, scalers)

    print(f'  error metric: '
          f'{"MAE, scaled space" if args.scaled_error else "MAE, inverse-transformed"}')

    normal_errors = build_normal_cache(normal_df, models, scalers, inverse)
    fault_errors = build_fault_cache(faults, models, scalers, inverse)

    print()
    save_cache(args.out, normal_errors, fault_errors)
    write_metadata(args.out, args.models, args.data_dir, fault_errors,
                   'MAE over features, scaled space' if args.scaled_error
                   else 'MAE over features, inverse-transformed')

    print(f'\nDone. Cache written to {args.out}/')
    print(f'  {fault_errors["faultNumber"].nunique()} faults, '
          f'{len(SENSORS)} sensors')
    if DISPUTED_FAULTS:
        print(f'  NOTE: fault {list(DISPUTED_FAULTS)} has disputed ground '
              f'truth - see the top of this file.')


if __name__ == '__main__':
    main()