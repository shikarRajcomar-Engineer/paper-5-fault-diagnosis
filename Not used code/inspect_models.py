"""
inspect_models.py — WHAT ARE MY SAVED MODELS ACTUALLY EXPECTING?
================================================================
Read-only. Loads nothing but the weight files and scalers, prints what
they expect, and changes nothing on disk.

Run it from inside "Paper 5 Combined":

    python inspect_models.py

TWO QUESTIONS IT ANSWERS
------------------------
1. Do the autoencoders take 3 engineered features (original, rolling mean,
   rolling std) or 1 raw column? The two candidate builders in the project
   history disagree, and the answer decides whether the existing TEP cache
   is valid.

2. Which three-tank model belongs to which sensor? There are 7 saved
   models but only 5 sensors after the duplicate columns are dropped. Each
   scaler carries the mean of the data it was fitted on, so comparing those
   means against the actual column means in the .mat file identifies the
   mapping directly.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

MODEL_DIRS = {
    "TEP": ("TEP_Models", [f"autoencoder_xmeas_{i}.h5" for i in (1, 2, 4)]
            + ["autoencoder_xmv_1.h5"]),
    "CSTR": ("CSTR_Models", [f"AE_model_feature_Stdscalar{i}.h5" for i in range(7)]),
    "TTS": ("TTS_Models", [f"TTS_model_{i}.h5" for i in range(7)]),
}

TTS_MAT = "[test].mat"
TTS_SENSOR_GUESS = ["h1", "h2", "h3", "Q1", "Q2"]
TTS_KEEP_COLS = [0, 1, 4, 5, 6]
FAULT_ONSET = 200        # samples before this index are normal operation


def find_mat(explicit: Optional[str]) -> Optional[str]:
    """
    Locate the DTS200 .mat file.

    Checks an explicit path first, then the working directory, then searches
    two levels down. The file often sits beside the data rather than beside
    the models, so a plain filename lookup misses it.
    """
    if explicit and os.path.exists(explicit):
        return explicit
    if os.path.exists(TTS_MAT):
        return TTS_MAT
    for root, _dirs, files in os.walk(".", topdown=True):
        if root.count(os.sep) > 2:
            continue
        for name in files:
            if name.endswith(".mat"):
                return os.path.join(root, name)
    return None


# ──────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────

def make_loader():
    """
    Return a function that loads an .h5 model without compiling it, or None
    if TensorFlow is unavailable.

    Returning None rather than raising matters: Part 2 needs only joblib and
    scipy, so a broken TensorFlow install must not block the three-tank
    mapping check, which is the more urgent of the two questions.
    """
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"  TensorFlow unavailable ({type(exc).__name__}) — "
              f"skipping Part 1, Part 2 will still run.")
        return None
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    try:
        import tf_keras
        print("  (tf-keras legacy loader available)")
        return lambda p: tf_keras.models.load_model(p, compile=False)
    except ImportError:
        return lambda p: tf.keras.models.load_model(p, compile=False)


def describe_model(loader, path: str) -> Optional[str]:
    """One line describing a model's input and output shape, or the error."""
    try:
        model = loader(path)
    except Exception as exc:
        return f"FAILED TO LOAD: {type(exc).__name__}: {exc}"
    in_shape = model.input_shape
    out_shape = model.output_shape
    n_in = in_shape[-1] if isinstance(in_shape, tuple) else in_shape[0][-1]
    n_layers = len(model.layers)
    return f"input={n_in}  output={out_shape[-1]}  layers={n_layers}"


def report_models() -> None:
    """Print the input dimension of a sample of models from every plant."""
    print("=" * 70)
    print("PART 1 — MODEL INPUT DIMENSIONS")
    print("=" * 70)
    print("  3 = trained on (original, rolling_mean, rolling_std)")
    print("  1 = trained on the raw scaled column\n")
    loader = make_loader()
    if loader is None:
        return
    for plant, (folder, filenames) in MODEL_DIRS.items():
        print(f"\n{plant}  ({folder})")
        if not os.path.isdir(folder):
            print(f"  folder not found — skipped")
            continue
        for name in filenames:
            path = os.path.join(folder, name)
            if not os.path.exists(path):
                print(f"  {name:<38} missing")
                continue
            print(f"  {name:<38} {describe_model(loader, path)}")


# ──────────────────────────────────────────────────────────────────────────
# SCALER FINGERPRINTS
# ──────────────────────────────────────────────────────────────────────────

class _StubObject:
    """
    Stand-in for a class that cannot be imported.

    Pickle rebuilds an object by creating it and handing it its state. We do
    not need scikit-learn's behaviour, only the numbers it stored, so a stub
    that simply absorbs the state is enough.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _install_stub_module(name: str) -> None:
    """
    Register a fake module so joblib can resolve a class it cannot import.

    Any attribute looked up on the stub returns a throwaway class that
    absorbs whatever state pickle hands it. Parent packages are registered
    too, since 'sklearn.preprocessing._data' needs 'sklearn' to exist first.
    """
    import sys
    import types

    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        partial = ".".join(parts[:depth])
        if partial in sys.modules:
            continue
        module = types.ModuleType(partial)
        module.__getattr__ = lambda attr: type(attr, (_StubObject,), {})
        module.__path__ = []
        sys.modules[partial] = module


def load_scaler_state(path: str, max_stubs: int = 12) -> dict:
    """
    Read a saved scaler's fitted attributes WITHOUT requiring scikit-learn.

    joblib needs the original classes importable, which fails in a bare
    environment. We only want the stored numbers, so each time the load
    reports a missing module we register a stub for it and retry. numpy is
    left alone, because the arrays themselves are genuinely numpy objects.
    """
    import joblib

    for _attempt in range(max_stubs):
        try:
            obj = joblib.load(path)
            return getattr(obj, "__dict__", {})
        except ModuleNotFoundError as exc:
            missing = exc.name
            if missing is None or missing.startswith("numpy"):
                raise
            _install_stub_module(missing)
    raise ImportError(f"Gave up stubbing imports for {path}")


def scaler_fingerprint(path: str) -> Optional[np.ndarray]:
    """
    The mean each scaler was fitted on — its fingerprint.

    The first element is the mean of the raw sensor value, because
    'original' is the first engineered feature. That is what lets us match
    a scaler back to the column it came from.
    """
    import joblib
    try:
        scaler = joblib.load(path)
        mean = getattr(scaler, "mean_", None)
    except Exception:
        # scikit-learn missing or version-mismatched: read the state directly.
        try:
            state = load_scaler_state(path)
        except Exception as exc:
            print(f"    could not read {path}: {type(exc).__name__}: {exc}")
            return None
        mean = state.get("mean_", state.get("data_min_"))
        if mean is None:
            print(f"    {path}: no fitted mean found. "
                  f"Stored keys: {sorted(state)[:8]}")
            return None
    return np.asarray(mean) if mean is not None else None


def report_tts_mapping(mat_path: Optional[str] = None) -> None:
    """
    Match each three-tank scaler to the raw column it was fitted on.

    There are 7 saved models for 5 sensors. If each model corresponds to one
    raw column, then the correct sensor mapping is the one KEEP_COLS implies
    (h3 -> model_4, Q1 -> model_5, Q2 -> model_6), NOT the sequential 0-4
    the current script uses.
    """
    print("\n" + "=" * 70)
    print("PART 2 — THREE-TANK: WHICH MODEL BELONGS TO WHICH SENSOR?")
    print("=" * 70)

    folder = "TTS_Models"
    if not os.path.isdir(folder):
        print("  TTS_Models not found — skipped")
        return

    print("\n  Scaler fitted means (first value = mean of the raw column):")
    fingerprints = {}
    for i in range(7):
        path = os.path.join(folder, f"TTS_scaler_{i}.pkl")
        if not os.path.exists(path):
            print(f"    scaler_{i}: missing")
            continue
        mean = scaler_fingerprint(path)
        if mean is None or len(mean) == 0:
            continue
        fingerprints[i] = mean
        print(f"    scaler_{i}: raw mean = {mean[0]:>12.5f}   "
              f"full = {np.round(mean, 4).tolist()}")

    located = find_mat(mat_path)
    if located is None:
        print(f"\n  No .mat file found. Re-run with the path, e.g.:")
        print(f"    python inspect_models.py --mat '/path/to/[test].mat'")
        print("  Compare the raw means above against your data by hand.")
        return
    print(f"\n  Using data file: {located}")

    from scipy.io import loadmat
    mat = loadmat(located)
    fault_keys = sorted(k for k in mat if k.startswith("fault"))
    if not fault_keys:
        print(f"\n  No fault array found in {located}. "
              f"Keys: {[k for k in mat if not k.startswith('__')][:10]}")
        return

    # The scalers were fitted on NORMAL operation, so comparing them against
    # a fault segment builds the fault's own offset into every distance. The
    # pre-onset samples of each file are normal, so pooling those across all
    # faults gives the reference the scalers were actually fitted against.
    baselines = [np.asarray(mat[k])[:FAULT_ONSET] for k in fault_keys]
    baseline = np.vstack(baselines)
    array = np.asarray(mat[fault_keys[0]])
    column_means = baseline.mean(axis=0)
    print(f"\n  Pre-fault column means (samples 0-{FAULT_ONSET - 1}, pooled "
          f"over {len(fault_keys)} faults, {baseline.shape[0]} rows):")
    for c, m in enumerate(column_means):
        note = ""
        if c in TTS_KEEP_COLS:
            note = f"  <- kept as {TTS_SENSOR_GUESS[TTS_KEEP_COLS.index(c)]}"
        print(f"    column {c}: {m:>12.5f}{note}")

    print("\n  BEST MATCH for each scaler (closest pre-fault column mean):")
    for i, mean in fingerprints.items():
        distances = np.abs(column_means - mean[0])
        best = int(np.argmin(distances))
        scale = abs(column_means[best]) + 1e-9
        flag = "" if distances[best] < 0.05 * scale else "   (weak — check)"
        print(f"    model_{i} -> column {best}  "
              f"(trained {mean[0]:.4f} vs column {column_means[best]:.4f})"
              f"{flag}")

    print("\n  DISTINCT SIGNALS AMONG THE SAVED MODELS:")
    groups: dict = {}
    for i, mean in fingerprints.items():
        key = round(float(mean[0]), 3)
        groups.setdefault(key, []).append(i)
    for key, members in sorted(groups.items()):
        note = "  <- DUPLICATE MODELS" if len(members) > 1 else ""
        print(f"    mean {key:>10}: model(s) {members}{note}")
    print(f"    {len(groups)} distinct signal(s) across "
          f"{len(fingerprints)} model files")

    print("\n  COLUMNS WITH NO MATCHING MODEL (nothing was trained on them):")
    orphans = []
    for c, cm in enumerate(column_means):
        nearest = min(abs(m[0] - cm) for m in fingerprints.values())
        if nearest > 0.05 * (abs(cm) + 1e-9):
            orphans.append(c)
            kept = " *** CURRENTLY IN KEEP_COLS ***" if c in TTS_KEEP_COLS else ""
            print(f"    column {c} (mean {cm:.4f}), nearest model is "
                  f"{nearest:.4f} away{kept}")
    if not orphans:
        print("    none — every column has a model")

    print("\n  DUPLICATE CHECK (columns with near-identical means):")
    for a in range(len(column_means)):
        for b in range(a + 1, len(column_means)):
            gap = abs(column_means[a] - column_means[b])
            if gap < 0.01 * (abs(column_means[a]) + 1e-9):
                print(f"    column {a} ~ column {b}  "
                      f"(means {column_means[a]:.4f} vs {column_means[b]:.4f})")

    print("\n  READ THIS AGAINST THE CURRENT SCRIPT, which assumes:")
    for i, sensor in enumerate(TTS_SENSOR_GUESS):
        print(f"    {sensor} -> model_{i}")
    print("\n  If the best matches above instead say model_4/5/6 carry "
          "columns 4/5/6,\n  then h3, Q1 and Q2 are currently being "
          "reconstructed by the WRONG models.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Report what the saved models and scalers expect.")
    parser.add_argument("--mat", default=None,
                        help="path to the DTS200 .mat file")
    args = parser.parse_args()

    report_models()
    report_tts_mapping(args.mat)
    print("\nNothing was modified. Paste this output back.")