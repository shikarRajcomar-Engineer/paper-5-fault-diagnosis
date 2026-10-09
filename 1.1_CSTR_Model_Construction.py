"""
CSTR INDIVIDUAL SENSOR AUTOENCODER — MODEL DEVELOPMENT
======================================================
Builds one autoencoder per sensor for the CSTR system, trained on NORMAL
operation only. Each sensor AE sees 3 features of its OWN signal:
    [ original , rolling_mean , rolling_std ]   (window=10)
giving the model temporal context of the fault propagating.

SCALING CONTRACT (this is the important part — read before editing):
  - Development fits ONE StandardScaler per sensor on the NORMAL training
    features, and SAVES it to disk next to the model.
  - Inference must LOAD that saved scaler and call .transform() ONLY —
    never .fit_transform() again. Refitting per file is what caused the
    per-file standardisation drift; saving the scaler here removes it at
    the source.

Saves, per sensor i in [Ci, Ti, T, Qc, Tci, Tc, C]:
    AE_model_feature_Stdscalar{i}.h5     (same names the magnitude script loads)
    AE_scaler_feature_Stdscalar{i}.pkl   (the fitted StandardScaler)

Training data : Model_Data.xlsx  (all rows Class==0; 31 concatenated runs)
Run with the SAME conda env you will run inference in (Paper5).
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.getcwd()
MODEL_DIR   = BASE_DIR                       # save .h5/.pkl here (magnitude
                                             # script loads from cwd)
PLOT_DIR    = os.path.join(BASE_DIR, "AE_dev_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

TRAIN_FILE   = "Model_Data.xlsx"             # all Class==0 normal data
SENSOR_COLS  = ["Ci", "Ti", "T", "Qc", "Tci", "Tc", "C"]
SENSOR_SLICE = slice(2, 9)                   # columns 2..8 = the 7 sensors

# Feature engineering (exactly your original: window 10, min_periods 5)
ROLL_WINDOW = 10
ROLL_MIN    = 5

# Training hyperparameters
VAL_FRAC   = 0.15
EPOCHS     = 500
BATCH_SIZE = 64
PATIENCE   = 30
SEED       = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — FEATURE ENGINEERING  (your function, applied PER RUN)
# ──────────────────────────────────────────────────────────────────────────────

def create_features(sensor_data, window=ROLL_WINDOW, min_periods=ROLL_MIN):
    """
    Three features per sensor sample:
      1. original value
      2. rolling mean (window=10)
      3. rolling std  (window=10)
    Gives the AE temporal context of a propagating fault.
    """
    rolling_mean = sensor_data.rolling(window=window, min_periods=min_periods).mean()
    rolling_std  = sensor_data.rolling(window=window, min_periods=min_periods).std()
    features_df = pd.DataFrame({
        "original":     sensor_data.values,
        "rolling_mean": rolling_mean.values,
        "rolling_std":  rolling_std.values,
    })
    features_df.dropna(inplace=True)
    return features_df


def build_sensor_features(df_all, sensor):
    """
    Build the 3-feature matrix for one sensor across ALL normal runs, computing
    the rolling features PER RUN so the window never straddles a run boundary.
    Runs are delimited by Time resetting to 0.
    """
    run_starts = df_all.index[df_all["Time"] == 0].tolist()
    run_starts.append(len(df_all))
    pieces = []
    for a, b in zip(run_starts[:-1], run_starts[1:]):
        run_series = df_all[sensor].iloc[a:b].reset_index(drop=True)
        pieces.append(create_features(run_series))
    return pd.concat(pieces, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — AUTOENCODER ARCHITECTURE
# Small symmetric AE for a 3-feature input: 3 -> 8 -> 2 -> 8 -> 3
# ──────────────────────────────────────────────────────────────────────────────

def build_autoencoder(n_features):
    """Compact symmetric autoencoder for the 3-feature per-sensor input."""
    inp = Input(shape=(n_features,))
    x   = Dense(16, activation="relu")(inp)      # was 8
    z   = Dense(3,  activation="relu")(x)        # bottleneck, was 2
    x   = Dense(16, activation="relu")(z)        # was 8
    out = Dense(n_features, activation="linear")(x)
    ae  = Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — LOAD NORMAL DATA
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("CSTR INDIVIDUAL SENSOR AUTOENCODER — DEVELOPMENT")
print("=" * 68)
print(f"Loading {TRAIN_FILE} ...")

df_all = pd.read_excel(TRAIN_FILE, engine="openpyxl")
# safety: this pipeline expects normal-only training data
n_fault = int((df_all["Class"] == 1).sum())
if n_fault:
    print(f"  WARNING: {n_fault} Class==1 rows present — keeping Class==0 only")
    df_all = df_all[df_all["Class"] == 0].reset_index(drop=True)

n_runs = int((df_all["Time"] == 0).sum())
print(f"  Rows: {len(df_all)}  |  concatenated normal runs: {n_runs}")
print(f"  Sensors: {SENSOR_COLS}\n")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — TRAIN ONE AE PER SENSOR
# ──────────────────────────────────────────────────────────────────────────────

summary = []
for i, sensor in enumerate(SENSOR_COLS):
    print(f"── Sensor {i}: {sensor} " + "─" * 40)

    # (a) features across all runs (rolling computed per run)
    feats = build_sensor_features(df_all, sensor)
    print(f"   feature matrix: {feats.shape}")

    # (b) FIT the scaler on NORMAL features, SAVE it — the scaling contract
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats.values)
    scaler_path = os.path.join(MODEL_DIR, f"AE_scaler_feature_Stdscalar{i}.pkl")
    joblib.dump(scaler, scaler_path)

    # (c) train / validation split
    n_val = int(len(scaled) * VAL_FRAC)
    idx   = np.random.permutation(len(scaled))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, X_val = scaled[tr_idx], scaled[val_idx]

    # (d) build + train the AE
    ae = build_autoencoder(scaled.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                       restore_best_weights=True)
    hist = ae.fit(X_tr, X_tr,
                  validation_data=(X_val, X_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=[es], verbose=0)

    # (e) save the model with the naming convention the magnitude script loads
    model_path = os.path.join(MODEL_DIR, f"AE_model_feature_Stdscalar{i}.h5")
    ae.save(model_path)

    # (f) sanity: reconstruction error on the (normal) training features
    #     version-safe MAE with numpy (Keras renamed the loss fn across versions)
    recon = ae.predict(scaled, verbose=0)
    mae   = np.mean(np.abs(recon - scaled), axis=1)
    print(f"   trained {len(hist.history['loss'])} epochs  |  "
          f"val_loss={hist.history['val_loss'][-1]:.4f}")
    print(f"   normal MAE: mean={mae.mean():.4f}  95th pct={np.percentile(mae,95):.4f}")
    print(f"   saved: {os.path.basename(model_path)} + "
          f"{os.path.basename(scaler_path)}\n")

    # (g) sanity plot: loss curve + normal-error histogram
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#111827")
    ax[0].plot(hist.history["loss"], color="#4488ff", label="train")
    ax[0].plot(hist.history["val_loss"], color="#00ff88", label="val")
    ax[0].set_title(f"{sensor} — training loss", color="white")
    ax[1].hist(mae, bins=50, color="#5577cc", edgecolor="none")
    ax[1].set_title(f"{sensor} — normal reconstruction MAE", color="white")
    for a in ax:
        a.set_facecolor("#0f1923"); a.tick_params(colors="white")
        for sp in a.spines.values(): sp.set_color("#334477")
    ax[0].legend(facecolor="#1a2035", labelcolor="white", edgecolor="none")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"sensor{i}_{sensor}_dev.png"),
                dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    summary.append({"sensor": sensor, "features": feats.shape[0],
                    "epochs": len(hist.history["loss"]),
                    "normal_mae_mean": round(float(mae.mean()), 5),
                    "normal_mae_p95": round(float(np.percentile(mae, 95)), 5)})

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 68)
print("DEVELOPMENT COMPLETE — per-sensor summary")
print("=" * 68)
print(pd.DataFrame(summary).to_string(index=False))
print(f"\nModels + scalers saved in: {MODEL_DIR}")
print(f"Sanity plots in: {PLOT_DIR}")
print("\nREMINDER FOR INFERENCE: load AE_scaler_feature_Stdscalar{i}.pkl and")
print("call .transform() — do NOT fit a new scaler per file.")