#!/usr/bin/env python3
"""
ICMPRS v2 — Hardware Benchmark
================================
Measures inference latency and memory footprint on the current hardware.
Target: ARM Cortex-A72 @ 1.8 GHz (Raspberry Pi 4).

Author : Nobhonil Roy Choudhury
Licence: MIT
"""

import os
import time
import tracemalloc
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

VOICE_FEATURES = [
    "jitter_local", "shimmer_local", "hnr_db", "pitch_mean_hz",
    "rpde", "dfa", "spread1", "spread2",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
    "jitter_rap", "shimmer_apq11", "nhr", "pitch_std_hz",
    "voiced_fraction", "speech_rate", "pause_ratio",
    "rca_index",
    "nre_nasal_energy_ratio", "nre_velopharyngeal_leakage",
    "nre_nasal_formant_bw_hz",
]

HAND_FEATURES = [
    "pen_velocity_cm_s", "drawing_time_s", "pen_pressure_norm",
    "velocity_direction_changes", "tremor_frequency_hz",
    "task_completion_time_s", "stroke_length_cm", "stroke_width_mm",
    "curvature_mean", "curvature_std", "writing_area_cm2",
    "horizontal_drift_mm", "vertical_drift_mm",
    "acceleration_mean_cm_s2", "deceleration_peaks",
    "dpld_lift_count", "dpld_lift_duration_ms", "dpld_restart_velocity_cm_s",
    "ajhi_conjunct_transition_ms", "ajhi_conjunct_pressure_spike",
    "ajhi_inter_akshar_pause_ratio",
]

GAIT_FEATURES = [
    "stride_length_m", "walking_speed_m_s", "cadence_steps_min",
    "cadence_variability", "step_width_cm", "lateral_trunk_sway_deg",
    "freeze_index", "stride_time_s", "swing_phase_pct",
    "stance_phase_pct", "double_support_pct", "stride_asymmetry",
    "step_time_variability_ms", "toe_clearance_cm", "arm_swing_asymmetry",
    "bsv_normalised",
]

MOVEMENT_FEATURES = HAND_FEATURES + GAIT_FEATURES


def main():
    print("Loading and training models for benchmark ...")

    voice = pd.read_csv(os.path.join(DATA_DIR, "voice_final.csv"))
    hand = pd.read_csv(os.path.join(DATA_DIR, "hand_final.csv"))
    gait = pd.read_csv(os.path.join(DATA_DIR, "gait_final.csv"))

    labels = voice["label"].values
    X_voice = voice[VOICE_FEATURES].values
    X_move = np.hstack([hand[HAND_FEATURES].values,
                        gait[GAIT_FEATURES].values])

    scaler_v = StandardScaler().fit(X_voice)
    scaler_m = StandardScaler().fit(X_move)
    X_vs = scaler_v.transform(X_voice)
    X_ms = scaler_m.transform(X_move)

    svm = SVC(C=10, gamma=0.01, kernel="rbf", probability=True,
              random_state=SEED)
    svm.fit(X_vs, labels)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                max_features=6, random_state=SEED)
    rf.fit(X_ms, labels)

    # --- Single-sample inference benchmark ---
    x_v = X_vs[0:1]
    x_m = X_ms[0:1]

    # Warmup
    for _ in range(100):
        svm.predict_proba(x_v)
        rf.predict_proba(x_m)

    N_ITER = 1000
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()

        # SVM prediction
        p_svm = svm.predict_proba(x_v)[0, 1]
        # RF prediction
        p_rf = rf.predict_proba(x_m)[0, 1]

        # ACG (12 FLOPs)
        sqi_v = 0.5  # placeholder
        sqi_m = 0.5
        w1 = sqi_v / (sqi_v + sqi_m)
        w2 = 1 - w1
        p_final = w1 * p_svm + w2 * p_rf

        # CMCC (2 FLOPs)
        cmcs = 1 - abs(p_svm - p_rf)

        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)

    # --- Memory benchmark ---
    tracemalloc.start()
    svm.predict_proba(x_v)
    rf.predict_proba(x_m)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Model sizes
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        svm_path = f.name
    joblib.dump(svm, svm_path)
    svm_size = os.path.getsize(svm_path) / (1024 * 1024)
    os.unlink(svm_path)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        rf_path = f.name
    joblib.dump(rf, rf_path)
    rf_size = os.path.getsize(rf_path) / (1024 * 1024)
    os.unlink(rf_path)

    total_size = svm_size + rf_size

    print("\n" + "=" * 60)
    print("  HARDWARE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n  Inference Latency (N={N_ITER} iterations):")
    print(f"    Mean:   {times.mean():.2f} ms")
    print(f"    Std:    {times.std():.2f} ms")
    print(f"    Median: {np.median(times):.2f} ms")
    print(f"    P95:    {np.percentile(times, 95):.2f} ms")
    print(f"    P99:    {np.percentile(times, 99):.2f} ms")
    print(f"\n  Memory Footprint:")
    print(f"    SVM model: {svm_size:.2f} MB")
    print(f"    RF model:  {rf_size:.2f} MB")
    print(f"    Total:     {total_size:.2f} MB")
    print(f"    Peak runtime: {peak / (1024*1024):.2f} MB")
    print(f"\n  Feature dimensions:")
    print(f"    Voice:    {len(VOICE_FEATURES)}")
    print(f"    Movement: {len(MOVEMENT_FEATURES)}")
    print(f"    Total:    {len(VOICE_FEATURES) + len(MOVEMENT_FEATURES)}")
    print(f"\n  Support vectors: {svm.n_support_.sum()}")
    print(f"  RF trees: {rf.n_estimators}, max depth: {rf.max_depth}")
    print("=" * 60)


if __name__ == "__main__":
    main()
