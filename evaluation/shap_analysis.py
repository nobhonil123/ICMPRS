#!/usr/bin/env python3
"""
ICMPRS v2 — SHAP Feature Importance Analysis
=============================================
TreeSHAP for Random Forest, KernelSHAP (1,000 background) for SVM.
Generates top-15 importance table and permutation significance test.

Author : Nobhonil Roy Choudhury
Licence: MIT
"""

import os
import warnings
import numpy as np
import pandas as pd
import shap
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature lists (must match train_ensemble.py)
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

# Novel features to highlight
NOVEL_FEATURES = {
    "rca_index", "nre_nasal_energy_ratio", "nre_velopharyngeal_leakage",
    "nre_nasal_formant_bw_hz", "dpld_lift_count", "dpld_lift_duration_ms",
    "dpld_restart_velocity_cm_s", "ajhi_conjunct_transition_ms",
    "ajhi_conjunct_pressure_spike", "ajhi_inter_akshar_pause_ratio",
    "bsv_normalised",
}


def modality_of(feat_name):
    if feat_name in VOICE_FEATURES:
        return "Voice"
    elif feat_name in HAND_FEATURES:
        return "Writing"
    elif feat_name in GAIT_FEATURES:
        return "Gait"
    return "Unknown"


def main():
    print("Loading data ...")
    voice = pd.read_csv(os.path.join(DATA_DIR, "voice_final.csv"))
    hand = pd.read_csv(os.path.join(DATA_DIR, "hand_final.csv"))
    gait = pd.read_csv(os.path.join(DATA_DIR, "gait_final.csv"))

    labels = voice["label"].values
    X_voice = voice[VOICE_FEATURES].values
    X_hand = hand[HAND_FEATURES].values
    X_gait = gait[GAIT_FEATURES].values
    X_move = np.hstack([X_hand, X_gait])

    all_features = VOICE_FEATURES + MOVEMENT_FEATURES

    # Train on full data for SHAP analysis
    scaler_v = StandardScaler()
    X_vs = scaler_v.fit_transform(X_voice)
    scaler_m = StandardScaler()
    X_ms = scaler_m.fit_transform(X_move)

    print("Training SVM ...")
    svm = SVC(C=10, gamma=0.01, kernel="rbf", random_state=SEED,
              probability=True)
    svm.fit(X_vs, labels)

    print("Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                max_features=6, random_state=SEED)
    rf.fit(X_ms, labels)

    # --- TreeSHAP for RF ---
    print("Computing TreeSHAP for RF ...")
    rf_explainer = shap.TreeExplainer(rf)
    rf_shap = rf_explainer.shap_values(X_ms)
    if isinstance(rf_shap, list):
        rf_shap = rf_shap[1]  # PD class
    rf_importance = np.abs(rf_shap).mean(axis=0)

    # --- KernelSHAP for SVM ---
    print("Computing KernelSHAP for SVM (1,000 background samples) ...")
    bg_idx = rng.choice(len(X_vs), 1000, replace=False)
    svm_explainer = shap.KernelExplainer(svm.predict_proba, X_vs[bg_idx])
    # Use a subset for speed
    sample_idx = rng.choice(len(X_vs), 500, replace=False)
    svm_shap = svm_explainer.shap_values(X_vs[sample_idx])
    if isinstance(svm_shap, list):
        svm_shap = svm_shap[1]
    svm_importance = np.abs(svm_shap).mean(axis=0)

    # --- Combine into unified ranking ---
    combined = {}
    for i, feat in enumerate(VOICE_FEATURES):
        combined[feat] = svm_importance[i]
    for i, feat in enumerate(MOVEMENT_FEATURES):
        combined[feat] = rf_importance[i]

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("  TOP-15 FEATURES BY MEAN |SHAP|")
    print("=" * 60)
    print(f"  {'Rank':<5} {'Feature':<40} {'|SHAP|':>8} {'Mod.':<8} {'Novel'}")
    print("  " + "-" * 70)

    total_shap = sum(v for _, v in ranked)
    novel_shap = 0

    for rank, (feat, imp) in enumerate(ranked[:15], 1):
        mod = modality_of(feat)
        is_novel = "★" if feat in NOVEL_FEATURES else ""
        print(f"  {rank:<5} {feat:<40} {imp:>8.3f} {mod:<8} {is_novel}")
        if feat in NOVEL_FEATURES:
            novel_shap += imp

    # Total novel feature contribution
    total_novel = sum(combined[f] for f in NOVEL_FEATURES if f in combined)
    novel_pct = 100 * total_novel / total_shap if total_shap > 0 else 0

    print(f"\n  Novel features cumulative contribution: {novel_pct:.1f}%")

    # --- Permutation significance test ---
    print("\n  Permutation test (1,000 draws of 5 random features) ...")
    all_vals = list(combined.values())
    n_novel = len(NOVEL_FEATURES)
    perm_sums = []
    for _ in range(1000):
        idx = rng.choice(len(all_vals), n_novel, replace=False)
        perm_sums.append(sum(all_vals[i] for i in idx))

    perm_mean = np.mean(perm_sums)
    perm_std = np.std(perm_sums)
    perm_pct = np.mean([s >= total_novel for s in perm_sums])

    print(f"    Null mean: {100*perm_mean/total_shap:.1f}% ± "
          f"{100*perm_std/total_shap:.1f}%")
    print(f"    Observed novel: {novel_pct:.1f}%")
    print(f"    p-value: {perm_pct:.4f}")

    # --- Save ---
    shap_df = pd.DataFrame(ranked, columns=["feature", "mean_abs_shap"])
    shap_df["rank"] = range(1, len(shap_df) + 1)
    shap_df["modality"] = shap_df["feature"].apply(modality_of)
    shap_df["novel"] = shap_df["feature"].apply(
        lambda f: True if f in NOVEL_FEATURES else False
    )
    shap_path = os.path.join(RESULTS_DIR, "shap_importance.csv")
    shap_df.to_csv(shap_path, index=False)
    print(f"\n  Saved: {shap_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
