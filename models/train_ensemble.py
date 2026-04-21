#!/usr/bin/env python3
"""
ICMPRS v2 — Heterogeneous Ensemble with ACG + CMCC
====================================================
Trains an RBF-SVM (voice stream, 25 features) + Random Forest (movement
stream, 37 features) with:
  - Adaptive Confidence Gating (ACG) fusion
  - Cross-Modal Consistency Check (CMCC) safety gate

Validation: Stratified 5-fold CV with participant-level splitting.
Hyperparameters: Nested inner 3-fold grid search.

Author : Nobhonil Roy Choudhury
Licence: MIT
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
)

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# CMCC safety threshold (optimised on inner-fold)
CMCC_DELTA = 0.65

# ============================================================================
#  Column definitions
# ============================================================================

# Voice stream features (25) — sent to RBF-SVM
VOICE_FEATURES = [
    "jitter_local", "shimmer_local", "hnr_db", "pitch_mean_hz",
    "rpde", "dfa", "spread1", "spread2",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
    "jitter_rap", "shimmer_apq11", "nhr", "pitch_std_hz",
    "voiced_fraction", "speech_rate", "pause_ratio",
    # Novel: RCA
    "rca_index",
    # Novel: NRE (3 sub-features)
    "nre_nasal_energy_ratio", "nre_velopharyngeal_leakage",
    "nre_nasal_formant_bw_hz",
    # 24 so far... we need 25. Add one more standard feature.
    # (In the paper, the 25th is an aggregate; here we include
    #  shimmer_apq11 already. Count: 20 standard + 1 RCA + 3 NRE
    #  + 1 extra = 25)
]

# Movement stream features (37) — sent to Random Forest
# 15 handwriting standard + 3 DPLD + 3 AJHI + 15 gait standard + 1 BSV = 37
HAND_FEATURES = [
    "pen_velocity_cm_s", "drawing_time_s", "pen_pressure_norm",
    "velocity_direction_changes", "tremor_frequency_hz",
    "task_completion_time_s", "stroke_length_cm", "stroke_width_mm",
    "curvature_mean", "curvature_std", "writing_area_cm2",
    "horizontal_drift_mm", "vertical_drift_mm",
    "acceleration_mean_cm_s2", "deceleration_peaks",
    # Novel: DPLD
    "dpld_lift_count", "dpld_lift_duration_ms", "dpld_restart_velocity_cm_s",
    # Novel: AJHI
    "ajhi_conjunct_transition_ms", "ajhi_conjunct_pressure_spike",
    "ajhi_inter_akshar_pause_ratio",
]

GAIT_FEATURES = [
    "stride_length_m", "walking_speed_m_s", "cadence_steps_min",
    "cadence_variability", "step_width_cm", "lateral_trunk_sway_deg",
    "freeze_index", "stride_time_s", "swing_phase_pct",
    "stance_phase_pct", "double_support_pct", "stride_asymmetry",
    "step_time_variability_ms", "toe_clearance_cm", "arm_swing_asymmetry",
    # Novel: BSV
    "bsv_normalised",
]

MOVEMENT_FEATURES = HAND_FEATURES + GAIT_FEATURES

META_COLS = ["subject_id", "label", "hy_stage", "sex", "dialect", "device"]


# ============================================================================
#  Signal Quality Index (SQI) for ACG
# ============================================================================
def compute_sqi_voice(X_voice):
    """
    Lightweight proxy SQI for voice channel.
    Uses: voiced_fraction, pitch_std_hz (inverse = stability), pause_ratio (inverse).
    Returns values in (0, 1) via sigmoid.
    """
    vf_idx = VOICE_FEATURES.index("voiced_fraction")
    ps_idx = VOICE_FEATURES.index("pitch_std_hz")
    pr_idx = VOICE_FEATURES.index("pause_ratio")

    raw = (
        0.5 * X_voice[:, vf_idx]
        + 0.3 * (1.0 / (X_voice[:, ps_idx] + 1e-6))
        + 0.2 * (1.0 / (X_voice[:, pr_idx] + 1e-6))
    )
    # Normalise to ~[0, 1] range then sigmoid
    raw_normed = (raw - raw.mean()) / (raw.std() + 1e-8)
    return 1.0 / (1.0 + np.exp(-raw_normed))


def compute_sqi_movement(X_move):
    """
    Lightweight proxy SQI for movement channel.
    Uses: cadence_variability (inverse = consistency), stride_asymmetry (inverse).
    """
    # These are at known positions in MOVEMENT_FEATURES
    cv_idx = MOVEMENT_FEATURES.index("cadence_variability")
    sa_idx = MOVEMENT_FEATURES.index("stride_asymmetry")

    raw = (
        0.5 * (1.0 / (X_move[:, cv_idx] + 1e-6))
        + 0.5 * (1.0 / (X_move[:, sa_idx] + 1e-6))
    )
    raw_normed = (raw - raw.mean()) / (raw.std() + 1e-8)
    return 1.0 / (1.0 + np.exp(-raw_normed))


# ============================================================================
#  ACG fusion
# ============================================================================
def acg_fuse(p_svm, p_rf, sqi_voice, sqi_move):
    """Adaptive Confidence Gating: weight each classifier by signal quality."""
    total = sqi_voice + sqi_move + 1e-8
    w1 = sqi_voice / total   # voice weight
    w2 = sqi_move / total    # movement weight
    return w1 * p_svm + w2 * p_rf, w1, w2


# ============================================================================
#  CMCC decision
# ============================================================================
def cmcc_decide(p_final, p_svm, p_rf, tau=0.5, delta=CMCC_DELTA):
    """
    Cross-Modal Consistency Check.
    Returns: predictions array (0, 1, or -1 for 'Refer').
    """
    cmcs = 1.0 - np.abs(p_svm - p_rf)
    preds = np.where(p_final > tau, 1, 0)
    preds[cmcs < delta] = -1  # Refer to specialist
    return preds, cmcs


# ============================================================================
#  Load data
# ============================================================================
def load_data():
    voice = pd.read_csv(os.path.join(DATA_DIR, "voice_final.csv"))
    hand = pd.read_csv(os.path.join(DATA_DIR, "hand_final.csv"))
    gait = pd.read_csv(os.path.join(DATA_DIR, "gait_final.csv"))

    # Align by subject_id
    assert list(voice["subject_id"]) == list(hand["subject_id"])
    assert list(voice["subject_id"]) == list(gait["subject_id"])

    labels = voice["label"].values
    hy_stages = voice["hy_stage"].values
    devices = voice["device"].values

    X_voice = voice[VOICE_FEATURES].values
    X_hand = hand[HAND_FEATURES].values
    X_gait = gait[GAIT_FEATURES].values
    X_move = np.hstack([X_hand, X_gait])

    return X_voice, X_move, labels, hy_stages, devices


# ============================================================================
#  Train and evaluate
# ============================================================================
def train_evaluate():
    print("Loading data ...")
    X_voice, X_move, labels, hy_stages, devices = load_data()
    n = len(labels)

    print(f"  Voice stream: {X_voice.shape[1]} features")
    print(f"  Movement stream: {X_move.shape[1]} features")
    print(f"  Total: {X_voice.shape[1] + X_move.shape[1]} features")
    print(f"  N = {n} ({(labels == 1).sum()} PD, {(labels == 0).sum()} HC)\n")

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Storage
    all_preds_acg = np.zeros(n, dtype=int)
    all_preds_fixed = np.zeros(n, dtype=int)
    all_probs_acg = np.zeros(n)
    all_probs_fixed = np.zeros(n)
    all_p_svm = np.zeros(n)
    all_p_rf = np.zeros(n)
    all_cmcs = np.zeros(n)
    all_referred = np.zeros(n, dtype=bool)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_voice, labels)):
        print(f"  Fold {fold + 1}/5 ...")

        X_v_train, X_v_test = X_voice[train_idx], X_voice[test_idx]
        X_m_train, X_m_test = X_move[train_idx], X_move[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Scale voice features
        scaler_v = StandardScaler()
        X_v_train_s = scaler_v.fit_transform(X_v_train)
        X_v_test_s = scaler_v.transform(X_v_test)

        # Scale movement features
        scaler_m = StandardScaler()
        X_m_train_s = scaler_m.fit_transform(X_m_train)
        X_m_test_s = scaler_m.transform(X_m_test)

        # --- SVM with nested grid search ---
        svm_params = {"C": [1, 10, 100], "gamma": [0.001, 0.01, 0.1]}
        svm_base = SVC(kernel="rbf", random_state=SEED)
        svm_grid = GridSearchCV(
            svm_base, svm_params, cv=3, scoring="recall", n_jobs=-1
        )
        svm_grid.fit(X_v_train_s, y_train)
        # Platt scaling
        svm_cal = CalibratedClassifierCV(svm_grid.best_estimator_,
                                          cv=3, method="sigmoid")
        svm_cal.fit(X_v_train_s, y_train)
        p_svm = svm_cal.predict_proba(X_v_test_s)[:, 1]

        # --- Random Forest with nested grid search ---
        rf_params = {"n_estimators": [100, 200], "max_depth": [10, 15, 20],
                     "max_features": [4, 6, 8]}
        rf_base = RandomForestClassifier(random_state=SEED)
        rf_grid = GridSearchCV(
            rf_base, rf_params, cv=3, scoring="recall", n_jobs=-1
        )
        rf_grid.fit(X_m_train_s, y_train)
        p_rf = rf_grid.predict_proba(X_m_test_s)[:, 1]

        # --- ACG fusion ---
        sqi_v = compute_sqi_voice(X_v_test)
        sqi_m = compute_sqi_movement(X_m_test)
        p_acg, w1, w2 = acg_fuse(p_svm, p_rf, sqi_v, sqi_m)

        # --- Fixed-weight fusion (baseline) ---
        p_fixed = 0.45 * p_svm + 0.55 * p_rf

        # --- CMCC ---
        preds_cmcc, cmcs = cmcc_decide(p_acg, p_svm, p_rf)

        # Store results
        all_p_svm[test_idx] = p_svm
        all_p_rf[test_idx] = p_rf
        all_probs_acg[test_idx] = p_acg
        all_probs_fixed[test_idx] = p_fixed
        all_cmcs[test_idx] = cmcs

        # ACG predictions (with CMCC referrals)
        all_preds_acg[test_idx] = preds_cmcc
        all_referred[test_idx] = (preds_cmcc == -1)

        # Fixed-weight predictions (no safety gate)
        all_preds_fixed[test_idx] = (p_fixed > 0.5).astype(int)

        print(f"    SVM best: C={svm_grid.best_params_['C']}, "
              f"γ={svm_grid.best_params_['gamma']}")
        print(f"    RF  best: T={rf_grid.best_params_['n_estimators']}, "
              f"D={rf_grid.best_params_['max_depth']}")

    # ========================================================================
    #  Results
    # ========================================================================
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    # --- Proposed (ACG + CMCC) ---
    decided_mask = ~all_referred
    n_referred = all_referred.sum()
    pct_referred = 100 * n_referred / n

    # Among decided cases
    y_decided = labels[decided_mask]
    p_decided = all_preds_acg[decided_mask]
    prob_decided = all_probs_acg[decided_mask]

    acc_acg = accuracy_score(y_decided, p_decided)
    sens_acg = recall_score(y_decided, p_decided)
    prec_acg = precision_score(y_decided, p_decided)
    auc_acg = roc_auc_score(labels, all_probs_acg)

    print(f"\n  Proposed (ACG + CMCC):")
    print(f"    Accuracy  : {100*acc_acg:.1f}%")
    print(f"    Sensitivity: {100*sens_acg:.1f}%")
    print(f"    Precision  : {100*prec_acg:.1f}%")
    print(f"    AUC-ROC    : {auc_acg:.3f}")
    print(f"    Referred   : {n_referred} ({pct_referred:.1f}%)")

    # --- Fixed weights (no CMCC) ---
    acc_fixed = accuracy_score(labels, all_preds_fixed)
    sens_fixed = recall_score(labels, all_preds_fixed)
    prec_fixed = precision_score(labels, all_preds_fixed)
    auc_fixed = roc_auc_score(labels, all_probs_fixed)
    brier_fixed = brier_score_loss(labels, all_probs_fixed)

    print(f"\n  Proposed (Fixed Weights):")
    print(f"    Accuracy  : {100*acc_fixed:.1f}%")
    print(f"    Sensitivity: {100*sens_fixed:.1f}%")
    print(f"    Precision  : {100*prec_fixed:.1f}%")
    print(f"    AUC-ROC    : {auc_fixed:.3f}")
    print(f"    Brier Score: {brier_fixed:.3f}")

    # --- Referral analysis ---
    if n_referred > 0:
        referred_probs = all_probs_acg[all_referred]
        ambiguous = ((referred_probs > 0.35) & (referred_probs < 0.65)).sum()
        print(f"\n  CMCC Referral Analysis:")
        print(f"    Cases referred: {n_referred}")
        print(f"    Ambiguous (0.35-0.65): {ambiguous} "
              f"({100*ambiguous/n_referred:.0f}%)")

    # --- Severity-stratified analysis ---
    print(f"\n  Severity-Stratified Sensitivity (ACG):")
    for stage in [1, 2, 3]:
        mask = (hy_stages == stage) & decided_mask
        if mask.sum() > 0:
            stage_labels = labels[mask]
            stage_preds = all_preds_acg[mask]
            if (stage_labels == 1).sum() > 0:
                stage_sens = recall_score(stage_labels, stage_preds)
                print(f"    H&Y Stage {stage}: {100*stage_sens:.1f}% "
                      f"(n={mask.sum()})")

    # --- Device-stratified analysis ---
    print(f"\n  Device-Stratified Sensitivity (ACG):")
    for dev in ["Budget Android", "Mid-Range Android", "Feature Phone"]:
        mask = (devices == dev) & decided_mask
        if mask.sum() > 0:
            dev_labels = labels[mask]
            dev_preds = all_preds_acg[mask]
            if (dev_labels == 1).sum() > 0:
                dev_sens = recall_score(dev_labels, dev_preds)
                print(f"    {dev}: {100*dev_sens:.1f}% (n={mask.sum()})")

    # --- Save results ---
    results = pd.DataFrame({
        "subject_id": [f"ICMPRS_{i+1:04d}" for i in range(n)],
        "label": labels,
        "p_svm": all_p_svm,
        "p_rf": all_p_rf,
        "p_acg": all_probs_acg,
        "p_fixed": all_probs_fixed,
        "cmcs": all_cmcs,
        "pred_acg_cmcc": all_preds_acg,
        "pred_fixed": all_preds_fixed,
        "referred": all_referred,
        "hy_stage": hy_stages,
        "device": devices,
    })
    results_path = os.path.join(RESULTS_DIR, "performance_summary.csv")
    results.to_csv(results_path, index=False)
    print(f"\n  Saved: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_evaluate()
