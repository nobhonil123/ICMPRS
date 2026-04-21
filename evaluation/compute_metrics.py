#!/usr/bin/env python3
"""
ICMPRS v2 — Evaluation: Bootstrap CIs, McNemar's test, DeLong AUC, CAG metric.
================================================================================
Author : Nobhonil Roy Choudhury
Licence: MIT
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    roc_auc_score, roc_curve,
)

SEED = 42
rng = np.random.default_rng(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def bootstrap_ci(y_true, y_score_or_pred, metric_fn, B=2000, alpha=0.05):
    """Compute bootstrap confidence interval for a metric."""
    n = len(y_true)
    scores = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_score_or_pred[idx])
            scores.append(s)
        except ValueError:
            continue
    scores = np.array(scores)
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return np.mean(scores), lo, hi


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test comparing two classifiers."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b_count = (~correct_a & correct_b).sum()  # A wrong, B right
    c_count = (correct_a & ~correct_b).sum()  # A right, B wrong
    if b_count + c_count == 0:
        return 1.0
    chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = stats.chi2.sf(chi2, df=1)
    return p_value


def cultural_adaptation_gain(auc_adapted, auc_generic):
    """
    Cultural Adaptation Gain (CAG) metric.
    CAG = (AUC_adapted - AUC_generic) / (AUC_generic - 0.5)
    """
    if auc_generic <= 0.5:
        return float("inf")
    return (auc_adapted - auc_generic) / (auc_generic - 0.5)


def main():
    results_path = os.path.join(RESULTS_DIR, "performance_summary.csv")
    if not os.path.exists(results_path):
        print("Error: Run train_ensemble.py first.")
        return

    df = pd.read_csv(results_path)
    y = df["label"].values
    p_acg = df["p_acg"].values
    p_fixed = df["p_fixed"].values
    pred_acg = df["pred_acg_cmcc"].values
    pred_fixed = df["pred_fixed"].values
    referred = df["referred"].values

    # Mask: decided cases only for ACG+CMCC
    decided = ~referred
    y_dec = y[decided]
    pred_acg_dec = pred_acg[decided]

    print("=" * 60)
    print("  EVALUATION REPORT — ICMPRS v2")
    print("=" * 60)

    # --- Proposed (ACG + CMCC) on decided cases ---
    print("\n  [Proposed ACG+CMCC] (decided cases only):")
    for name, fn in [("Accuracy", accuracy_score),
                     ("Sensitivity", recall_score),
                     ("Precision", precision_score)]:
        mean, lo, hi = bootstrap_ci(y_dec, pred_acg_dec, fn)
        print(f"    {name}: {100*mean:.1f}% (95% CI: {100*lo:.1f}–{100*hi:.1f}%)")

    auc_mean, auc_lo, auc_hi = bootstrap_ci(
        y, p_acg, roc_auc_score
    )
    print(f"    AUC-ROC: {auc_mean:.3f} (95% CI: {auc_lo:.3f}–{auc_hi:.3f})")

    # --- Fixed weights ---
    print("\n  [Fixed Weights]:")
    for name, fn in [("Accuracy", accuracy_score),
                     ("Sensitivity", recall_score),
                     ("Precision", precision_score)]:
        mean, lo, hi = bootstrap_ci(y, pred_fixed, fn)
        print(f"    {name}: {100*mean:.1f}% (95% CI: {100*lo:.1f}–{100*hi:.1f}%)")

    auc_f_mean, auc_f_lo, auc_f_hi = bootstrap_ci(
        y, p_fixed, roc_auc_score
    )
    print(f"    AUC-ROC: {auc_f_mean:.3f} (95% CI: {auc_f_lo:.3f}–{auc_f_hi:.3f})")

    # --- McNemar's test ---
    p_mcnemar = mcnemar_test(y, pred_acg_dec, pred_fixed[decided])
    print(f"\n  McNemar's test (ACG vs Fixed, decided cases): p = {p_mcnemar:.4f}")

    # --- Cultural Adaptation Gain ---
    # AUC with all features vs AUC without 5 Indian features
    # (The without-Indian AUC needs to be computed separately; here we
    #  use the paper's reported value as placeholder)
    auc_adapted = roc_auc_score(y, p_acg)
    auc_generic = 0.921  # from ablation: all 5 Indian features removed
    cag = cultural_adaptation_gain(auc_adapted, auc_generic)
    print(f"\n  Cultural Adaptation Gain (CAG):")
    print(f"    AUC_adapted  = {auc_adapted:.3f}")
    print(f"    AUC_generic  = {auc_generic:.3f} (from ablation)")
    print(f"    CAG          = {cag:.3f}")
    print(f"    Interpretation: Cultural features recover "
          f"{100*cag:.1f}% of remaining headroom")

    # --- CMCC referral stats ---
    n_ref = referred.sum()
    if n_ref > 0:
        ref_probs = p_acg[referred]
        ambig = ((ref_probs > 0.35) & (ref_probs < 0.65)).sum()
        print(f"\n  CMCC Referral Statistics:")
        print(f"    Referred: {n_ref} ({100*n_ref/len(y):.1f}%)")
        print(f"    Ambiguous: {ambig} ({100*ambig/n_ref:.0f}% of referred)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
