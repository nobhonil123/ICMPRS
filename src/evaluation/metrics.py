"""
evaluation/metrics.py
=====================
Evaluation utilities for the ICMPRS/CGMS paper.

Covers:
  - Full diagnostic metrics table (Table VI, manuscript)
  - McNemar's test with Bonferroni correction
  - DeLong AUC confidence intervals
  - Brier score and calibration
  - Cultural Adaptation Gain (CAG, Definition 2 / Eq. 16)
  - Conditional distribution-shift bound (Proposition 1 / Eq. 18)
  - Prevalence-adjusted PPV/NPV (Eq. 19)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             confusion_matrix)

# ---------------------------------------------------------------------------
# Diagnostic metrics bundle
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticMetrics:
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    auc: float
    brier: float
    ppv: float
    npv: float
    # 95% bootstrap CIs
    acc_ci: tuple[float, float] = (0.0, 0.0)
    sens_ci: tuple[float, float] = (0.0, 0.0)
    auc_ci: tuple[float, float] = (0.0, 0.0)

    def as_dict(self) -> dict:
        return {
            "Accuracy":    f"{self.accuracy*100:.1f} ({self.acc_ci[0]*100:.1f}–{self.acc_ci[1]*100:.1f})",
            "Sensitivity": f"{self.sensitivity*100:.1f} ({self.sens_ci[0]*100:.1f}–{self.sens_ci[1]*100:.1f})",
            "Specificity": f"{self.specificity*100:.1f}",
            "Precision":   f"{self.precision*100:.1f}",
            "F1":          f"{self.f1*100:.1f}",
            "AUC":         f"{self.auc:.3f} ({self.auc_ci[0]:.3f}–{self.auc_ci[1]:.3f})",
            "Brier":       f"{self.brier:.3f}",
            "PPV":         f"{self.ppv*100:.1f}",
            "NPV":         f"{self.npv*100:.1f}",
        }


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    n_boot: int = 2000,
                    seed: int = 42) -> DiagnosticMetrics:
    """
    Compute full diagnostic metrics bundle with bootstrap CIs.

    Parameters
    ----------
    y_true : ground-truth binary labels.
    y_pred : predicted binary labels (referred cases excluded beforehand).
    y_prob : predicted probabilities (for AUC/Brier).
    """
    rng = np.random.default_rng(seed)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc  = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    ppv  = tp / (tp + fp + 1e-9)
    npv  = tn / (tn + fn + 1e-9)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    # Bootstrap CIs
    acc_scores, sens_scores, auc_scores = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            acc_scores.append(accuracy_score(y_true[idx], y_pred[idx]))
            sens_scores.append(recall_score(y_true[idx], y_pred[idx],
                                            zero_division=0))
            auc_scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
        except Exception:
            continue

    ci = lambda s: (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))

    return DiagnosticMetrics(
        accuracy=acc, sensitivity=sens, specificity=spec,
        precision=ppv, f1=f1, auc=auc, brier=brier, ppv=ppv, npv=npv,
        acc_ci=ci(acc_scores),
        sens_ci=ci(sens_scores),
        auc_ci=ci(auc_scores),
    )


# ---------------------------------------------------------------------------
# McNemar's test with Bonferroni correction
# ---------------------------------------------------------------------------

def mcnemar_bonferroni(y_true: np.ndarray,
                       preds_a: np.ndarray,
                       preds_b: np.ndarray,
                       n_comparisons: int = 5) -> dict:
    """
    McNemar's test between model A and model B with Bonferroni correction.

    Returns
    -------
    dict with keys: 'chi2', 'p_raw', 'p_corrected', 'significant'
    """
    # Contingency: (A right B wrong), (A wrong B right)
    ab = ((preds_a == y_true) & (preds_b != y_true)).sum()
    ba = ((preds_a != y_true) & (preds_b == y_true)).sum()
    denom = ab + ba
    if denom == 0:
        return {"chi2": 0.0, "p_raw": 1.0,
                "p_corrected": 1.0, "significant": False}
    chi2 = (abs(ab - ba) - 1) ** 2 / denom
    p_raw = float(stats.chi2.sf(chi2, df=1))
    p_corr = min(1.0, p_raw * n_comparisons)
    return {
        "chi2": float(chi2),
        "p_raw": p_raw,
        "p_corrected": p_corr,
        "significant": p_corr < 0.01,   # Bonferroni threshold α=0.01
    }


# ---------------------------------------------------------------------------
# Cultural Adaptation Gain (CAG) — Definition 2, Eq. 16
# ---------------------------------------------------------------------------

def cultural_adaptation_gain(auc_adapted: float,
                              auc_generic: float) -> float:
    """
    CAG = (AUC_ad - AUC_gen) / (AUC_gen - 0.5).

    Returns NaN if AUC_gen <= 0.5 (degenerate case).
    Positive CAG is guaranteed by construction on the synthetic benchmark;
    interpret the returned value as an upper-bound estimate pending
    real-cohort confirmation.
    """
    if auc_generic <= 0.5:
        warnings.warn("AUC_generic <= 0.5 — CAG is undefined.")
        return float("nan")
    return (auc_adapted - auc_generic) / (auc_generic - 0.5)


# ---------------------------------------------------------------------------
# Conditional distribution-shift bound — Proposition 1, Eq. 18
# ---------------------------------------------------------------------------

def distribution_shift_bound(empirical_risk: float,
                              n: int,
                              epsilon: float,
                              delta_prime: float = 0.05) -> float:
    """
    Upper bound on true risk under distribution shift (Proposition 1).

        R_T <= R_hat_S + 2*epsilon + sqrt(ln(2/delta') / (2*N))

    Parameters
    ----------
    empirical_risk : R_hat_S (1 - accuracy on synthetic benchmark).
    n              : number of synthetic test samples.
    epsilon        : total variation distance dTV(S, T).
    delta_prime    : confidence level complement (default 0.05 → 95% CI).

    Returns
    -------
    Upper bound on R_T (as error rate; 1 - this = lower accuracy bound).
    """
    hoeffding_term = np.sqrt(np.log(2.0 / delta_prime) / (2.0 * n))
    return empirical_risk + 2.0 * epsilon + hoeffding_term


def accuracy_lower_bound(synthetic_accuracy: float,
                         n: int,
                         epsilon: float,
                         delta_prime: float = 0.05) -> float:
    """Convenience wrapper returning accuracy lower bound."""
    r_hat = 1.0 - synthetic_accuracy
    r_upper = distribution_shift_bound(r_hat, n, epsilon, delta_prime)
    return max(0.0, 1.0 - r_upper)


# ---------------------------------------------------------------------------
# Prevalence-adjusted PPV / NPV — Eq. 19
# ---------------------------------------------------------------------------

def prevalence_adjusted_ppv_npv(sensitivity: float,
                                 specificity: float,
                                 prevalence: float) -> tuple[float, float]:
    """
    Compute PPV and NPV at a given population prevalence.

        PPV = (sens * prev) / (sens * prev + (1-spec) * (1-prev))
        NPV = (spec * (1-prev)) / (spec * (1-prev) + (1-sens) * prev)
    """
    p = prevalence
    ppv_num = sensitivity * p
    ppv_den = sensitivity * p + (1.0 - specificity) * (1.0 - p)
    ppv = ppv_num / ppv_den if ppv_den > 0 else float("nan")

    npv_num = specificity * (1.0 - p)
    npv_den = specificity * (1.0 - p) + (1.0 - sensitivity) * p
    npv = npv_num / npv_den if npv_den > 0 else float("nan")

    return ppv, npv


# ---------------------------------------------------------------------------
# MMD calibration check (Table III-B)
# ---------------------------------------------------------------------------

def mmd_squared(X: np.ndarray,
                Y: np.ndarray,
                gamma: float | None = None) -> float:
    """
    Unbiased estimator of MMD² with Gaussian kernel.
    Used in Table III calibration checks (Section III-G).
    """
    n, m = len(X), len(Y)
    if gamma is None:
        # Median heuristic
        from scipy.spatial.distance import pdist
        all_pts = np.vstack([X, Y])
        gamma = 1.0 / (2.0 * np.median(pdist(all_pts)) ** 2 + 1e-9)

    def _rbf_matrix(A, B):
        sq_dist = (np.sum(A**2, axis=1)[:, None]
                   + np.sum(B**2, axis=1)[None, :]
                   - 2.0 * A @ B.T)
        return np.exp(-gamma * sq_dist)

    Kxx = _rbf_matrix(X, X)
    Kyy = _rbf_matrix(Y, Y)
    Kxy = _rbf_matrix(X, Y)

    mmd2 = (Kxx.sum() / (n * (n - 1))
            + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean())
    return float(mmd2)
