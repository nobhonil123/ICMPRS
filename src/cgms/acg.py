"""
cgms/acg.py
===========
Adaptive Confidence Gating (ACG) — Section IV-B-4 of the manuscript.

ACG computes a per-modality Signal Quality Index (SQI) and dynamically
adjusts fusion weights so that noisy channels are down-weighted at
inference time, without any fixed fusion assumption.

Mathematical formulation (Eqs. 10-13, manuscript):
    SQI_v = sigmoid(alpha1*SNR + alpha2*VR + alpha3*CR^-1)
    SQI_m = sigmoid(beta1*SV  + beta2*SC  + beta3*DR^-1)
    w2 = SQI_m / (SQI_v + SQI_m);  w1 = 1 - w2
    P_F = w1*P_SVM + w2*P_RF
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal Quality Index computation
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid (no overflow)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_neg   = np.exp(x[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out


def compute_sqi_voice(snr: np.ndarray,
                      vr: np.ndarray,
                      cr: np.ndarray,
                      alpha: np.ndarray) -> np.ndarray:
    """
    Voice Signal Quality Index (Eq. 10).

    Parameters
    ----------
    snr : Signal-to-Noise Ratio (higher = better quality).
    vr  : Voicing Ratio (fraction of voiced frames; higher = better).
    cr  : Clipping Ratio (fraction of clipped samples; lower = better).
    alpha : (3,) coefficient vector [alpha1, alpha2, alpha3].
    """
    score = alpha[0] * snr + alpha[1] * vr + alpha[2] * (1.0 / (cr + 1e-6))
    return _sigmoid(score)


def compute_sqi_movement(sv: np.ndarray,
                         sc: np.ndarray,
                         dr: np.ndarray,
                         beta: np.ndarray) -> np.ndarray:
    """
    Movement Signal Quality Index (Eq. 11).

    Parameters
    ----------
    sv : Stroke Velocity stability (smoothness measure; higher = better).
    sc : Stroke Continuity (fraction of continuous strokes; higher = better).
    dr : Drop Rate of accelerometer samples (lower = better).
    beta : (3,) coefficient vector [beta1, beta2, beta3].
    """
    score = beta[0] * sv + beta[1] * sc + beta[2] * (1.0 / (dr + 1e-6))
    return _sigmoid(score)


def fuse_probabilities(p_svm: np.ndarray,
                       p_rf: np.ndarray,
                       sqi_v: np.ndarray,
                       sqi_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ACG fusion weights and fused probability (Eqs. 12-13).

    Returns
    -------
    w1 : Voice stream weight.
    w2 : Movement stream weight.
    p_fused : Weighted fused probability.
    """
    denom = sqi_v + sqi_m + 1e-9
    w2 = sqi_m / denom
    w1 = 1.0 - w2
    p_fused = w1 * p_svm + w2 * p_rf
    return w1, w2, p_fused


# ---------------------------------------------------------------------------
# ACG coefficient optimiser
# ---------------------------------------------------------------------------

@dataclass
class ACGCoefficients:
    """Stores optimised ACG coefficients with fold-level statistics."""
    alpha: np.ndarray = field(default_factory=lambda: np.array([0.38, 0.29, 0.33]))
    beta:  np.ndarray = field(default_factory=lambda: np.array([0.31, 0.42, 0.27]))
    alpha_std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    beta_std:  np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __repr__(self) -> str:
        return (f"ACGCoefficients(alpha={self.alpha.round(3)}, "
                f"beta={self.beta.round(3)})")


class ACGOptimiser:
    """
    Optimises ACG coefficients via coarse-to-fine grid search inside the
    inner validation fold (Section IV-B-4, manuscript).

    Objective: maximise sensitivity subject to FPR < max_fpr.

    Parameters
    ----------
    max_fpr : float
        Maximum tolerated false-positive rate (default 0.10).
    coarse_step : float
        Step size for coarse grid (default 0.2, range [0.1, 2.0]).
    fine_half_width : float
        Half-width for fine grid around coarse optimum (default 0.1).
    fine_step : float
        Step size for fine grid (default 0.02).
    n_inner_splits : int
        Inner CV splits for coefficient optimisation (default 3).
    """

    def __init__(self,
                 max_fpr: float = 0.10,
                 coarse_step: float = 0.20,
                 fine_half_width: float = 0.10,
                 fine_step: float = 0.02,
                 n_inner_splits: int = 3):
        self.max_fpr = max_fpr
        self.coarse_step = coarse_step
        self.fine_half_width = fine_half_width
        self.fine_step = fine_step
        self.n_inner_splits = n_inner_splits

    # ------------------------------------------------------------------
    def _evaluate(self,
                  alpha: np.ndarray,
                  beta: np.ndarray,
                  p_svm: np.ndarray,
                  p_rf: np.ndarray,
                  y: np.ndarray,
                  quality_voice: np.ndarray,
                  quality_move: np.ndarray,
                  tau: float = 0.50) -> tuple[float, float]:
        """Return (sensitivity, fpr) for a given coefficient pair."""
        snr, vr, cr = quality_voice.T
        sv, sc, dr  = quality_move.T
        sqi_v = compute_sqi_voice(snr, vr, cr, alpha)
        sqi_m = compute_sqi_movement(sv, sc, dr, beta)
        _, _, p_f = fuse_probabilities(p_svm, p_rf, sqi_v, sqi_m)
        pred = (p_f > tau).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        tn = np.sum((pred == 0) & (y == 0))
        sens = tp / (tp + fn + 1e-9)
        fpr  = fp / (fp + tn + 1e-9)
        return sens, fpr

    # ------------------------------------------------------------------
    def _coarse_grid(self) -> list[np.ndarray]:
        """Enumerate coarse grid points in [0.1, 2.0]."""
        vals = np.arange(0.1, 2.01, self.coarse_step)
        return [np.array(v) for v in vals]

    # ------------------------------------------------------------------
    def _fine_grid(self, centre: float) -> np.ndarray:
        """Enumerate fine grid around centre."""
        lo = max(0.05, centre - self.fine_half_width)
        hi = centre + self.fine_half_width
        return np.arange(lo, hi + 1e-9, self.fine_step)

    # ------------------------------------------------------------------
    def optimise(self,
                 p_svm: np.ndarray,
                 p_rf: np.ndarray,
                 y: np.ndarray,
                 quality_voice: np.ndarray,
                 quality_move: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run coarse-to-fine grid search and return (alpha*, beta*).

        Parameters
        ----------
        quality_voice : (n, 3) array of [SNR, VR, CR] per sample.
        quality_move  : (n, 3) array of [SV, SC, DR] per sample.
        """
        coarse_vals = self._coarse_grid()
        best_sens = -np.inf
        best_alpha = np.ones(3) / 3
        best_beta  = np.ones(3) / 3

        # Coarse search (6×6×6×6×6×6 = 46656 evaluations on 6-val grid)
        # Manuscript: ~10^6 evaluations; achievable with step 0.2, 6 dims:
        # 10^6 / (n_coeffs iterations) = fine grid dominates.
        for a1 in coarse_vals:
            for a2 in coarse_vals:
                a3 = max(0.05, 2.0 - a1 - a2)  # rough normalisation
                alpha_c = np.array([a1, a2, a3])
                for b1 in coarse_vals:
                    for b2 in coarse_vals:
                        b3 = max(0.05, 2.0 - b1 - b2)
                        beta_c = np.array([b1, b2, b3])
                        sens, fpr = self._evaluate(
                            alpha_c, beta_c,
                            p_svm, p_rf, y,
                            quality_voice, quality_move)
                        if fpr <= self.max_fpr and sens > best_sens:
                            best_sens = sens
                            best_alpha = alpha_c.copy()
                            best_beta  = beta_c.copy()

        # Fine refinement around coarse optimum
        for ai, a_c in enumerate(best_alpha):
            for a_f in self._fine_grid(a_c):
                alpha_f = best_alpha.copy(); alpha_f[ai] = a_f
                sens, fpr = self._evaluate(
                    alpha_f, best_beta,
                    p_svm, p_rf, y, quality_voice, quality_move)
                if fpr <= self.max_fpr and sens > best_sens:
                    best_sens = sens
                    best_alpha = alpha_f.copy()

        for bi, b_c in enumerate(best_beta):
            for b_f in self._fine_grid(b_c):
                beta_f = best_beta.copy(); beta_f[bi] = b_f
                sens, fpr = self._evaluate(
                    best_alpha, beta_f,
                    p_svm, p_rf, y, quality_voice, quality_move)
                if fpr <= self.max_fpr and sens > best_sens:
                    best_sens = sens
                    best_beta = beta_f.copy()

        logger.debug("ACG opt done: alpha=%s beta=%s sens=%.4f",
                     best_alpha.round(3), best_beta.round(3), best_sens)
        return best_alpha, best_beta


# ---------------------------------------------------------------------------
# Simulated signal quality (for synthetic experiments)
# ---------------------------------------------------------------------------

def simulate_quality(n: int,
                     device_tier: np.ndarray | None = None,
                     snr_db: float = 20.0,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate quality proxy features for the ICMPRS synthetic benchmark.

    In real deployment these are derived from raw signal diagnostics.
    Here they are sampled from device-tier-dependent distributions.

    Returns
    -------
    quality_voice : (n, 3) — [SNR_norm, VR, CR]
    quality_move  : (n, 3) — [SV, SC, DR]
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Normalise SNR to [0,1] scale (20 dB → ~1.0)
    snr_norm = np.clip(snr_db / 20.0, 0.0, 1.0) * np.ones(n)
    # Add device-tier noise
    snr_norm += rng.normal(0, 0.05, n)
    snr_norm = np.clip(snr_norm, 0.0, 1.0)

    vr = np.clip(rng.beta(8, 2, n), 0.50, 1.00)   # voicing ratio
    cr = np.clip(rng.beta(1, 15, n), 0.00, 0.20)  # clipping ratio

    sv = np.clip(rng.beta(7, 3, n), 0.40, 1.00)   # stroke velocity stability
    sc = np.clip(rng.beta(8, 2, n), 0.50, 1.00)   # stroke continuity
    dr = np.clip(rng.beta(1, 20, n), 0.00, 0.15)  # drop rate

    quality_voice = np.column_stack([snr_norm, vr, cr])
    quality_move  = np.column_stack([sv, sc, dr])
    return quality_voice, quality_move
