"""
cgms/cmcc.py
============
Cross-Modal Consistency Check (CMCC) — Section IV-B-5 of the manuscript.

Definition 1 (Cross-Modal Consistency Score):
    CMCS = 1 - |P_SVM - P_RF|

Decision rule (Eq. 14):
    yhat = PD      if P_F > tau  AND CMCS >= delta
    yhat = HC      if P_F <= tau AND CMCS >= delta
    yhat = Refer   if CMCS < delta
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

REFER_LABEL = -1   # sentinel value for "refer to specialist"


# ---------------------------------------------------------------------------
# Core CMCC logic
# ---------------------------------------------------------------------------

def cross_modal_consistency_score(p_svm: np.ndarray,
                                  p_rf: np.ndarray) -> np.ndarray:
    """CMCS = 1 - |P_SVM - P_RF|  (Definition 1, manuscript)."""
    return 1.0 - np.abs(p_svm - p_rf)


def apply_cmcc(p_fused: np.ndarray,
               p_svm: np.ndarray,
               p_rf: np.ndarray,
               delta: float = 0.65,
               tau: float = 0.50) -> np.ndarray:
    """
    Apply CMCC decision rule (Eq. 14).

    Parameters
    ----------
    p_fused : ACG-fused probability per sample.
    p_svm   : Voice-stream SVM probability.
    p_rf    : Movement-stream RF probability.
    delta   : Consistency threshold (optimised per fold; default 0.65).
    tau     : Classification threshold (default 0.50).

    Returns
    -------
    predictions : int array  {1 = PD,  0 = HC,  -1 = Refer}
    """
    cmcs = cross_modal_consistency_score(p_svm, p_rf)
    predictions = np.where(
        cmcs < delta,
        REFER_LABEL,                          # refer
        np.where(p_fused > tau, 1, 0)         # PD or HC
    ).astype(int)
    return predictions


# ---------------------------------------------------------------------------
# Threshold optimiser
# ---------------------------------------------------------------------------

def optimise_delta(p_fused: np.ndarray,
                   p_svm: np.ndarray,
                   p_rf: np.ndarray,
                   y: np.ndarray,
                   delta_grid: np.ndarray | None = None,
                   target_sensitivity: float = 0.95,
                   tau: float = 0.50) -> float:
    """
    Find the smallest delta that keeps sensitivity >= target_sensitivity
    on non-referred cases (manuscript: minimise referral subject to
    sensitivity constraint).

    Parameters
    ----------
    delta_grid : sequence of delta values to evaluate.
                 Default: np.arange(0.50, 0.91, 0.01).
    target_sensitivity : minimum acceptable sensitivity on classified cases.

    Returns
    -------
    Optimal delta (float).
    """
    if delta_grid is None:
        delta_grid = np.arange(0.50, 0.91, 0.01)

    best_delta = 0.65   # fallback
    best_refer_rate = np.inf

    for delta in delta_grid:
        preds = apply_cmcc(p_fused, p_svm, p_rf, delta=delta, tau=tau)
        classified_mask = (preds != REFER_LABEL)
        refer_rate = 1.0 - classified_mask.mean()

        if classified_mask.sum() == 0:
            continue

        y_class = y[classified_mask]
        p_class = preds[classified_mask]
        tp = ((p_class == 1) & (y_class == 1)).sum()
        fn = ((p_class == 0) & (y_class == 1)).sum()
        sens = tp / (tp + fn + 1e-9)

        if sens >= target_sensitivity and refer_rate < best_refer_rate:
            best_refer_rate = refer_rate
            best_delta = delta

    logger.debug("CMCC optimised delta=%.2f (refer_rate=%.3f)",
                 best_delta, best_refer_rate)
    return float(best_delta)


# ---------------------------------------------------------------------------
# Summary statistics for CMCC output
# ---------------------------------------------------------------------------

@dataclass
class CMCCReport:
    """Statistics produced by the CMCC gate."""
    n_total: int
    n_referred: int
    n_classified: int
    refer_rate: float
    referred_in_borderline_zone: int    # |P_fused - 0.5| < 0.15
    referred_high_disagreement: int
    classified_sensitivity: float
    classified_specificity: float
    delta_used: float

    def __str__(self) -> str:
        lines = [
            f"CMCC Report (delta={self.delta_used:.2f})",
            f"  Total: {self.n_total}",
            f"  Referred: {self.n_referred} ({self.refer_rate*100:.1f}%)",
            f"    → borderline prob zone: {self.referred_in_borderline_zone}"
            f" ({100*self.referred_in_borderline_zone/max(1,self.n_referred):.0f}%)",
            f"    → high disagreement:    {self.referred_high_disagreement}"
            f" ({100*self.referred_high_disagreement/max(1,self.n_referred):.0f}%)",
            f"  Classified sensitivity: {self.classified_sensitivity*100:.1f}%",
            f"  Classified specificity: {self.classified_specificity*100:.1f}%",
        ]
        return "\n".join(lines)


def cmcc_report(preds: np.ndarray,
                p_fused: np.ndarray,
                p_svm: np.ndarray,
                p_rf: np.ndarray,
                y: np.ndarray,
                delta: float) -> CMCCReport:
    """Build a CMCCReport from predictions."""
    referred = (preds == REFER_LABEL)
    classified = ~referred

    n_ref = referred.sum()
    n_cls = classified.sum()
    refer_rate = n_ref / len(preds)

    # Categorise referred cases
    borderline = referred & (np.abs(p_fused - 0.5) < 0.15)
    high_disagree = referred & ~borderline

    # Classified-case metrics
    y_cls = y[classified]
    p_cls = preds[classified]
    tp = ((p_cls == 1) & (y_cls == 1)).sum()
    fn = ((p_cls == 0) & (y_cls == 1)).sum()
    fp = ((p_cls == 1) & (y_cls == 0)).sum()
    tn = ((p_cls == 0) & (y_cls == 0)).sum()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)

    return CMCCReport(
        n_total=len(preds),
        n_referred=int(n_ref),
        n_classified=int(n_cls),
        refer_rate=float(refer_rate),
        referred_in_borderline_zone=int(borderline.sum()),
        referred_high_disagreement=int(high_disagree.sum()),
        classified_sensitivity=float(sens),
        classified_specificity=float(spec),
        delta_used=delta,
    )
