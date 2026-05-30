#!/usr/bin/env python3
"""
scripts/cross_dataset_eval.py
==============================
Evaluates CGMS on public real-patient datasets using only the
57 generic (non-Indian-specific) features — Table XII of the manuscript.

NOTE: Only the RBF-SVM voice stream is active here; ACG and CMCC
are NOT operative because handwriting/gait modalities are absent
from UCI Voice and mPower. See Section VI-I of the manuscript.

Datasets required
-----------------
UCI Parkinsons Voice:
  Download: https://archive.ics.uci.edu/ml/datasets/parkinsons
  Save as: data/uci_parkinsons.csv

mPower (curated subset, Synapse syn4993293):
  Access: https://www.synapse.org/#!Synapse:syn4993293
  Save as: data/mpower_voice.csv

Usage
-----
    python scripts/cross_dataset_eval.py \\
        --uci   data/uci_parkinsons.csv \\
        --mpower data/mpower_voice.csv \\
        --out   results/table_XII_crossdataset.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icmprs.generator import ICMPRSGenerator
from evaluation.metrics import (compute_metrics, prevalence_adjusted_ppv_npv,
                                  accuracy_lower_bound)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

# ---------------------------------------------------------------------------
# UCI feature name mapping (Little 2009 column names)
# ---------------------------------------------------------------------------
UCI_FEATURE_COLS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
    "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]
UCI_LABEL_COL = "status"   # 1=PD, 0=HC

# mPower feature subset (15 computable acoustic features)
MPOWER_FEATURE_COLS = [
    "F0", "jitter_local", "jitter_abs", "shimmer_local",
    "shimmer_db", "shimmer_apq3", "shimmer_apq11",
    "hnr", "rpde", "dfa", "spread1", "spread2", "d2", "ppe", "nhr",
]
MPOWER_LABEL_COL = "medTimepoint"   # mapped to binary below


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_svm_cv(X: np.ndarray,
                     y: np.ndarray,
                     n_folds: int = 10,
                     seed: int = SEED) -> dict:
    """Stratified k-fold CV for the SVM-only voice stream."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    all_probs, all_labels, all_preds = [], [], []

    for tr, te in cv.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        X_tr_s = scaler.transform(X[tr])
        X_te_s = scaler.transform(X[te])
        svc = CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, gamma=0.01, random_state=seed,
                class_weight="balanced"),
            cv=3, method="sigmoid"
        )
        svc.fit(X_tr_s, y[tr])
        prob = svc.predict_proba(X_te_s)[:, 1]
        all_probs.extend(prob)
        all_preds.extend((prob > 0.5).astype(int))
        all_labels.extend(y[te])

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_pred, y_prob, n_boot=2000, seed=seed)
    return {
        "accuracy": metrics.accuracy,
        "sensitivity": metrics.sensitivity,
        "specificity": metrics.specificity,
        "auc": metrics.auc,
        "acc_ci_lo": metrics.acc_ci[0],
        "acc_ci_hi": metrics.acc_ci[1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uci",    default=None, help="Path to UCI CSV")
    parser.add_argument("--mpower", default=None, help="Path to mPower CSV")
    parser.add_argument("--out",    default="results/table_XII_crossdataset.csv")
    args = parser.parse_args()

    rows = []

    # ── UCI Parkinsons Voice ─────────────────────────────────────────────
    if args.uci and Path(args.uci).exists():
        logger.info("Evaluating on UCI Parkinsons Voice (%s)…", args.uci)
        uci_df = pd.read_csv(args.uci)
        # Keep only the overlapping 21 features
        available = [c for c in UCI_FEATURE_COLS if c in uci_df.columns]
        X_uci = uci_df[available].values
        y_uci = uci_df[UCI_LABEL_COL].values.astype(int)
        logger.info("  n=%d  PD=%d  HC=%d  features=%d",
                    len(y_uci), y_uci.sum(), (y_uci == 0).sum(), X_uci.shape[1])
        res_uci = _evaluate_svm_cv(X_uci, y_uci, n_folds=10)
        res_uci["dataset"] = "UCI Voice"
        res_uci["n"] = len(y_uci)
        rows.append(res_uci)
        logger.info("  UCI: acc=%.1f%% (%.1f–%.1f) sens=%.1f%% auc=%.4f",
                    res_uci["accuracy"] * 100,
                    res_uci["acc_ci_lo"] * 100,
                    res_uci["acc_ci_hi"] * 100,
                    res_uci["sensitivity"] * 100,
                    res_uci["auc"])
    else:
        logger.warning("UCI file not found — skipping. "
                       "Download from archive.ics.uci.edu/ml/datasets/parkinsons")

    # ── mPower ──────────────────────────────────────────────────────────
    if args.mpower and Path(args.mpower).exists():
        logger.info("Evaluating on mPower (%s)…", args.mpower)
        mp_df = pd.read_csv(args.mpower)

        # Map to binary label (self-report)
        if "professional-diagnosis" in mp_df.columns:
            y_mp = mp_df["professional-diagnosis"].astype(int).values
        elif MPOWER_LABEL_COL in mp_df.columns:
            y_mp = (mp_df[MPOWER_LABEL_COL] != "immediately before Parkinson medication"
                    ).astype(int).values
        else:
            raise ValueError("Cannot identify mPower label column.")

        available = [c for c in MPOWER_FEATURE_COLS if c in mp_df.columns]
        X_mp = mp_df[available].dropna().values
        y_mp = y_mp[:len(X_mp)]
        logger.info("  n=%d  PD=%d  HC=%d  features=%d",
                    len(y_mp), y_mp.sum(), (y_mp == 0).sum(), X_mp.shape[1])
        res_mp = _evaluate_svm_cv(X_mp, y_mp, n_folds=10)
        res_mp["dataset"] = "mPower"
        res_mp["n"] = len(y_mp)
        rows.append(res_mp)
        logger.info("  mPower: acc=%.1f%% sens=%.1f%% auc=%.4f",
                    res_mp["accuracy"] * 100,
                    res_mp["sensitivity"] * 100,
                    res_mp["auc"])
    else:
        logger.warning("mPower file not found — skipping. "
                       "Access via synapse.org syn4993293")

    # ── Prevalence-adjusted PPV at 1% ────────────────────────────────────
    for row in rows:
        ppv, npv = prevalence_adjusted_ppv_npv(
            row["sensitivity"], row["specificity"], 0.01)
        row["ppv_at_1pct"] = round(ppv * 100, 1)
        row["npv_at_1pct"] = round(npv * 100, 2)

    # ── Distribution-shift epsilon ────────────────────────────────────────
    # Estimate eps from accuracy drop relative to ICMPRS-generic (89.6%)
    icmprs_generic_acc = 0.896
    for row in rows:
        drop = max(0.0, icmprs_generic_acc - row["accuracy"])
        row["epsilon_hat"] = round(drop / 2.0, 3)  # lower bound from Prop. 1

    # ── Save ──────────────────────────────────────────────────────────────
    if rows:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(rows)
        results_df.to_csv(out, index=False)
        logger.info("Results saved to %s", out)
        print(results_df.to_string(index=False))
    else:
        logger.warning("No datasets evaluated. "
                       "Provide --uci and/or --mpower paths.")


if __name__ == "__main__":
    main()
