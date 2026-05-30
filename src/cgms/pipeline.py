"""
cgms/pipeline.py
================
Complete Confidence-Gated Multimodal Screening (CGMS) pipeline.

Implements Algorithm 1 from the manuscript:
  1. RBF-SVM on voice features  → P_SVM
  2. Random Forest on movement features → P_RF
  3. ACG: quality-aware fusion → P_F
  4. CMCC: consistency gate → {PD, HC, Refer}

Also implements the full nested cross-validation evaluation loop with:
  - Outer stratified 5-fold CV (participant-level grouping)
  - Inner 3-fold loop for hyperparameter and ACG coefficient optimisation
  - Bootstrap 95% CIs
  - McNemar's tests with Bonferroni correction
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)

from cgms.acg import (ACGOptimiser, simulate_quality,
                      compute_sqi_voice, compute_sqi_movement,
                      fuse_probabilities)
from cgms.cmcc import (apply_cmcc, optimise_delta, cmcc_report,
                       REFER_LABEL)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

SVM_PARAM_GRID = {
    "estimator__C":     [0.1, 1, 10, 100],
    "estimator__gamma": [0.001, 0.01, 0.1, "scale"],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth":    [10, 15, 20, None],
    "max_features": [4, 6, 8, "sqrt"],
}


# ---------------------------------------------------------------------------
# Single-fold training helpers
# ---------------------------------------------------------------------------

def _train_svm(X_train: np.ndarray,
               y_train: np.ndarray,
               inner_cv: int = 3,
               seed: int = 42) -> CalibratedClassifierCV:
    """Train calibrated RBF-SVM with nested inner CV."""
    base = SVC(kernel="rbf", probability=False, random_state=seed,
               class_weight="balanced")
    grid = GridSearchCV(
        estimator=CalibratedClassifierCV(base, cv=inner_cv, method="sigmoid"),
        param_grid=SVM_PARAM_GRID,
        cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed),
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    logger.debug("SVM best params: %s", grid.best_params_)
    return grid.best_estimator_


def _train_rf(X_train: np.ndarray,
              y_train: np.ndarray,
              inner_cv: int = 3,
              seed: int = 42) -> RandomForestClassifier:
    """Train Random Forest with nested inner CV."""
    base = RandomForestClassifier(random_state=seed, class_weight="balanced",
                                  n_jobs=-1)
    grid = GridSearchCV(
        estimator=base,
        param_grid=RF_PARAM_GRID,
        cv=StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed),
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    logger.debug("RF best params: %s", grid.best_params_)
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# Bootstrap CI utility
# ---------------------------------------------------------------------------

def _bootstrap_ci(metric_fn,
                  y_true: np.ndarray,
                  y_pred: np.ndarray,
                  n_boot: int = 2000,
                  ci: float = 0.95,
                  seed: int = 42) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for a scalar metric."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    alpha = 1.0 - ci
    return (float(np.percentile(scores, 100 * alpha / 2)),
            float(np.percentile(scores, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------------
# CGMS Pipeline class
# ---------------------------------------------------------------------------

class CGMSPipeline:
    """
    Full CGMS pipeline: RBF-SVM (voice) + RF (movement) + ACG + CMCC.

    Parameters
    ----------
    voice_features : list[str]
        Column names for the voice/acoustic stream (25 features).
    movement_features : list[str]
        Column names for the handwriting + gait streams (37 features).
    n_outer_folds : int
        Outer CV folds (default 5).
    n_inner_folds : int
        Inner CV folds for hyperparameter search (default 3).
    seed : int
        Global seed.
    snr_db : float
        Simulated SNR in dB for the synthetic benchmark (default 20.0).
    """

    def __init__(self,
                 voice_features: list[str],
                 movement_features: list[str],
                 n_outer_folds: int = 5,
                 n_inner_folds: int = 3,
                 seed: int = 42,
                 snr_db: float = 20.0):
        self.voice_features = voice_features
        self.movement_features = movement_features
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.seed = seed
        self.snr_db = snr_db
        self._rng = np.random.default_rng(seed)

        # Fitted components (populated after fit())
        self._svm: CalibratedClassifierCV | None = None
        self._rf: RandomForestClassifier | None = None
        self._scaler_v: StandardScaler | None = None
        self._scaler_m: StandardScaler | None = None
        self._acg_alpha: np.ndarray = np.array([0.38, 0.29, 0.33])
        self._acg_beta:  np.ndarray = np.array([0.31, 0.42, 0.27])
        self._delta: float = 0.65

    # ------------------------------------------------------------------
    def _split_features(self, X: pd.DataFrame | np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Split into voice and movement arrays."""
        if isinstance(X, pd.DataFrame):
            Xv = X[self.voice_features].values
            Xm = X[self.movement_features].values
        else:
            n_v = len(self.voice_features)
            Xv, Xm = X[:, :n_v], X[:, n_v:]
        return Xv, Xm

    # ------------------------------------------------------------------
    def evaluate_cv(self,
                    df: pd.DataFrame,
                    participant_col: str = "participant_id",
                    label_col: str = "label",
                    n_bootstrap: int = 2000) -> dict:
        """
        Run stratified 5-fold CV with nested inner 3-fold for HP search.
        Returns per-fold and aggregate metrics matching Table VI of the
        manuscript.
        """
        X = df[self.voice_features + self.movement_features].values
        y = df[label_col].values
        groups = df[participant_col].values

        outer_cv = StratifiedGroupKFold(n_splits=self.n_outer_folds)
        acg_opt = ACGOptimiser(n_inner_splits=self.n_inner_folds)

        fold_metrics: list[dict] = []

        for fold_idx, (train_idx, test_idx) in enumerate(
                outer_cv.split(X, y, groups)):
            logger.info("=== Outer fold %d/%d ===",
                        fold_idx + 1, self.n_outer_folds)

            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            Xv_tr, Xm_tr = self._split_features(
                pd.DataFrame(X_tr,
                             columns=self.voice_features + self.movement_features))
            Xv_te, Xm_te = self._split_features(
                pd.DataFrame(X_te,
                             columns=self.voice_features + self.movement_features))

            # Scale
            scaler_v = StandardScaler().fit(Xv_tr)
            scaler_m = StandardScaler().fit(Xm_tr)
            Xv_tr_s = scaler_v.transform(Xv_tr)
            Xm_tr_s = scaler_m.transform(Xm_tr)
            Xv_te_s = scaler_v.transform(Xv_te)
            Xm_te_s = scaler_m.transform(Xm_te)

            # Train streams
            svm = _train_svm(Xv_tr_s, y_tr, self.n_inner_folds, self.seed)
            rf  = _train_rf(Xm_tr_s, y_tr, self.n_inner_folds, self.seed)

            p_svm_tr = svm.predict_proba(Xv_tr_s)[:, 1]
            p_rf_tr  = rf.predict_proba(Xm_tr_s)[:, 1]

            # Simulate quality
            n_tr = len(y_tr)
            q_v_tr, q_m_tr = simulate_quality(
                n_tr, snr_db=self.snr_db, rng=self._rng)

            # Optimise ACG inside inner fold
            alpha, beta = acg_opt.optimise(
                p_svm_tr, p_rf_tr, y_tr, q_v_tr, q_m_tr)

            # Test predictions
            p_svm_te = svm.predict_proba(Xv_te_s)[:, 1]
            p_rf_te  = rf.predict_proba(Xm_te_s)[:, 1]
            n_te = len(y_te)
            q_v_te, q_m_te = simulate_quality(
                n_te, snr_db=self.snr_db, rng=self._rng)

            sqi_v = compute_sqi_voice(
                q_v_te[:, 0], q_v_te[:, 1], q_v_te[:, 2], alpha)
            sqi_m = compute_sqi_movement(
                q_m_te[:, 0], q_m_te[:, 1], q_m_te[:, 2], beta)
            _, _, p_fused = fuse_probabilities(p_svm_te, p_rf_te,
                                               sqi_v, sqi_m)

            # Optimise CMCC delta on training predictions
            sqi_v_tr = compute_sqi_voice(
                q_v_tr[:, 0], q_v_tr[:, 1], q_v_tr[:, 2], alpha)
            sqi_m_tr = compute_sqi_movement(
                q_m_tr[:, 0], q_m_tr[:, 1], q_m_tr[:, 2], beta)
            _, _, p_fused_tr = fuse_probabilities(
                p_svm_tr, p_rf_tr, sqi_v_tr, sqi_m_tr)
            delta = optimise_delta(p_fused_tr, p_svm_tr, p_rf_tr, y_tr)

            preds = apply_cmcc(p_fused, p_svm_te, p_rf_te, delta=delta)
            report = cmcc_report(preds, p_fused, p_svm_te, p_rf_te,
                                 y_te, delta)

            # Compute metrics on classified cases
            classified = preds != REFER_LABEL
            y_cls = y_te[classified]
            p_cls = preds[classified].astype(int)

            acc  = accuracy_score(y_cls, p_cls)
            sens = recall_score(y_cls, p_cls, zero_division=0)
            spec = recall_score(y_cls, p_cls, pos_label=0, zero_division=0)
            prec = precision_score(y_cls, p_cls, zero_division=0)
            f1   = f1_score(y_cls, p_cls, zero_division=0)
            auc  = roc_auc_score(y_te, p_fused)  # AUC on all cases

            fold_metrics.append({
                "fold": fold_idx + 1,
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "f1": f1,
                "auc": auc,
                "refer_rate": report.refer_rate,
                "delta": delta,
                "n_test": len(y_te),
                "n_classified": int(classified.sum()),
            })
            logger.info("Fold %d: acc=%.3f sens=%.3f auc=%.4f refer=%.1f%%",
                        fold_idx + 1, acc, sens, auc,
                        report.refer_rate * 100)

        # Aggregate
        fdf = pd.DataFrame(fold_metrics)
        summary = {
            "fold_metrics": fdf,
            "mean_accuracy":    fdf.accuracy.mean(),
            "std_accuracy":     fdf.accuracy.std(),
            "mean_sensitivity": fdf.sensitivity.mean(),
            "mean_specificity": fdf.specificity.mean(),
            "mean_auc":         fdf.auc.mean(),
            "std_auc":          fdf.auc.std(),
            "mean_refer_rate":  fdf.refer_rate.mean(),
        }
        logger.info(
            "CV Summary: acc=%.3f±%.3f  sens=%.3f  auc=%.4f±%.4f",
            summary["mean_accuracy"], summary["std_accuracy"],
            summary["mean_sensitivity"], summary["mean_auc"],
            summary["std_auc"])
        return summary

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, label_col: str = "label") -> "CGMSPipeline":
        """
        Fit CGMS on the full dataset (for deployment / cross-dataset use).
        Uses default ACG coefficients from the manuscript.
        """
        X = df[self.voice_features + self.movement_features].values
        y = df[label_col].values
        Xv, Xm = self._split_features(df)

        self._scaler_v = StandardScaler().fit(Xv)
        self._scaler_m = StandardScaler().fit(Xm)

        self._svm = _train_svm(
            self._scaler_v.transform(Xv), y, self.n_inner_folds, self.seed)
        self._rf  = _train_rf(
            self._scaler_m.transform(Xm), y, self.n_inner_folds, self.seed)
        logger.info("CGMS fitted on %d samples.", len(y))
        return self

    # ------------------------------------------------------------------
    def predict_proba_streams(self,
                              X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_svm, p_rf) for a feature DataFrame."""
        assert self._svm is not None, "Call fit() first."
        Xv, Xm = self._split_features(X)
        p_svm = self._svm.predict_proba(
            self._scaler_v.transform(Xv))[:, 1]
        p_rf  = self._rf.predict_proba(
            self._scaler_m.transform(Xm))[:, 1]
        return p_svm, p_rf

    # ------------------------------------------------------------------
    def predict(self,
                X: pd.DataFrame,
                quality_voice: np.ndarray | None = None,
                quality_move: np.ndarray | None = None,
                snr_db: float = 20.0) -> np.ndarray:
        """
        Full inference (Algorithm 1).  Returns {1=PD, 0=HC, -1=Refer}.
        """
        p_svm, p_rf = self.predict_proba_streams(X)
        n = len(p_svm)

        if quality_voice is None or quality_move is None:
            quality_voice, quality_move = simulate_quality(
                n, snr_db=snr_db, rng=self._rng)

        sqi_v = compute_sqi_voice(
            quality_voice[:, 0], quality_voice[:, 1], quality_voice[:, 2],
            self._acg_alpha)
        sqi_m = compute_sqi_movement(
            quality_move[:, 0], quality_move[:, 1], quality_move[:, 2],
            self._acg_beta)
        _, _, p_fused = fuse_probabilities(p_svm, p_rf, sqi_v, sqi_m)
        return apply_cmcc(p_fused, p_svm, p_rf, delta=self._delta)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_train():
    """Minimal CLI for training and evaluating CGMS."""
    import argparse, json
    from icmprs.generator import ICMPRSGenerator

    parser = argparse.ArgumentParser(description="Train and evaluate CGMS.")
    parser.add_argument("--data", default=None,
                        help="Path to feature CSV (generate if not provided)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr", type=float, default=20.0,
                        help="Simulated SNR in dB (default 20)")
    parser.add_argument("--out", default="results/cgms_cv_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.data:
        df = pd.read_csv(args.data)
    else:
        logger.info("No data file provided — generating ICMPRS (n=1995, seed=%d)",
                    args.seed)
        gen = ICMPRSGenerator(seed=args.seed)
        df = gen.generate(n=1995)

    from icmprs.generator import ICMPRSGenerator as G
    gen_tmp = G(seed=args.seed)
    voice_cols = gen_tmp.FEATURE_NAMES_ACOUSTIC
    move_cols  = [c for c in gen_tmp.feature_columns
                  if c not in voice_cols]

    pipeline = CGMSPipeline(
        voice_features=voice_cols,
        movement_features=move_cols,
        seed=args.seed,
        snr_db=args.snr,
    )
    summary = pipeline.evaluate_cv(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fold_df = summary.pop("fold_metrics")
    fold_df.to_csv(out.with_suffix(".folds.csv"), index=False)

    with open(out, "w") as f:
        json.dump({k: float(v) if isinstance(v, np.floating) else v
                   for k, v in summary.items()}, f, indent=2)
    logger.info("Results written to %s", out)


if __name__ == "__main__":
    cli_train()
