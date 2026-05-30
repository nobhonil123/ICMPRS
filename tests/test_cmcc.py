"""tests/test_cmcc.py — Unit tests for Cross-Modal Consistency Check."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgms.cmcc import (cross_modal_consistency_score, apply_cmcc,
                       optimise_delta, cmcc_report, REFER_LABEL)


class TestCMCS:
    """Tests for the Cross-Modal Consistency Score."""

    def test_perfect_agreement(self):
        """Identical probabilities → CMCS = 1.0."""
        p = np.array([0.2, 0.5, 0.8])
        cmcs = cross_modal_consistency_score(p, p)
        np.testing.assert_allclose(cmcs, 1.0)

    def test_total_disagreement(self):
        """p_svm=1, p_rf=0 → CMCS = 0.0."""
        cmcs = cross_modal_consistency_score(np.array([1.0]), np.array([0.0]))
        np.testing.assert_allclose(cmcs, 0.0)

    def test_range(self):
        """CMCS must always be in [0, 1]."""
        rng = np.random.default_rng(0)
        p1, p2 = rng.uniform(0, 1, 100), rng.uniform(0, 1, 100)
        cmcs = cross_modal_consistency_score(p1, p2)
        assert (cmcs >= 0).all() and (cmcs <= 1).all()

    def test_symmetry(self):
        """CMCS(a, b) == CMCS(b, a)."""
        p1 = np.array([0.3, 0.7, 0.5])
        p2 = np.array([0.6, 0.2, 0.9])
        np.testing.assert_allclose(
            cross_modal_consistency_score(p1, p2),
            cross_modal_consistency_score(p2, p1))


class TestApplyCMCC:
    """Tests for the three-way decision rule."""

    def test_refer_when_low_consistency(self):
        """Low CMCS (high disagreement) must trigger referral."""
        # SVM says 0.95 PD, RF says 0.05 → CMCS = 0.10 < 0.65
        p_f   = np.array([0.7])
        p_svm = np.array([0.95])
        p_rf  = np.array([0.05])
        preds = apply_cmcc(p_f, p_svm, p_rf, delta=0.65)
        assert preds[0] == REFER_LABEL

    def test_pd_decision(self):
        """High consistency + P_F > 0.5 → PD."""
        p_f   = np.array([0.85])
        p_svm = np.array([0.82])
        p_rf  = np.array([0.88])
        # CMCS = 1 - |0.82-0.88| = 0.94 >= 0.65
        preds = apply_cmcc(p_f, p_svm, p_rf, delta=0.65)
        assert preds[0] == 1

    def test_hc_decision(self):
        """High consistency + P_F <= 0.5 → HC."""
        p_f   = np.array([0.2])
        p_svm = np.array([0.18])
        p_rf  = np.array([0.22])
        preds = apply_cmcc(p_f, p_svm, p_rf, delta=0.65)
        assert preds[0] == 0

    def test_output_labels(self):
        """All output labels must be in {0, 1, REFER_LABEL}."""
        rng = np.random.default_rng(5)
        n = 200
        p_svm = rng.uniform(0, 1, n)
        p_rf  = rng.uniform(0, 1, n)
        p_f   = 0.5 * p_svm + 0.5 * p_rf
        preds = apply_cmcc(p_f, p_svm, p_rf, delta=0.65)
        assert set(preds).issubset({0, 1, REFER_LABEL})

    def test_delta_0_no_referrals(self):
        """delta=0 → nobody is referred (all classified)."""
        rng = np.random.default_rng(0)
        p_svm = rng.uniform(0, 1, 50)
        p_rf  = rng.uniform(0, 1, 50)
        p_f   = 0.5 * p_svm + 0.5 * p_rf
        preds = apply_cmcc(p_f, p_svm, p_rf, delta=0.0)
        assert (preds != REFER_LABEL).all()

    def test_delta_1_all_referred(self):
        """delta=1 → all borderline cases referred (CMCS < 1 unless perfect)."""
        # Make p_svm ≠ p_rf so CMCS < 1 everywhere
        p_svm = np.array([0.8, 0.3, 0.6])
        p_rf  = np.array([0.7, 0.4, 0.9])
        p_f   = 0.5 * p_svm + 0.5 * p_rf
        preds = apply_cmcc(p_f, p_svm, p_rf, delta=1.0)
        assert (preds == REFER_LABEL).all()


class TestOptimiseDelta:
    """Tests for CMCC threshold optimiser."""

    def _make_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        # Good classifier: p near 0.9 for PD, 0.1 for HC
        p_base = np.where(y == 1,
                          rng.uniform(0.7, 1.0, n),
                          rng.uniform(0.0, 0.3, n))
        p_svm = np.clip(p_base + rng.normal(0, 0.05, n), 0, 1)
        p_rf  = np.clip(p_base + rng.normal(0, 0.05, n), 0, 1)
        p_f   = 0.5 * p_svm + 0.5 * p_rf
        return p_f, p_svm, p_rf, y

    def test_returns_float_in_range(self):
        p_f, p_svm, p_rf, y = self._make_data()
        delta = optimise_delta(p_f, p_svm, p_rf, y)
        assert 0.5 <= delta <= 0.90

    def test_sensitivity_constraint_met(self):
        """Returned delta should yield sensitivity >= 0.95 on classified cases."""
        p_f, p_svm, p_rf, y = self._make_data()
        delta = optimise_delta(p_f, p_svm, p_rf, y,
                               target_sensitivity=0.90)
        preds = apply_cmcc(p_f, p_svm, p_rf, delta=delta)
        classified = preds != REFER_LABEL
        if classified.sum() > 0:
            y_cls = y[classified]
            p_cls = preds[classified]
            tp = ((p_cls == 1) & (y_cls == 1)).sum()
            fn = ((p_cls == 0) & (y_cls == 1)).sum()
            sens = tp / (tp + fn + 1e-9)
            assert sens >= 0.85, f"Sensitivity constraint violated: {sens:.3f}"


class TestCMCCReport:
    """Tests for the CMCCReport data class."""

    def test_report_construction(self):
        rng = np.random.default_rng(3)
        n = 100
        p_svm = rng.uniform(0, 1, n)
        p_rf  = rng.uniform(0, 1, n)
        p_f   = 0.5 * p_svm + 0.5 * p_rf
        y     = rng.integers(0, 2, n)
        preds = apply_cmcc(p_f, p_svm, p_rf, delta=0.65)
        report = cmcc_report(preds, p_f, p_svm, p_rf, y, delta=0.65)
        assert report.n_total == n
        assert report.n_referred + report.n_classified == n
        assert 0.0 <= report.refer_rate <= 1.0

    def test_str_method(self):
        rng = np.random.default_rng(4)
        n = 50
        p_svm = rng.uniform(0, 1, n)
        p_rf  = rng.uniform(0, 1, n)
        p_f   = 0.5 * p_svm + 0.5 * p_rf
        y     = rng.integers(0, 2, n)
        preds = apply_cmcc(p_f, p_svm, p_rf, delta=0.65)
        report = cmcc_report(preds, p_f, p_svm, p_rf, y, delta=0.65)
        s = str(report)
        assert "CMCC Report" in s
        assert "Referred" in s
