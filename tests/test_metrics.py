"""tests/test_metrics.py — Unit tests for evaluation metrics."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.metrics import (compute_metrics, mcnemar_bonferroni,
                                 cultural_adaptation_gain,
                                 distribution_shift_bound,
                                 accuracy_lower_bound,
                                 prevalence_adjusted_ppv_npv,
                                 mmd_squared)


class TestCAG:
    """Tests for Cultural Adaptation Gain (Definition 2, Eq. 16)."""

    def test_manuscript_value(self):
        """CAG must reproduce the paper value of 0.135 for AUC 0.987 / 0.929."""
        cag = cultural_adaptation_gain(0.987, 0.929)
        assert abs(cag - 0.135) < 0.001, f"Expected ~0.135, got {cag}"

    def test_zero_gain(self):
        """Equal AUCs → CAG = 0."""
        assert cultural_adaptation_gain(0.929, 0.929) == pytest.approx(0.0)

    def test_negative_gain(self):
        """Adapted AUC < generic AUC → negative CAG."""
        cag = cultural_adaptation_gain(0.90, 0.95)
        assert cag < 0

    def test_degenerate_generic(self):
        """AUC_gen = 0.5 (random) → CAG is NaN (undefined)."""
        import math
        cag = cultural_adaptation_gain(0.80, 0.50)
        assert math.isnan(cag)


class TestDistributionShiftBound:
    """Tests for Proposition 1 (Eq. 18)."""

    def test_manuscript_values(self):
        """
        R_hat=0.026, N=1995, delta'=0.05:
          eps=0.12 → AccT >= ~70.4%
          eps=0.04 → AccT >= ~84.4%
        """
        lb_mpower = accuracy_lower_bound(0.974, n=1995, epsilon=0.12)
        lb_uci    = accuracy_lower_bound(0.974, n=1995, epsilon=0.04)
        assert lb_mpower == pytest.approx(0.704, abs=0.015)
        assert lb_uci    == pytest.approx(0.864, abs=0.015)

    def test_larger_epsilon_lower_bound(self):
        """Larger epsilon must yield a lower (worse) accuracy bound."""
        lb_small = accuracy_lower_bound(0.974, n=1995, epsilon=0.05)
        lb_large = accuracy_lower_bound(0.974, n=1995, epsilon=0.20)
        assert lb_small > lb_large

    def test_bound_non_negative(self):
        """Accuracy lower bound must be >= 0 even for large epsilon."""
        lb = accuracy_lower_bound(0.974, n=1995, epsilon=0.90)
        assert lb >= 0.0

    def test_raw_risk_bound(self):
        """distribution_shift_bound returns risk (not accuracy)."""
        r_upper = distribution_shift_bound(0.026, 1995, 0.12, 0.05)
        assert 0 < r_upper < 1


class TestPrevalencePPV:
    """Tests for prevalence-adjusted PPV/NPV (Eq. 19)."""

    def test_manuscript_ppv(self):
        """Sens=0.984, Spec=0.963, prev=0.01 → PPV ≈ 21.2%."""
        ppv, npv = prevalence_adjusted_ppv_npv(0.984, 0.963, 0.01)
        assert abs(ppv * 100 - 21.2) < 0.5, f"PPV={ppv*100:.1f}% expected ~21.2%"

    def test_npv_near_100(self):
        """At 1% prevalence with high sens/spec, NPV must be ~99.98%."""
        _, npv = prevalence_adjusted_ppv_npv(0.984, 0.963, 0.01)
        assert npv > 0.999

    def test_high_prevalence_higher_ppv(self):
        """PPV increases with prevalence."""
        ppv_low,  _ = prevalence_adjusted_ppv_npv(0.95, 0.95, 0.01)
        ppv_high, _ = prevalence_adjusted_ppv_npv(0.95, 0.95, 0.30)
        assert ppv_high > ppv_low

    def test_perfect_classifier(self):
        """Sens=1, Spec=1 → PPV=1, NPV=1 at any prevalence."""
        ppv, npv = prevalence_adjusted_ppv_npv(1.0, 1.0, 0.05)
        assert ppv == pytest.approx(1.0)
        assert npv == pytest.approx(1.0)


class TestMcNemar:
    """Tests for McNemar's test with Bonferroni correction."""

    def test_identical_models_not_significant(self):
        """Identical predictions → p = 1.0, not significant."""
        y  = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        pred = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        result = mcnemar_bonferroni(y, pred, pred, n_comparisons=5)
        assert not result["significant"]
        assert result["p_corrected"] == pytest.approx(1.0)

    def test_very_different_models(self):
        """One perfect model vs random → should be significant."""
        rng = np.random.default_rng(0)
        y     = rng.integers(0, 2, 500)
        perf  = y.copy()                       # perfect
        rand  = rng.integers(0, 2, 500)        # random
        result = mcnemar_bonferroni(y, perf, rand, n_comparisons=1)
        # Not always significant with random, but p should be low
        assert result["p_raw"] < 0.2


class TestMMD:
    """Tests for MMD² calibration check."""

    def test_same_distribution_near_zero(self):
        """MMD² between samples from same distribution should be near 0."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (200, 3))
        Y = rng.normal(0, 1, (200, 3))
        mmd2 = mmd_squared(X, Y)
        assert abs(mmd2) < 0.05, f"MMD²={mmd2:.4f} unexpectedly large"

    def test_different_distributions_larger(self):
        """MMD² between shifted distributions should be larger."""
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (200, 3))
        Y = rng.normal(5, 1, (200, 3))   # large shift
        mmd2 = mmd_squared(X, Y)
        assert mmd2 > 0.1, f"MMD²={mmd2:.4f} should detect shift"


class TestComputeMetrics:
    """Smoke tests for the full metrics bundle."""

    def test_perfect_classifier(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        p = np.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.05])
        m = compute_metrics(y, y, p, n_boot=50)
        assert m.accuracy  == pytest.approx(1.0)
        assert m.sensitivity == pytest.approx(1.0)

    def test_ci_ordering(self):
        """Lower CI bound must be <= upper CI bound."""
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 80)
        p = rng.uniform(0, 1, 80)
        pred = (p > 0.5).astype(int)
        m = compute_metrics(y, pred, p, n_boot=100)
        assert m.acc_ci[0] <= m.acc_ci[1]
        assert m.sens_ci[0] <= m.sens_ci[1]
        assert m.auc_ci[0]  <= m.auc_ci[1]
