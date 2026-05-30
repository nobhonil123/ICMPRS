"""tests/test_acg.py — Unit tests for Adaptive Confidence Gating."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgms.acg import (compute_sqi_voice, compute_sqi_movement,
                      fuse_probabilities, simulate_quality,
                      ACGOptimiser)


RNG = np.random.default_rng(0)


class TestSQI:
    """Tests for Signal Quality Index computation."""

    def test_sqi_range(self):
        """SQI must always be in (0, 1)."""
        alpha = np.array([0.38, 0.29, 0.33])
        snr = np.linspace(-5, 30, 50)
        vr  = np.ones(50) * 0.9
        cr  = np.ones(50) * 0.02
        sqi = compute_sqi_voice(snr, vr, cr, alpha)
        assert (sqi > 0).all() and (sqi < 1).all()

    def test_higher_snr_higher_sqi(self):
        """Higher SNR must produce higher SQI_voice."""
        alpha = np.array([0.38, 0.29, 0.33])
        vr = np.array([0.9, 0.9])
        cr = np.array([0.01, 0.01])
        s_low  = compute_sqi_voice(np.array([0.0]),  vr[:1], cr[:1], alpha)
        s_high = compute_sqi_voice(np.array([20.0]), vr[:1], cr[:1], alpha)
        assert s_high > s_low

    def test_noisy_voice_down_weighted(self):
        """At 0 dB SNR the voice weight must be < 0.5."""
        alpha = np.array([0.38, 0.29, 0.33])
        beta  = np.array([0.31, 0.42, 0.27])
        n = 10
        # Low SNR for voice
        sqi_v = compute_sqi_voice(np.zeros(n), np.ones(n)*0.5,
                                  np.ones(n)*0.1, alpha)
        sqi_m = compute_sqi_movement(np.ones(n)*0.8, np.ones(n)*0.9,
                                     np.zeros(n), beta)
        w1, w2, _ = fuse_probabilities(np.ones(n)*0.5, np.ones(n)*0.5,
                                        sqi_v, sqi_m)
        assert (w1 < 0.5).all(), "Voice should be down-weighted at 0 dB"

    def test_equal_quality_equal_weights(self):
        """Equal quality must yield w1 ≈ w2 ≈ 0.5."""
        alpha = np.array([1.0, 1.0, 1.0])
        beta  = np.array([1.0, 1.0, 1.0])
        n = 5
        # Identical inputs for both modalities
        sqi_v = compute_sqi_voice(np.ones(n), np.ones(n), np.ones(n), alpha)
        sqi_m = compute_sqi_movement(np.ones(n), np.ones(n), np.ones(n), beta)
        w1, w2, _ = fuse_probabilities(np.ones(n)*0.5, np.ones(n)*0.5,
                                        sqi_v, sqi_m)
        np.testing.assert_allclose(w1, 0.5, atol=0.01)
        np.testing.assert_allclose(w2, 0.5, atol=0.01)

    def test_weights_sum_to_one(self):
        """ACG weights must always sum to 1."""
        alpha = np.array([0.38, 0.29, 0.33])
        beta  = np.array([0.31, 0.42, 0.27])
        qv, qm = simulate_quality(50, snr_db=10.0, rng=RNG)
        sqi_v = compute_sqi_voice(qv[:,0], qv[:,1], qv[:,2], alpha)
        sqi_m = compute_sqi_movement(qm[:,0], qm[:,1], qm[:,2], beta)
        w1, w2, _ = fuse_probabilities(np.ones(50)*0.5, np.ones(50)*0.5,
                                        sqi_v, sqi_m)
        np.testing.assert_allclose(w1 + w2, np.ones(50), atol=1e-6)


class TestFusion:
    """Tests for probability fusion."""

    def test_fused_probability_in_range(self):
        """Fused probability must be in [0, 1]."""
        rng = np.random.default_rng(1)
        p_svm = rng.uniform(0, 1, 100)
        p_rf  = rng.uniform(0, 1, 100)
        qv, qm = simulate_quality(100, rng=rng)
        alpha = np.array([0.38, 0.29, 0.33])
        beta  = np.array([0.31, 0.42, 0.27])
        sqi_v = compute_sqi_voice(qv[:,0], qv[:,1], qv[:,2], alpha)
        sqi_m = compute_sqi_movement(qm[:,0], qm[:,1], qm[:,2], beta)
        _, _, pf = fuse_probabilities(p_svm, p_rf, sqi_v, sqi_m)
        assert (pf >= 0).all() and (pf <= 1).all()

    def test_fused_is_weighted_average(self):
        """Fused probability must equal w1*p_svm + w2*p_rf exactly."""
        p_svm = np.array([0.8])
        p_rf  = np.array([0.6])
        sqi_v = np.array([0.7])
        sqi_m = np.array([0.3])
        w1, w2, pf = fuse_probabilities(p_svm, p_rf, sqi_v, sqi_m)
        expected = w1 * p_svm + w2 * p_rf
        np.testing.assert_allclose(pf, expected, atol=1e-10)


class TestSimulateQuality:
    """Tests for quality proxy simulation."""

    def test_output_shape(self):
        qv, qm = simulate_quality(30, rng=RNG)
        assert qv.shape == (30, 3)
        assert qm.shape == (30, 3)

    def test_snr_effect(self):
        """Lower SNR should produce lower mean SQI_voice."""
        alpha = np.array([0.38, 0.29, 0.33])
        rng = np.random.default_rng(7)
        qv_high, _ = simulate_quality(200, snr_db=20.0, rng=rng)
        rng2 = np.random.default_rng(7)
        qv_low,  _ = simulate_quality(200, snr_db=0.0,  rng=rng2)
        sqi_high = compute_sqi_voice(qv_high[:,0], qv_high[:,1],
                                     qv_high[:,2], alpha).mean()
        sqi_low  = compute_sqi_voice(qv_low[:,0],  qv_low[:,1],
                                     qv_low[:,2],  alpha).mean()
        assert sqi_high > sqi_low
