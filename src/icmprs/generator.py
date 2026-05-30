"""
icmprs/generator.py
===================
Parametric cohort-generation engine for the Indian Context Multimodal
Parkinson's Reference Standard (ICMPRS).

Generates synthetic voice, handwriting, and gait feature vectors whose
statistical profiles are anchored to published Indian clinical measurements.
See Section III of the accompanying manuscript for full parameterisation.

Usage (CLI)
-----------
    python -m icmprs.generator --n 1995 --seed 42 --out data/icmprs_features.csv

Usage (API)
-----------
    from icmprs.generator import ICMPRSGenerator
    gen = ICMPRSGenerator(seed=42)
    df = gen.generate(n=1995)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _tn(mu: float, sigma: float, lo: float, hi: float,
        rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample from a truncated normal TN(mu, sigma^2, [lo, hi])."""
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size,
                         random_state=rng.integers(0, 2**31 - 1))


def _poisson(lam: float, rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample from a Poisson distribution."""
    return rng.poisson(lam=lam, size=size).astype(float)


# ---------------------------------------------------------------------------
# Distributional parameters (Tables III-D through III-F, manuscript)
# PD = Parkinson's Disease; HC = Healthy Control
# ---------------------------------------------------------------------------

# ── Acoustic (25 features) ──────────────────────────────────────────────────
ACOUSTIC_PARAMS = {
    # Standard dysphonia features (21) – anchored to Little 2009, Tsanas 2012, Harel 2004
    "jitter_local":     dict(pd=(0.028, 0.008, 0.005, 0.080), hc=(0.005, 0.002, 0.001, 0.020)),
    "jitter_abs":       dict(pd=(0.00004, 0.00001, 0.000005, 0.0002), hc=(0.000012, 0.000005, 0.000001, 0.00005)),
    "jitter_rap":       dict(pd=(0.015, 0.005, 0.002, 0.06), hc=(0.003, 0.001, 0.0005, 0.012)),
    "jitter_ppq5":      dict(pd=(0.016, 0.005, 0.002, 0.07), hc=(0.003, 0.001, 0.0005, 0.013)),
    "jitter_ddp":       dict(pd=(0.045, 0.015, 0.006, 0.18), hc=(0.009, 0.003, 0.001, 0.036)),
    "shimmer_local":    dict(pd=(0.174, 0.045, 0.040, 0.400), hc=(0.048, 0.015, 0.010, 0.120)),
    "shimmer_local_db": dict(pd=(1.820, 0.480, 0.400, 4.200), hc=(0.490, 0.155, 0.100, 1.200)),
    "shimmer_apq3":     dict(pd=(0.100, 0.028, 0.020, 0.250), hc=(0.026, 0.008, 0.005, 0.070)),
    "shimmer_apq5":     dict(pd=(0.112, 0.031, 0.022, 0.280), hc=(0.029, 0.009, 0.006, 0.080)),
    "shimmer_apq11":    dict(pd=(0.155, 0.040, 0.030, 0.380), hc=(0.042, 0.012, 0.008, 0.110)),
    "shimmer_dda":      dict(pd=(0.301, 0.083, 0.060, 0.750), hc=(0.079, 0.025, 0.015, 0.210)),
    "hnr":              dict(pd=(13.6, 3.2, 5.0, 25.0), hc=(22.9, 2.8, 15.0, 32.0)),
    "rpde":             dict(pd=(0.565, 0.082, 0.300, 0.850), hc=(0.430, 0.065, 0.200, 0.650)),
    "dfa":              dict(pd=(0.718, 0.068, 0.500, 0.950), hc=(0.650, 0.055, 0.430, 0.850)),
    "spread1":          dict(pd=(-5.60, 0.95, -9.00, -2.50), hc=(-7.25, 0.82, -10.00, -4.00)),
    "spread2":          dict(pd=(0.278, 0.062, 0.080, 0.500), hc=(0.168, 0.042, 0.040, 0.360)),
    "d2":               dict(pd=(2.50, 0.40, 1.20, 4.00), hc=(2.00, 0.32, 0.90, 3.40)),
    "ppe":              dict(pd=(0.258, 0.060, 0.060, 0.500), hc=(0.110, 0.035, 0.020, 0.260)),
    "nhr":              dict(pd=(0.032, 0.012, 0.005, 0.100), hc=(0.010, 0.004, 0.001, 0.035)),
    "fundamental_freq": dict(pd=(145.0, 28.0, 80.0, 260.0), hc=(175.0, 32.0, 90.0, 300.0)),
    "voiced_fraction":  dict(pd=(0.820, 0.065, 0.600, 0.980), hc=(0.912, 0.042, 0.750, 0.990)),
    # Indian-specific features (4)
    # Feature 1: RCA Degradation Index (Eq. 3, manuscript)
    "rca_index":        dict(pd=(168.0, 42.0, 80.0, 280.0), hc=(38.0, 16.0, 10.0, 90.0)),
    # Feature 2: NRE — three sub-features
    "nre_ner":          dict(pd=(0.41, 0.09, 0.20, 0.65), hc=(0.22, 0.05, 0.10, 0.38)),
    "nre_vli":          dict(pd=(0.18, 0.06, 0.04, 0.35), hc=(0.05, 0.02, 0.01, 0.12)),
    "nre_nfb":          dict(pd=(285.0, 62.0, 150.0, 450.0), hc=(165.0, 38.0, 80.0, 280.0)),
}

# ── Kinematic (21 features) ─────────────────────────────────────────────────
KINEMATIC_PARAMS = {
    # Standard handwriting features (15) – anchored to Drotar 2016, Impedovo 2019
    "pen_velocity_mean":  dict(pd=(8.2, 2.5, 2.0, 20.0), hc=(18.5, 3.8, 8.0, 32.0)),
    "pen_velocity_std":   dict(pd=(3.8, 1.1, 0.8, 9.0),  hc=(2.1, 0.7, 0.5, 5.0)),
    "pen_pressure_mean":  dict(pd=(1.65, 0.32, 0.80, 2.80), hc=(1.10, 0.22, 0.55, 1.90)),
    "pen_pressure_std":   dict(pd=(0.42, 0.12, 0.10, 0.90), hc=(0.18, 0.06, 0.04, 0.40)),
    "stroke_duration":    dict(pd=(520.0, 120.0, 200.0, 950.0), hc=(280.0, 72.0, 100.0, 500.0)),
    "stroke_length":      dict(pd=(3.2, 0.9, 1.0, 7.0),   hc=(5.8, 1.2, 2.5, 10.0)),
    "num_strokes":        dict(pd=(22.0, 5.5, 8.0, 45.0),  hc=(18.0, 4.2, 6.0, 35.0)),
    "in_air_time":        dict(pd=(380.0, 95.0, 120.0, 700.0), hc=(180.0, 55.0, 50.0, 380.0)),
    "on_paper_time":      dict(pd=(2400.0, 480.0, 1000.0, 4200.0), hc=(1600.0, 340.0, 700.0, 3000.0)),
    "time_ratio":         dict(pd=(0.38, 0.09, 0.15, 0.65), hc=(0.20, 0.06, 0.08, 0.42)),
    "jerk_mean":          dict(pd=(0.82, 0.22, 0.25, 1.60), hc=(0.38, 0.12, 0.10, 0.80)),
    "smoothness":         dict(pd=(0.42, 0.11, 0.10, 0.80), hc=(0.72, 0.10, 0.40, 0.95)),
    "axial_force":        dict(pd=(0.68, 0.18, 0.20, 1.30), hc=(0.45, 0.12, 0.12, 0.90)),
    "velocity_entropy":   dict(pd=(3.85, 0.65, 2.00, 5.50), hc=(2.60, 0.48, 1.20, 4.20)),
    "pressure_entropy":   dict(pd=(3.12, 0.58, 1.50, 4.80), hc=(2.15, 0.42, 0.90, 3.60)),
    # Feature 3: DPLD — three sub-features (Eq. body, manuscript)
    "dpld_lift_duration":      dict(pd=(724.0, 165.0, 350.0, 1200.0), hc=(210.0, 68.0, 80.0, 450.0)),
    "dpld_restart_velocity":   dict(pd=(6.5, 2.4, 2.0, 14.0), hc=(17.2, 3.8, 8.0, 28.0)),
    # Feature 4: AJHI — three sub-features
    "ajhi_transition_time":    dict(pd=(412.0, 95.0, 200.0, 680.0), hc=(185.0, 48.0, 80.0, 320.0)),
    "ajhi_pressure_spike":     dict(pd=(1.82, 0.35, 1.10, 2.80), hc=(1.12, 0.18, 0.80, 1.60)),
    "ajhi_pause_ratio":        dict(pd=(2.45, 0.58, 1.20, 4.00), hc=(1.18, 0.22, 0.80, 1.80)),
}

# Pen-lift count is Poisson; handled separately
DPLD_LIFT_COUNT_LAMBDA = dict(pd=9.5, hc=4.2)

# ── Ambulatory (16 features) ─────────────────────────────────────────────────
AMBULATORY_PARAMS = {
    # Standard gait features (15) – anchored to Wahid 2015, Zeng 2016, Hollman 2011
    "stride_length":      dict(pd=(0.68, 0.12, 0.25, 1.10), hc=(1.09, 0.14, 0.60, 1.50)),
    "stride_speed":       dict(pd=(0.62, 0.14, 0.20, 1.10), hc=(1.10, 0.16, 0.55, 1.60)),
    "cadence":            dict(pd=(94.0, 12.0, 55.0, 130.0), hc=(112.0, 10.0, 80.0, 145.0)),
    "step_length_asym":   dict(pd=(0.12, 0.04, 0.02, 0.28), hc=(0.04, 0.02, 0.00, 0.12)),
    "swing_time":         dict(pd=(0.42, 0.08, 0.22, 0.65), hc=(0.38, 0.06, 0.22, 0.58)),
    "stance_time":        dict(pd=(0.68, 0.10, 0.42, 0.98), hc=(0.62, 0.08, 0.40, 0.88)),
    "double_support":     dict(pd=(0.32, 0.08, 0.12, 0.58), hc=(0.22, 0.05, 0.08, 0.40)),
    "stride_time_cv":     dict(pd=(0.085, 0.025, 0.020, 0.180), hc=(0.032, 0.012, 0.008, 0.080)),
    "step_width":         dict(pd=(0.12, 0.03, 0.04, 0.24), hc=(0.10, 0.025, 0.03, 0.20)),
    "trunk_accel_rms":    dict(pd=(0.38, 0.10, 0.12, 0.72), hc=(0.22, 0.07, 0.06, 0.48)),
    "gait_regularity":    dict(pd=(0.62, 0.12, 0.25, 0.88), hc=(0.82, 0.08, 0.55, 0.98)),
    "freezing_index":     dict(pd=(0.28, 0.09, 0.05, 0.60), hc=(0.08, 0.03, 0.01, 0.18)),
    "turn_duration":      dict(pd=(2.85, 0.65, 1.20, 5.00), hc=(1.65, 0.42, 0.70, 3.20)),
    "step_count_var":     dict(pd=(0.18, 0.05, 0.05, 0.38), hc=(0.08, 0.03, 0.01, 0.20)),
    "cadence_variability":dict(pd=(8.5, 2.5, 2.5, 18.0), hc=(3.8, 1.2, 1.0, 9.0)),
    # Feature 5: BSV — Barefoot Stride Variance (Eqs. 4-5, manuscript)
    "bsv":                dict(pd=(0.32, 0.09, 0.12, 0.55), hc=(0.08, 0.03, 0.02, 0.20)),
}

# Footwear distribution weights (used for subgroup labelling only)
FOOTWEAR_DIST = {
    "barefoot_paved":   0.206,
    "barefoot_unpaved": 0.193,
    "slippers":         0.300,
    "shoes":            0.301,
}

# Dialect region distribution
DIALECT_DIST = {
    "North-Hindi":   0.263,
    "West-Marathi":  0.246,
    "South-Tamil":   0.247,
    "East-Bengali":  0.244,
}

# Device tier distribution (ITU 2023)
DEVICE_DIST = {
    "Budget-Android":   0.422,
    "MidRange-Android": 0.312,
    "Feature-Phone":    0.266,
}

# H&Y stage distribution (among PD participants)
HY_DIST = {1: 0.33, 2: 0.34, 3: 0.33}


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class ICMPRSGenerator:
    """
    Parametric generator for the ICMPRS synthetic cohort.

    Parameters
    ----------
    seed : int
        Global random seed for reproducibility.
    pd_ratio : float
        Fraction of generated participants with PD (default 0.4940 ≈ 985/1995).
    """

    FEATURE_NAMES_ACOUSTIC = list(ACOUSTIC_PARAMS.keys())
    FEATURE_NAMES_KINEMATIC = list(KINEMATIC_PARAMS.keys()) + ["dpld_lift_count"]
    FEATURE_NAMES_AMBULATORY = list(AMBULATORY_PARAMS.keys())

    def __init__(self, seed: int = 42, pd_ratio: float = 985 / 1995):
        self.seed = seed
        self.pd_ratio = pd_ratio
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _sample_modality(self, params: dict, label: int,
                         n: int) -> dict[str, np.ndarray]:
        """Sample all features for one modality given a PD/HC label."""
        key = "pd" if label == 1 else "hc"
        out = {}
        for feat, p in params.items():
            mu, sigma, lo, hi = p[key]
            out[feat] = _tn(mu, sigma, lo, hi, self._rng, n)
        return out

    # ------------------------------------------------------------------
    def _assign_subgroups(self, n: int) -> dict[str, np.ndarray]:
        """Assign dialect, device tier, and footwear labels."""
        def _choice(dist: dict, size: int) -> np.ndarray:
            keys = list(dist.keys())
            probs = np.array(list(dist.values()))
            probs /= probs.sum()
            return self._rng.choice(keys, size=size, p=probs)

        return {
            "dialect":   _choice(DIALECT_DIST, n),
            "device":    _choice(DEVICE_DIST, n),
            "footwear":  _choice(FOOTWEAR_DIST, n),
        }

    # ------------------------------------------------------------------
    def _assign_hy_stage(self, n_pd: int) -> np.ndarray:
        """Assign Hoehn & Yahr stages to PD participants (generator-assigned)."""
        keys = list(HY_DIST.keys())
        probs = np.array(list(HY_DIST.values()))
        probs /= probs.sum()
        return self._rng.choice(keys, size=n_pd, p=probs)

    # ------------------------------------------------------------------
    def generate(self, n: int = 1995) -> pd.DataFrame:
        """
        Generate a synthetic ICMPRS cohort of n participants.

        Returns
        -------
        pd.DataFrame  with columns:
            participant_id, label, hy_stage, dialect, device, footwear,
            <25 acoustic features>, <21 kinematic features>,
            <16 ambulatory features>
        """
        n_pd = round(n * self.pd_ratio)
        n_hc = n - n_pd
        logger.info("Generating %d PD + %d HC = %d participants (seed=%d)",
                    n_pd, n_hc, n, self.seed)

        records: list[dict] = []
        for label, count in [(1, n_pd), (0, n_hc)]:
            acous = self._sample_modality(ACOUSTIC_PARAMS, label, count)
            kinem = self._sample_modality(KINEMATIC_PARAMS, label, count)
            ambul = self._sample_modality(AMBULATORY_PARAMS, label, count)

            # Poisson: pen-lift count
            lam = DPLD_LIFT_COUNT_LAMBDA["pd" if label else "hc"]
            lift_count = _poisson(lam, self._rng, count)

            subs = self._assign_subgroups(count)
            hy = (self._assign_hy_stage(count)
                  if label == 1
                  else np.zeros(count, dtype=int))

            for i in range(count):
                row: dict = {
                    "participant_id": None,   # assigned after concatenation
                    "label": label,
                    "hy_stage": int(hy[i]),
                    "dialect":  subs["dialect"][i],
                    "device":   subs["device"][i],
                    "footwear": subs["footwear"][i],
                }
                for k, v in acous.items():
                    row[k] = float(v[i])
                for k, v in kinem.items():
                    row[k] = float(v[i])
                row["dpld_lift_count"] = float(lift_count[i])
                for k, v in ambul.items():
                    row[k] = float(v[i])
                records.append(row)

        # Shuffle and assign IDs
        self._rng.shuffle(records)      # type: ignore[arg-type]
        df = pd.DataFrame(records)
        # Overwrite placeholder participant_id values with sequential IDs
        df["participant_id"] = [f"P{i:04d}" for i in range(1, len(df) + 1)]
        # Reorder so participant_id is first column
        cols = ["participant_id"] + [c for c in df.columns if c != "participant_id"]
        df = df[cols]
        logger.info("Generated cohort shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    @property
    def feature_columns(self) -> list[str]:
        """Return ordered list of all 62 feature column names."""
        return (
            self.FEATURE_NAMES_ACOUSTIC
            + self.FEATURE_NAMES_KINEMATIC
            + self.FEATURE_NAMES_AMBULATORY
        )

    @property
    def indian_specific_features(self) -> list[str]:
        """Return the 5 + 4 Indian-specific feature sub-columns."""
        return [
            "rca_index",
            "nre_ner", "nre_vli", "nre_nfb",
            "dpld_lift_count", "dpld_lift_duration", "dpld_restart_velocity",
            "ajhi_transition_time", "ajhi_pressure_spike", "ajhi_pause_ratio",
            "bsv",
        ]

    @property
    def generic_features(self) -> list[str]:
        """Return the 57 generic (non-Indian-specific) feature names."""
        indian = set(self.indian_specific_features)
        return [f for f in self.feature_columns if f not in indian]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_generate():
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate the ICMPRS synthetic cohort."
    )
    parser.add_argument("--n", type=int, default=1995,
                        help="Number of participants (default: 1995)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out", type=str,
                        default="data/icmprs_features.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    gen = ICMPRSGenerator(seed=args.seed)
    df = gen.generate(n=args.n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    cli_generate()
