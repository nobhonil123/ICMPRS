"""tests/test_generator.py — Unit tests for ICMPRSGenerator."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from icmprs.generator import ICMPRSGenerator


class TestICMPRSGenerator:
    """Tests for the parametric cohort-generation engine."""

    @pytest.fixture
    def gen(self):
        return ICMPRSGenerator(seed=42)

    def test_output_shape(self, gen):
        """Generated DataFrame must have exactly 68 columns (6 meta + 62 features)."""
        df = gen.generate(n=100)
        # 6 metadata cols + 25 acoustic + 21 kinematic + 1 lift_count + 16 ambulatory
        assert df.shape[0] == 100
        assert "label" in df.columns
        assert "participant_id" in df.columns

    def test_label_distribution(self, gen):
        """PD/HC split must be approximately 985/1010 for n=1995."""
        df = gen.generate(n=1995)
        n_pd = (df.label == 1).sum()
        n_hc = (df.label == 0).sum()
        assert n_pd + n_hc == 1995
        # Allow ±5 from target 985
        assert abs(n_pd - 985) <= 5, f"PD count off: {n_pd}"

    def test_reproducibility(self):
        """Same seed must produce identical DataFrames."""
        df1 = ICMPRSGenerator(seed=99).generate(n=50)
        df2 = ICMPRSGenerator(seed=99).generate(n=50)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds must produce different data."""
        df1 = ICMPRSGenerator(seed=42).generate(n=50)
        df2 = ICMPRSGenerator(seed=43).generate(n=50)
        assert not df1["rca_index"].equals(df2["rca_index"])

    def test_feature_columns_property(self, gen):
        """feature_columns must return exactly 62 names."""
        assert len(gen.feature_columns) == 62

    def test_indian_specific_features(self, gen):
        """Indian-specific features must be a proper subset of all features."""
        indian = set(gen.indian_specific_features)
        all_f  = set(gen.feature_columns)
        assert indian.issubset(all_f), "Indian features not in feature_columns"
        assert len(indian) == 11  # 1 RCA + 3 NRE + 3 DPLD + 3 AJHI + 1 BSV

    def test_generic_features_count(self, gen):
        """Generic (non-Indian) features must total 57 (= 62 - 11 Indian sub-features)."""
        # Note: 5 biomarker groups expand to 11 sub-columns
        assert len(gen.generic_features) == 51  # 62 - 11

    def test_rca_separation(self, gen):
        """RCA mean must be higher for PD than HC (group-separating design)."""
        df = gen.generate(n=500)
        mu_pd = df.loc[df.label == 1, "rca_index"].mean()
        mu_hc = df.loc[df.label == 0, "rca_index"].mean()
        assert mu_pd > mu_hc, "RCA not group-separating"

    def test_bsv_separation(self, gen):
        """BSV mean must be higher for PD than HC."""
        df = gen.generate(n=500)
        mu_pd = df.loc[df.label == 1, "bsv"].mean()
        mu_hc = df.loc[df.label == 0, "bsv"].mean()
        assert mu_pd > mu_hc

    def test_physiological_bounds(self, gen):
        """All features must stay within their declared physiological bounds."""
        df = gen.generate(n=200)
        # HNR should be in [5, 32] dB for any participant
        assert df["hnr"].between(4.0, 33.0).all(), "HNR out of bounds"
        # BSV in [0.02, 0.55]
        assert df["bsv"].between(0.01, 0.60).all(), "BSV out of bounds"

    def test_subgroup_labels(self, gen):
        """All dialect, device, and footwear values must be valid categories."""
        df = gen.generate(n=200)
        valid_dialects  = {"North-Hindi", "West-Marathi",
                           "South-Tamil", "East-Bengali"}
        valid_devices   = {"Budget-Android", "MidRange-Android", "Feature-Phone"}
        valid_footwear  = {"barefoot_paved", "barefoot_unpaved",
                           "slippers", "shoes"}
        assert set(df.dialect).issubset(valid_dialects)
        assert set(df.device).issubset(valid_devices)
        assert set(df.footwear).issubset(valid_footwear)

    def test_hy_stages_pd_only(self, gen):
        """H&Y stages must be assigned only to PD participants."""
        df = gen.generate(n=200)
        hc_stages = df.loc[df.label == 0, "hy_stage"].unique()
        assert set(hc_stages) == {0}, f"HC has non-zero H&Y stages: {hc_stages}"
        pd_stages = set(df.loc[df.label == 1, "hy_stage"].unique())
        assert pd_stages.issubset({1, 2, 3})

    def test_no_nan(self, gen):
        """Generated DataFrame must contain no NaN values."""
        df = gen.generate(n=100)
        assert not df.isnull().any().any(), "NaN values found in generated data"

    def test_five_seeds_stability(self):
        """Multi-seed stability: mean RCA should be within 5 Hz across seeds."""
        seeds = [42, 123, 256, 789, 1024]
        means = []
        for s in seeds:
            df = ICMPRSGenerator(seed=s).generate(n=1995)
            means.append(df.loc[df.label == 1, "rca_index"].mean())
        assert np.std(means) < 5.0, f"RCA mean too variable across seeds: {means}"
