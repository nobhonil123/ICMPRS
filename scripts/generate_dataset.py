#!/usr/bin/env python3
"""
scripts/generate_dataset.py
============================
Generate the ICMPRS synthetic cohort for all five seeds used in the
multi-seed stability analysis (Section III-H, manuscript).

Usage:
    python scripts/generate_dataset.py [--out_dir data/]
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icmprs.generator import ICMPRSGenerator

SEEDS = [42, 123, 256, 789, 1024]
N_PARTICIPANTS = 1995

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ICMPRS synthetic cohorts for all five seeds.")
    parser.add_argument("--out_dir", default="data",
                        help="Directory to write CSV files (default: data/)")
    parser.add_argument("--n", type=int, default=N_PARTICIPANTS,
                        help=f"Participants per cohort (default: {N_PARTICIPANTS})")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        logger.info("Generating cohort for seed=%d ...", seed)
        gen = ICMPRSGenerator(seed=seed)
        df = gen.generate(n=args.n)
        out_path = out_dir / f"icmprs_seed{seed}.csv"
        df.to_csv(out_path, index=False)
        logger.info("  Saved: %s  (shape %s)", out_path, df.shape)

    # Also save primary seed to a canonical filename
    logger.info("Primary cohort (seed=42) also saved as data/icmprs_features.csv")
    import shutil
    shutil.copy(out_dir / "icmprs_seed42.csv",
                out_dir / "icmprs_features.csv")
    logger.info("Done.")


if __name__ == "__main__":
    main()
