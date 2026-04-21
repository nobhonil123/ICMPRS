#!/usr/bin/env python3
"""
ICMPRS v2 — Generalisability Bound (PAC-style)
===============================================
Implements Theorem 1 from the paper: distribution-shift degradation bound.

R_T ≤ R̂_S + 2ε + sqrt(ln(2/δ') / 2N)

Author : Nobhonil Roy Choudhury
Licence: MIT
"""

import numpy as np


def generalisation_bound(empirical_error, n_samples, epsilon, delta_prime=0.05):
    """
    Compute the worst-case real-world error under distribution shift.

    Parameters
    ----------
    empirical_error : float
        Error on synthetic data (1 - accuracy).
    n_samples : int
        Number of training/test samples.
    epsilon : float
        Total variation distance bound between synthetic and real distributions.
    delta_prime : float
        Confidence parameter (default 0.05 for 95% confidence).

    Returns
    -------
    dict with bound components and worst-case accuracy.
    """
    finite_sample = np.sqrt(np.log(2.0 / delta_prime) / (2 * n_samples))
    worst_error = empirical_error + 2 * epsilon + finite_sample
    worst_accuracy = max(0, 1 - worst_error)

    return {
        "empirical_error": empirical_error,
        "epsilon": epsilon,
        "finite_sample_correction": finite_sample,
        "worst_case_error": worst_error,
        "worst_case_accuracy": worst_accuracy,
        "confidence": 1 - delta_prime,
    }


def main():
    print("=" * 60)
    print("  GENERALISABILITY BOUND — Theorem 1")
    print("=" * 60)

    # Paper parameters
    empirical_accuracy = 0.962
    empirical_error = 1 - empirical_accuracy
    n = 1995

    epsilons = [0.05, 0.10, 0.15, 0.20]

    print(f"\n  Empirical error (synthetic): {empirical_error:.3f}")
    print(f"  N = {n}")
    print(f"  Confidence: 95%\n")
    print(f"  {'ε (TV dist)':<15} {'Worst Error':<15} {'Worst Acc.':<15}")
    print(f"  {'-'*45}")

    for eps in epsilons:
        result = generalisation_bound(empirical_error, n, eps)
        print(f"  {eps:<15.2f} {result['worst_case_error']:<15.3f} "
              f"{100*result['worst_case_accuracy']:<15.1f}%")

    print(f"\n  Interpretation:")
    r_conservative = generalisation_bound(empirical_error, n, 0.10)
    print(f"    At ε = 0.10 (conservative): ≥ "
          f"{100*r_conservative['worst_case_accuracy']:.1f}% accuracy")
    r_moderate = generalisation_bound(empirical_error, n, 0.05)
    print(f"    At ε = 0.05 (moderate):      ≥ "
          f"{100*r_moderate['worst_case_accuracy']:.1f}% accuracy")
    print(f"\n    Even worst-case guarantees exceed chance (50%).")
    print("=" * 60)


if __name__ == "__main__":
    main()
