"""Exploratory stats + per-neuron discriminative-power analysis.

n=12 (6 math / 6 others). With this few samples, p-values and CV scores are
illustrative, not statistically powerful -- treated as such throughout.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from analysis.load_data import load_samples, Sample
from analysis.features import pooled_stats


def per_sample_summary(samples: list[Sample]) -> None:
    print(f"{'name':10s} {'label':7s} {'tokens':6s} {'mean':>9s} {'std':>9s} "
          f"{'min':>9s} {'max':>9s} {'l2norm':>10s} {'sparsity':>9s}")
    for s in samples:
        m = s.matrix
        sparsity = (np.abs(m) < 1e-3).mean()  # fraction near-zero (GELU floor)
        print(f"{s.name:10s} {s.label:7s} {m.shape[0]:<6d} {m.mean():9.4f} {m.std():9.4f} "
              f"{m.min():9.4f} {m.max():9.4f} {np.linalg.norm(m):10.3f} {sparsity:9.3f}")


def class_level_comparison(samples: list[Sample]) -> None:
    math_means = [s.matrix.mean() for s in samples if s.label == "math"]
    other_means = [s.matrix.mean() for s in samples if s.label == "others"]
    math_norms = [np.linalg.norm(s.matrix) / s.matrix.shape[0] for s in samples if s.label == "math"]
    other_norms = [np.linalg.norm(s.matrix) / s.matrix.shape[0] for s in samples if s.label == "others"]

    print("\n--- class-level activation magnitude (n=6 vs 6) ---")
    print(f"math   mean-activation: {np.mean(math_means):.5f} +/- {np.std(math_means):.5f}")
    print(f"others mean-activation: {np.mean(other_means):.5f} +/- {np.std(other_means):.5f}")
    t, p = stats.ttest_ind(math_means, other_means)
    print(f"t-test on per-sample mean activation: t={t:.3f}, p={p:.3f}")

    print(f"math   per-token L2 norm: {np.mean(math_norms):.3f} +/- {np.std(math_norms):.3f}")
    print(f"others per-token L2 norm: {np.mean(other_norms):.3f} +/- {np.std(other_norms):.3f}")


def neuron_discriminative_ranking(samples: list[Sample], top_k: int = 20) -> np.ndarray:
    """Rank the 2816 MLP neurons by how well their (mean-pooled) activation
    separates math vs others: Welch t-stat, Cohen's d, and point-biserial MI proxy."""
    X = np.stack([pooled_stats(s.matrix)[:2816] for s in samples])  # pooled MEAN per neuron
    y = np.array([1 if s.label == "math" else 0 for s in samples])

    math_X = X[y == 1]
    other_X = X[y == 0]

    mean_diff = math_X.mean(axis=0) - other_X.mean(axis=0)
    pooled_std = np.sqrt((math_X.var(axis=0) + other_X.var(axis=0)) / 2) + 1e-9
    cohens_d = mean_diff / pooled_std

    t_stat, p_val = stats.ttest_ind(math_X, other_X, axis=0, equal_var=False)

    order = np.argsort(-np.abs(cohens_d))
    print(f"\n--- top {top_k} discriminative neurons (by |Cohen's d| on mean-pooled activation) ---")
    print(f"{'neuron_idx':>10s} {'cohens_d':>10s} {'t_stat':>8s} {'p_value':>9s} "
          f"{'math_mean':>10s} {'other_mean':>10s}")
    for idx in order[:top_k]:
        print(f"{idx:10d} {cohens_d[idx]:10.3f} {t_stat[idx]:8.3f} {p_val[idx]:9.4f} "
              f"{math_X[:, idx].mean():10.4f} {other_X[:, idx].mean():10.4f}")

    n_nominal_sig = (p_val < 0.05).sum()
    print(f"\n{n_nominal_sig}/2816 neurons have nominal p<0.05 (uncorrected; with n=12 "
          f"and 2816 tests this is expected by chance alone -- NOT Bonferroni-significant; "
          f"largest |Cohen's d| effect sizes above are more trustworthy descriptively than the p-values).")
    return order


if __name__ == "__main__":
    samples = load_samples()
    per_sample_summary(samples)
    class_level_comparison(samples)
    neuron_discriminative_ranking(samples)
