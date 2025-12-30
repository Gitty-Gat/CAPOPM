"""
Statistical tests for Phase 7 evaluation (paired by seed).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np


def paired_t_test(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    """Paired t-test using normal approximation for p-value."""

    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    diff = x - y
    n = diff.size
    if n < 2:
        return 0.0, 1.0
    mean = diff.mean()
    std = diff.std(ddof=1)
    if std == 0.0:
        return 0.0, 1.0
    t_stat = mean / (std / math.sqrt(n))
    p = 2.0 * (1.0 - normal_cdf(abs(t_stat)))
    return float(t_stat), float(p)


def wilcoxon_signed_rank(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test using normal approximation."""

    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    diff = x - y
    diff = diff[diff != 0.0]
    n = diff.size
    if n == 0:
        return 0.0, 1.0
    ranks = rank_abs(diff)
    w_plus = ranks[diff > 0.0].sum()
    mean = n * (n + 1.0) / 4.0
    var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
    if var == 0.0:
        return float(w_plus), 1.0
    z = (w_plus - mean) / math.sqrt(var)
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    return float(w_plus), float(p)


def bootstrap_ci(diff: Iterable[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    """Bootstrap CI for mean difference."""

    diff = np.asarray(list(diff), dtype=float)
    if diff.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(diff, size=diff.size, replace=True)
        means.append(sample.mean())
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def holm_correction(p_values: List[float]) -> List[float]:
    """Holm multiple-comparison correction."""

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    corrected = [0.0] * m
    for i, (idx, p) in enumerate(indexed):
        corrected[idx] = min((m - i) * p, 1.0)
    return corrected


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Bonferroni correction."""

    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def normal_cdf(x: float) -> float:
    """Standard normal CDF."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rank_abs(diff: np.ndarray) -> np.ndarray:
    """Average ranks for ties based on absolute differences."""

    abs_diff = np.abs(diff)
    order = np.argsort(abs_diff)
    ranks = np.empty_like(abs_diff)
    i = 0
    rank = 1
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs_diff[order[j + 1]] == abs_diff[order[i]]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        rank += (j - i) + 1
        i = j + 1
    return ranks
