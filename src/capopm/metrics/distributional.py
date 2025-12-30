"""
Phase 7 distributional metrics: bias, variance ratio, MAD median, Wasserstein.
"""

from __future__ import annotations

import math

import numpy as np

from ..pricing import beta_ppf, betainc_reg


def posterior_mean_bias(p_hat: float, p_true: float) -> float:
    """Bias of posterior mean: p_hat - p_true."""

    return p_hat - p_true


def posterior_variance_ratio(var_adj: float, var_independent: float) -> float:
    """Variance ratio relative to independent case."""

    if var_independent <= 0.0:
        return float("nan")
    return var_adj / var_independent


def mad_posterior_median(alpha: float, beta: float, p_true: float) -> float:
    """Mean absolute deviation of posterior median from p_true."""

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    median = beta_ppf(0.5, alpha, beta)
    return abs(median - p_true)


def wasserstein_distance_beta(alpha: float, beta: float, p_true: float, n_grid: int = 1000) -> float:
    """Approximate W1 distance between Beta(alpha,beta) and point mass at p_true."""

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if n_grid <= 10:
        raise ValueError("n_grid must be > 10")
    grid = np.linspace(0.0, 1.0, n_grid)
    pdf = beta_pdf(grid, alpha, beta)
    dist = np.trapz(np.abs(grid - p_true) * pdf, grid)
    return float(dist)


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Beta pdf using lgamma for normalization."""

    log_norm = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return np.exp((a - 1.0) * np.log(x + 1e-12) + (b - 1.0) * np.log(1.0 - x + 1e-12) - log_norm)
