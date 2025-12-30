"""
Phase 2: Structural–ML hybrid prior fusion for CAPOPM.

Implements Beta pseudo-count construction and additive fusion as specified
in the Phase 2 hybrid prior definition.
"""

from __future__ import annotations

from typing import Tuple


def build_structural_beta(q_str: float, n_str: float) -> Tuple[float, float]:
    """Phase 2 structural Beta prior parameters from q_str and n_str."""

    if not 0.0 < q_str < 1.0:
        raise ValueError("q_str must be in (0,1)")
    if n_str < 0.0:
        raise ValueError("n_str must be nonnegative")
    alpha = n_str * q_str
    beta = n_str * (1.0 - q_str)
    return alpha, beta


def build_ml_beta(p_ml: float, n_ml: float) -> Tuple[float, float]:
    """Phase 2 ML Beta prior parameters from p_ML and n_ML."""

    if not 0.0 < p_ml < 1.0:
        raise ValueError("p_ml must be in (0,1)")
    if n_ml < 0.0:
        raise ValueError("n_ml must be nonnegative")
    alpha = n_ml * p_ml
    beta = n_ml * (1.0 - p_ml)
    return alpha, beta


def fuse_priors(
    alpha_str: float, beta_str: float, alpha_ml: float, beta_ml: float
) -> Tuple[float, float]:
    """Phase 2 conjugate fusion: add Beta pseudo-counts."""

    if alpha_str < 0.0 or beta_str < 0.0 or alpha_ml < 0.0 or beta_ml < 0.0:
        raise ValueError("Beta parameters must be nonnegative")
    return alpha_str + alpha_ml, beta_str + beta_ml


def hybrid_weights(n_str: float, n_ml: float) -> Tuple[float, float]:
    """Phase 2 hybrid weights from pseudo-count totals."""

    if n_str < 0.0 or n_ml < 0.0:
        raise ValueError("Pseudo-counts must be nonnegative")
    total = n_str + n_ml
    if total == 0.0:
        return 0.0, 0.0
    return n_str / total, n_ml / total
