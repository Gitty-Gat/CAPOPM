"""
Phase 7 scoring rules: Brier, log score, MAE probability error.
"""

from __future__ import annotations

import math


def brier(p_true: float, p_hat: float) -> float:
    """Phase 7 Brier score: (p_true - p_hat)^2."""

    return (p_true - p_hat) ** 2


def log_score(p_hat: float, outcome: int) -> float:
    """Phase 7 log score: log(p_hat) if outcome==1 else log(1-p_hat)."""

    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    eps = 1e-12
    p = min(max(p_hat, eps), 1.0 - eps)
    if outcome == 1:
        return math.log(p)
    return math.log(1.0 - p)


def mae_prob(p_true: float, p_hat: float) -> float:
    """Phase 7 mean absolute probability error: |p_true - p_hat|."""

    return abs(p_true - p_hat)
