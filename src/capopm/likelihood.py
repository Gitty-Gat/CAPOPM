"""
Phase 4: Parimutuel Likelihood and Bayesian Updating.

Implements the Beta–Binomial conjugate update from the paper's Phase 4.
"""

from __future__ import annotations

from typing import Iterable, Tuple


def beta_binomial_update(alpha0: float, beta0: float, y: float, n: float) -> Tuple[float, float]:
    """Phase 4 update: Beta prior + Binomial likelihood -> Beta posterior."""

    assert alpha0 > 0.0 and beta0 > 0.0
    assert n >= 0.0 and 0.0 <= y <= n
    alpha_post = alpha0 + y
    beta_post = beta0 + (n - y)
    assert alpha_post > 0.0 and beta_post > 0.0
    return alpha_post, beta_post


def posterior_moments(alpha: float, beta: float) -> Tuple[float, float]:
    """Phase 4 closed-form Beta posterior mean and variance."""

    assert alpha > 0.0 and beta > 0.0
    denom = alpha + beta
    mean = alpha / denom
    var = (alpha * beta) / (denom * denom * (denom + 1.0))
    return mean, var


def counts_from_trade_tape(trade_tape: Iterable) -> Tuple[float, float]:
    """Phase 4 raw counts from trades: y = YES size sum, n = total size sum."""

    y = 0.0
    n = 0.0
    for trade in trade_tape:
        size = float(trade.size)
        n += size
        if trade.side == "YES":
            y += size
    return y, n
