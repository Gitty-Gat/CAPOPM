"""
Phase 5: Posterior Predictive Derivative Pricing.

Implements posterior mean pricing, central Beta credible intervals, and the
arbitrage-free projection of YES/NO prices described in Phase 5 and Phase 7.8.
"""

from __future__ import annotations

import math
from typing import Tuple


def posterior_prices(alpha: float, beta: float) -> Tuple[float, float]:
    """Phase 5 posterior predictive YES/NO prices from Beta mean."""

    assert alpha > 0.0 and beta > 0.0
    p_yes = alpha / (alpha + beta)
    p_no = 1.0 - p_yes
    return p_yes, p_no


def credible_intervals(alpha: float, beta: float, level: float) -> Tuple[float, float]:
    """Phase 5 central Beta credible interval at the given level."""

    assert alpha > 0.0 and beta > 0.0
    assert 0.0 < level < 1.0
    tail = 0.5 * (1.0 - level)
    lo = beta_ppf(tail, alpha, beta)
    hi = beta_ppf(1.0 - tail, alpha, beta)
    return lo, hi


def arbitrage_free_projection(pi_yes: float, pi_no: float, eps: float = 1e-12) -> Tuple[float, float]:
    """Project YES/NO prices onto the simplex (Phase 7.8)."""

    if pi_yes >= 0.0 and pi_no >= 0.0 and abs((pi_yes + pi_no) - 1.0) <= eps:
        return pi_yes, pi_no

    v_yes = max(pi_yes, eps)
    v_no = max(pi_no, eps)
    s = v_yes + v_no
    return v_yes / s, v_no / s


def beta_ppf(q: float, a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Inverse CDF for Beta(a,b) via bisection on the regularized incomplete beta."""

    assert 0.0 <= q <= 1.0
    assert a > 0.0 and b > 0.0
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cdf = betainc_reg(mid, a, b)
        if abs(cdf - q) <= tol:
            return mid
        if cdf < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def betainc_reg(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a,b) using a continued fraction."""

    assert 0.0 <= x <= 1.0
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp((a * math.log(x)) + (b * math.log(1.0 - x)) - ln_beta) / a

    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(x, a, b)
    else:
        return 1.0 - front * betacf(1.0 - x, b, a)


def betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 3e-14) -> float:
    """Continued fraction for incomplete beta (Numerical Recipes style)."""

    am = 1.0
    bm = 1.0
    az = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap

    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        if bpp == 0.0:
            break
        am = ap / bpp
        bm = bp / bpp
        aold = az
        az = app / bpp
        bz = 1.0
        if abs(az - aold) < eps * abs(az):
            break

    return az
