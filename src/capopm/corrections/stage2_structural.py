"""
Phase 6.3 Stage 2: Structural correction layer.

Implements the linear offset special case (Phase 6.3.1) and regime-mixture
corrections (Phase 6.3, Theorem 15). Mixture regime weights use a Beta–Binomial
marginal likelihood L_r(D2) per AGENTS.md defaults.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


def apply_linear_offsets(y1: float, n1: float, cfg: Dict) -> Tuple[float, float]:
    """Apply linear offsets: y* = y1 + delta_plus, n* = n1 + delta_plus + delta_minus."""

    if n1 < 0.0:
        raise ValueError("n1 must be nonnegative")
    if y1 < 0.0 or y1 > n1:
        raise ValueError("y1 must satisfy 0 <= y1 <= n1")

    delta_plus = float(cfg.get("delta_plus", 0.0))
    delta_minus = float(cfg.get("delta_minus", 0.0))
    if delta_plus < 0.0 or delta_minus < 0.0:
        raise ValueError("delta_plus and delta_minus must be nonnegative")

    y_star = y1 + delta_plus
    n_star = n1 + delta_plus + delta_minus
    if n_star < 0.0:
        raise ValueError("n_star must be nonnegative")
    if y_star < 0.0 or y_star > n_star:
        raise ValueError("y_star must satisfy 0 <= y_star <= n_star")

    return y_star, n_star


def offset_summary(y1: float, n1: float, y_star: float, n_star: float, cfg: Dict) -> Dict:
    """Summarize linear offset adjustments for Stage 2."""

    delta_plus = float(cfg.get("delta_plus", 0.0))
    delta_minus = float(cfg.get("delta_minus", 0.0))
    return {
        "delta_plus": delta_plus,
        "delta_minus": delta_minus,
        "y_shift": y_star - y1,
        "no_shift": (n_star - y_star) - (n1 - y1),
    }


def summarize_stage1_stats(trade_tape: Iterable, y_used: float, n_used: float, cfg: Dict) -> Dict:
    """Summary statistics s for regime functions g_r±(s) (Phase 6.3)."""

    total_size = 0.0
    max_size = 0.0
    yes_size = 0.0
    no_size = 0.0
    for trade in trade_tape:
        size = float(getattr(trade, "size"))
        total_size += size
        max_size = max(max_size, size)
        side = getattr(trade, "side")
        if side == "YES":
            yes_size += size
        elif side == "NO":
            no_size += size
        else:
            raise ValueError("trade.side must be 'YES' or 'NO'")

    frac_yes = (y_used / n_used) if n_used > 0.0 else 0.5
    concentration = (max_size / total_size) if total_size > 0.0 else 0.0
    imbalance = (abs(yes_size - no_size) / total_size) if total_size > 0.0 else 0.0

    return {
        "y": y_used,
        "n": n_used,
        "frac_yes": frac_yes,
        "concentration": concentration,
        "imbalance": imbalance,
    }


def compute_g_r(s: Dict, regime: Dict, alpha_base: float, beta_base: float) -> Tuple[float, float, Dict]:
    """Compute regime-specific g_plus/g_minus with admissibility clamping."""

    g_plus_scale = float(regime.get("g_plus_scale", 0.0))
    g_minus_scale = float(regime.get("g_minus_scale", 0.0))
    imbalance = float(s.get("imbalance", 0.0))
    concentration = float(s.get("concentration", 0.0))
    frac_yes = float(s.get("frac_yes", 0.5))

    g_plus = g_plus_scale * (0.5 - frac_yes) + g_plus_scale * concentration
    g_minus = g_minus_scale * (frac_yes - 0.5) + g_minus_scale * imbalance

    clamp_flags = {"g_plus_clamped": False, "g_minus_clamped": False}

    if alpha_base + g_plus <= 0.0:
        g_plus = -alpha_base + 1e-12
        clamp_flags["g_plus_clamped"] = True
    if beta_base + g_minus <= 0.0:
        g_minus = -beta_base + 1e-12
        clamp_flags["g_minus_clamped"] = True

    return g_plus, g_minus, clamp_flags


def beta_binomial_marginal_likelihood(alpha: float, beta: float, y: float, n: float) -> float:
    """Log marginal likelihood log B(alpha+y, beta+n-y) - log B(alpha, beta)."""

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if n < 0.0 or y < 0.0 or y > n:
        raise ValueError("y and n must satisfy 0 <= y <= n and n >= 0")

    def log_beta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    return log_beta(alpha + y, beta + (n - y)) - log_beta(alpha, beta)


def regime_weights(priors_pi: List[float], logL: List[float]) -> List[float]:
    """Compute posterior regime weights using log-space softmax."""

    if len(priors_pi) != len(logL) or not priors_pi:
        raise ValueError("priors_pi and logL must have the same nonzero length")
    if any(p < 0.0 for p in priors_pi):
        raise ValueError("prior weights must be nonnegative")

    max_logL = max(logL)
    weights = []
    total = 0.0
    for pi_r, logL_r in zip(priors_pi, logL):
        w = pi_r * math.exp(logL_r - max_logL)
        weights.append(w)
        total += w
    if total == 0.0:
        priors_sum = sum(priors_pi)
        if priors_sum > 0.0:
            return [p / priors_sum for p in priors_pi]
        return [1.0 / len(weights) for _ in weights]
    return [w / total for w in weights]


def mixture_posterior_params(
    alpha_base: float, beta_base: float, s: Dict, regimes: List[Dict], y: float, n: float
) -> Dict:
    """Compute regime-mixture posterior parameters per Phase 6.3/Theorem 15."""

    if alpha_base <= 0.0 or beta_base <= 0.0:
        raise ValueError("alpha_base and beta_base must be positive")

    alphas_betas: List[Tuple[float, float]] = []
    logL: List[float] = []
    priors_pi: List[float] = []
    diagnostics = {
        "min_g_plus": None,
        "max_g_plus": None,
        "min_g_minus": None,
        "max_g_minus": None,
        "clamping": [],
    }

    y_used = float(y)
    n_used = float(n)

    for regime in regimes:
        priors_pi.append(float(regime.get("pi", 0.0)))
        g_plus, g_minus, clamp_flags = compute_g_r(s, regime, alpha_base, beta_base)
        alpha_r_prior = alpha_base + g_plus
        beta_r_prior = beta_base + g_minus
        if alpha_r_prior <= 0.0 or beta_r_prior <= 0.0:
            raise ValueError("Regime parameters must be positive after clamping")
        alpha_r_post = alpha_r_prior + y_used
        beta_r_post = beta_r_prior + (n_used - y_used)
        alphas_betas.append((alpha_r_post, beta_r_post))
        logL.append(beta_binomial_marginal_likelihood(alpha_r_prior, beta_r_prior, y_used, n_used))

        diagnostics["min_g_plus"] = g_plus if diagnostics["min_g_plus"] is None else min(diagnostics["min_g_plus"], g_plus)
        diagnostics["max_g_plus"] = g_plus if diagnostics["max_g_plus"] is None else max(diagnostics["max_g_plus"], g_plus)
        diagnostics["min_g_minus"] = g_minus if diagnostics["min_g_minus"] is None else min(diagnostics["min_g_minus"], g_minus)
        diagnostics["max_g_minus"] = g_minus if diagnostics["max_g_minus"] is None else max(diagnostics["max_g_minus"], g_minus)
        diagnostics["clamping"].append(clamp_flags)

    weights = regime_weights(priors_pi, logL)
    mixture_mean = 0.0
    mixture_second = 0.0
    for w, (a_r, b_r) in zip(weights, alphas_betas):
        mean_r = a_r / (a_r + b_r)
        var_r = (a_r * b_r) / ((a_r + b_r) ** 2 * (a_r + b_r + 1.0))
        mixture_mean += w * mean_r
        mixture_second += w * (var_r + mean_r * mean_r)
    mixture_var = mixture_second - mixture_mean * mixture_mean

    return {
        "regime_weights": weights,
        "regime_params": alphas_betas,
        "mixture_mean": mixture_mean,
        "mixture_var": mixture_var,
        "diagnostics": diagnostics,
    }
