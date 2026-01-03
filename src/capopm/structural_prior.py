"""
Phase 1: Structural prior SURROGATE for q_str = Q(S_T > K).

This is a SURROGATE implementation (per AGENTS.md) and does NOT solve the
tempered fractional Heston model. It preserves the Phase 1 interface and
parameterization, producing a deterministic q_str in (0,1) from the provided
structural parameters unless rng is explicitly used.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .invariant_runtime import require_invariant

# Tolerances from Stage B.0 governance addendum.
POS_TOL = 1e-8
TAIL_MONO_REL_TOL = 1e-6
DEFAULT_TAIL_GRID_POINTS = 10
DEFAULT_T = 1.0
DEFAULT_R = 0.0
DEFAULT_Q = 0.0


def compute_q_str(params: Dict, rng: Optional[np.random.Generator] = None) -> float:
    """Phase 1 surrogate q_str computation; deterministic unless rng is provided."""

    # Required Phase 1 inputs (used directly in the surrogate mapping).
    if "T" not in params or "K" not in params:
        raise ValueError("params must include at least T and K")
    T = float(params["T"])
    K = float(params["K"])
    if T <= 0.0 or K <= 0.0:
        raise ValueError("T and K must be positive")

    # Optional structural inputs (used if provided).
    S0 = float(params.get("S0", 1.0))
    V0 = float(params.get("V0", 0.04))
    kappa = float(params.get("kappa", 1.0))
    theta = float(params.get("theta", 0.04))
    xi = float(params.get("xi", 0.2))
    rho = float(params.get("rho", -0.3))
    alpha = float(params.get("alpha", 0.7))
    lam = float(params.get("lambda", 0.1))

    if S0 <= 0.0 or V0 <= 0.0:
        raise ValueError("S0 and V0 must be positive")

    # Surrogate: map structural inputs to a smooth logit of moneyness.
    moneyness = math.log(S0 / K)
    vol_scale = math.sqrt(max(V0, 1e-12))
    rough_adj = 0.1 * (alpha - 0.5)
    mean_reversion_adj = 0.05 * (kappa - 1.0) + 0.05 * (theta - V0)
    corr_adj = 0.05 * rho
    temper_adj = -0.05 * lam
    volvol_adj = 0.02 * xi

    score = (
        moneyness / max(vol_scale * math.sqrt(T), 1e-12)
        + rough_adj
        + mean_reversion_adj
        + corr_adj
        + temper_adj
        + volvol_adj
    )

    if rng is not None:
        jitter = float(params.get("surrogate_jitter", 0.0))
        if jitter < 0.0:
            raise ValueError("surrogate_jitter must be nonnegative")
        if jitter > 0.0:
            score += float(rng.normal(0.0, jitter))

    # Logistic to (0,1), clamp to avoid boundary issues.
    q_str = 1.0 / (1.0 + math.exp(-score))
    q_str = min(max(q_str, 1e-12), 1.0 - 1e-12)
    return q_str


def surrogate_q_str(params: Dict, rng: Optional[np.random.Generator] = None) -> float:
    """Explicit alias for the Phase 1 SURROGATE q_str computation."""

    return compute_q_str(params, rng=rng)


def enforce_structural_invariants(structural_cfg: Dict, q_str: float) -> None:
    """Enforce Stage B.1 structural invariants (hard fail on violation)."""
    # B1-CHG-05: surrogate structural prior invariant enforcement.

    require_invariant(
        bool(q_str > POS_TOL) and bool(q_str < 1.0 - POS_TOL),
        invariant_id="S-1",
        message="q_str positivity in (0,1)",
        tolerance=POS_TOL,
        data={"q_str": float(q_str)},
    )

    # Tail monotonicity proxy over the governance-specified domain.
    S0 = float(structural_cfg.get("S0", 1.0))
    T = float(structural_cfg.get("T", DEFAULT_T))
    if S0 <= 0.0 or T <= 0.0:
        raise ValueError("S0 and T must be positive for structural invariants")
    K_min = 0.5 * S0
    K_max = 1.5 * S0
    grid_points = int(structural_cfg.get("monotonic_grid_points", DEFAULT_TAIL_GRID_POINTS))
    grid_points = max(grid_points, DEFAULT_TAIL_GRID_POINTS)
    Ks = np.linspace(K_min, K_max, grid_points)
    prev_val = None
    prev_K = None
    for K in Ks:
        params = dict(structural_cfg)
        params["K"] = float(K)
        params["T"] = float(T)
        val = compute_q_str(params, rng=None)
        require_invariant(
            bool(val > POS_TOL) and bool(val < 1.0 - POS_TOL),
            invariant_id="S-1",
            message="q_str positivity across tail grid",
            tolerance=POS_TOL,
            data={"K": float(K), "q_str": float(val)},
        )
        if prev_val is not None:
            allowed = abs(prev_val) * TAIL_MONO_REL_TOL
            require_invariant(
                val <= prev_val + allowed,
                invariant_id="S-3",
                message="Tail monotonicity q_str(K) non-increasing in K",
                tolerance=TAIL_MONO_REL_TOL,
                data={"K_prev": float(prev_K), "K": float(K), "q_prev": float(prev_val), "q": float(val)},
            )
        prev_val = val
        prev_K = K
