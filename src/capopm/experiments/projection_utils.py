"""
Experiment-only projection utilities for arbitrage/coherence tests.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List


def project_probs(prob_vector: Iterable[float], method: str = "euclidean", eps: float = 1e-12) -> List[float]:
    """Project probability vector onto the simplex using the chosen method."""

    v = [max(float(p), eps) for p in prob_vector]
    if not v:
        raise ValueError("prob_vector must be non-empty")
    if method == "kl":
        total = sum(v)
        if total <= 0.0:
            raise ValueError("Sum of probabilities must be positive for KL projection")
        return [max(p / total, eps) for p in v]
    if method == "euclidean":
        return _euclidean_simplex_projection(v, eps=eps)
    raise ValueError(f"Unknown projection method: {method}")


def projection_distance(before: Iterable[float], after: Iterable[float], metrics=None, eps: float = 1e-12) -> Dict[str, float]:
    """Compute distances between two probability vectors."""

    metrics = metrics or ["l1", "l2", "kl"]
    b = [float(x) for x in before]
    a = [float(x) for x in after]
    if len(b) != len(a):
        raise ValueError("before and after must have the same length")
    out: Dict[str, float] = {}
    if "l1" in metrics:
        out["proj_l1"] = float(sum(abs(x - y) for x, y in zip(b, a)))
    if "l2" in metrics:
        out["proj_l2"] = float(math.sqrt(sum((x - y) ** 2 for x, y in zip(b, a))))
    if "kl" in metrics:
        kl = 0.0
        for x, y in zip(b, a):
            x_safe = max(x, eps)
            y_safe = max(y, eps)
            kl += x_safe * math.log(x_safe / y_safe)
        out["proj_kl"] = float(kl)
    return out


def detect_violation(prob_vector: Iterable[float], tol: float = 1e-9, require_sum: bool = True) -> Dict[str, object]:
    """Detect simplex/coherence violations for a probability vector."""

    v = [float(p) for p in prob_vector]
    below_zero = any(p < -tol for p in v)
    above_one = any(p > 1.0 + tol for p in v)
    sum_val = sum(v)
    sum_deviation = abs(sum_val - 1.0) if require_sum else 0.0
    sum_bad = require_sum and sum_deviation > tol
    violated = below_zero or above_one or sum_bad
    return {
        "violated": bool(violated),
        "below_zero": bool(below_zero),
        "above_one": bool(above_one),
        "sum_deviation": float(sum_deviation),
        "sum_bad": bool(sum_bad),
        "sum_val": float(sum_val),
    }


def _euclidean_simplex_projection(v: List[float], eps: float) -> List[float]:
    """Euclidean projection onto the simplex (nonnegative, sum=1)."""

    n = len(v)
    if n == 0:
        raise ValueError("Vector length must be positive")
    # Sort descending
    u = sorted(v, reverse=True)
    cssv = [0.0] * n
    for i in range(n):
        cssv[i] = u[i] + (cssv[i - 1] if i > 0 else 0.0)
    rho = -1
    theta = 0.0
    for i in range(n):
        t = (cssv[i] - 1.0) / float(i + 1)
        if u[i] - t > 0.0:
            rho = i
            theta = t
    if rho == -1:
        theta = (sum(u) - 1.0) / float(n)
        rho = n - 1
    w = [max(x - theta, eps) for x in v]
    total = sum(w)
    if total <= 0.0:
        return [1.0 / n for _ in w]
    return [x / total for x in w]

