"""
Regime diagnostics for Stage 2 mixture models.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def regime_entropy(weights: Iterable[float], eps: float = 1e-12) -> float:
    """Shannon entropy of mixture weights with epsilon clamp."""

    total = 0.0
    w_list = []
    for w in weights:
        w_f = float(w)
        w_list.append(w_f)
        total += w_f
    if total <= 0.0:
        return float("nan")
    entropy = 0.0
    for w in w_list:
        p = max(w / total, eps)
        entropy -= p * math.log(p)
    return float(entropy)


def regime_max_weight(weights: Iterable[float]) -> float:
    """Maximum regime weight (NaN if empty)."""

    w_list = [float(w) for w in weights]
    if not w_list:
        return float("nan")
    return float(max(w_list))
