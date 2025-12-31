"""
Time-path metrics for Tier A convergence experiments.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def group_trades_by_step(trade_tape: Iterable, n_steps: int) -> List[List]:
    """Group trades by their integer timestep index."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    grouped: List[List] = [[] for _ in range(n_steps)]
    for trade in trade_tape:
        t = int(getattr(trade, "t"))
        if t < 0 or t >= n_steps:
            raise ValueError(f"trade.t out of bounds: {t}")
        grouped[t].append(trade)
    return grouped


def compute_time_to_eps(p_hat_path: Sequence[float], p_true: float, eps: float) -> float:
    """Return the first 1-based timestep where squared error <= eps (NaN if never)."""

    if eps < 0.0:
        raise ValueError("eps must be nonnegative")
    for idx, p_hat in enumerate(p_hat_path):
        err = (float(p_hat) - float(p_true)) ** 2
        if err <= eps:
            return float(idx + 1)
    return float("nan")


def compute_var_decay_slope(var_path: Sequence[float]) -> float:
    """OLS slope of variance vs time index (1..T); NaN if insufficient points."""

    if not var_path:
        return float("nan")
    xs = []
    ys = []
    for i, v in enumerate(var_path, start=1):
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        xs.append(float(i))
        ys.append(float(v))
    if len(xs) < 2:
        return float("nan")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0.0:
        return float("nan")
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return float(numer / denom)
