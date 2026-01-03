"""
Phase 6.2 Stage 1: Behavioral bias correction (long-shot + herding).

Implements the underspecified weighting rule from AGENTS.md:
  w = clip(w_min, w_max, w_LS(p_mkt) * w_H(streak))

Config keys:
  - w_min, w_max: bounds on weights (positive)
  - longshot_ref_p: reference probability for long-shot downweighting
  - longshot_gamma: strength of long-shot downweighting
  - herding_lambda: strength of herding downweighting
  - herding_window: max history window for streak detection
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from ..invariant_runtime import record_fallback


def compute_w_ls(p_mkt: float, cfg: Dict) -> float:
    """Long-shot downweighting w_LS(p_mkt) per Phase 6.2 Stage 1."""

    w_min = float(cfg.get("w_min", 0.1))
    w_max = float(cfg.get("w_max", 1.0))
    ref_p = float(cfg.get("longshot_ref_p", 0.5))
    gamma = float(cfg.get("longshot_gamma", 1.0))
    if not 0.0 < ref_p <= 1.0:
        raise ValueError("longshot_ref_p must be in (0,1]")
    if gamma < 0.0:
        raise ValueError("longshot_gamma must be nonnegative")
    if p_mkt <= 0.0:
        p_mkt = 1e-12
    ratio = p_mkt / ref_p
    w = ratio**gamma
    return clip_weight(w, w_min, w_max)


def compute_w_herd(streak_len: int, cfg: Dict) -> float:
    """Herding downweighting w_H(streak) per Phase 6.2 Stage 1."""

    w_min = float(cfg.get("w_min", 0.1))
    w_max = float(cfg.get("w_max", 1.0))
    lam = float(cfg.get("herding_lambda", 0.0))
    if lam < 0.0:
        raise ValueError("herding_lambda must be nonnegative")
    # Exponential decay in streak length.
    w = 1.0 / (1.0 + lam * max(streak_len, 0))
    return clip_weight(w, w_min, w_max)


def compute_w_beh(trade, history: List[str], cfg: Dict) -> float:
    """Behavioral weight w_i from long-shot and herding components."""

    w_min = float(cfg.get("w_min", 0.1))
    w_max = float(cfg.get("w_max", 1.0))
    w_ls = compute_w_ls(float(getattr(trade, "implied_yes_before")), cfg)
    streak_len = current_streak_len(history, getattr(trade, "side"))
    w_h = compute_w_herd(streak_len, cfg)
    return clip_weight(w_ls * w_h, w_min, w_max)


def apply_behavioral_weights(
    trade_tape: Iterable, cfg: Dict
) -> Tuple[float, float, Dict]:
    """Apply Phase 6.2 Stage 1 weights to compute effective counts."""

    history: List[str] = []
    w_min = float(cfg.get("w_min", 0.1))
    w_max = float(cfg.get("w_max", 1.0))
    window = int(cfg.get("herding_window", 50))
    if window <= 0:
        raise ValueError("herding_window must be positive")

    y1 = 0.0
    n1 = 0.0
    w_min_seen = None
    w_max_seen = None
    w_sum = 0.0
    downweighted = 0
    total = 0
    clipped_to_min = 0
    clipped_to_max = 0

    for trade in trade_tape:
        if len(history) > window:
            history = history[-window:]
        w = compute_w_beh(trade, history, cfg)
        size = float(getattr(trade, "size"))
        if size <= 0.0:
            raise ValueError("trade size must be positive")
        total += 1
        if w <= w_min:
            clipped_to_min += 1
        if w >= w_max:
            clipped_to_max += 1
        if w < 1.0:
            downweighted += 1
        w_sum += w
        w_min_seen = w if w_min_seen is None else min(w_min_seen, w)
        w_max_seen = w if w_max_seen is None else max(w_max_seen, w)

        if getattr(trade, "side") == "YES":
            y1 += w * size
        else:
            n1 += w * size
        history.append(getattr(trade, "side"))

    n1 = y1 + n1
    if n1 < 0.0 or y1 < 0.0 or y1 > n1:
        raise ValueError("Stage 1 counts must satisfy 0 <= y1 <= n1 and n1 >= 0")

    weights_summary = {
        "w_min": w_min,
        "w_max": w_max,
        "min_weight": w_min_seen if w_min_seen is not None else 0.0,
        "max_weight": w_max_seen if w_max_seen is not None else 0.0,
        "mean_weight": (w_sum / total) if total > 0 else 0.0,
        "downweighted_trades": downweighted,
        "total_trades": total,
        "clipped_to_min": clipped_to_min,
        "clipped_to_max": clipped_to_max,
    }
    record_fallback(
        "AF-01",
        {
            "w_min": w_min,
            "w_max": w_max,
            "clipped_to_min": clipped_to_min,
            "clipped_to_max": clipped_to_max,
            "total_trades": total,
        },
    )
    return y1, n1, weights_summary


def current_streak_len(history: List[str], side: str) -> int:
    """Compute the current streak length of the same side in history."""

    streak = 0
    for s in reversed(history):
        if s == side:
            streak += 1
        else:
            break
    return streak


def clip_weight(w: float, w_min: float, w_max: float) -> float:
    """Clip weights to admissible bounds."""

    if w_min <= 0.0 or w_max <= 0.0 or w_min > w_max:
        raise ValueError("w_min and w_max must be positive with w_min <= w_max")
    if w < w_min:
        return w_min
    if w > w_max:
        return w_max
    return w
