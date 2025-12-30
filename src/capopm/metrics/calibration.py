"""
Phase 7 calibration metrics: reliability bins, ECE, and interval coverage.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..pricing import beta_ppf, credible_intervals


def reliability_bins(
    p_hat_list: List[float],
    outcome_list: List[int],
    n_bins: int = 10,
    binning: str = "equal_width",
) -> Dict[str, np.ndarray]:
    """Compute reliability bins.

    binning options:
    - equal_width: fixed edges on [0,1]
    - equal_mass: quantile-based edges on p_hat_list
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if len(p_hat_list) != len(outcome_list):
        raise ValueError("p_hat_list and outcome_list must have same length")
    p_hat = np.asarray(p_hat_list, dtype=float)
    outcome = np.asarray(outcome_list, dtype=int)
    if p_hat.ndim != 1 or outcome.ndim != 1:
        raise ValueError("p_hat_list and outcome_list must be 1D sequences")
    if p_hat.size == 0:
        raise ValueError("p_hat_list must be non-empty")
    if np.isnan(p_hat).any():
        raise ValueError("p_hat_list contains NaN")
    if not np.isin(outcome, [0, 1]).all():
        raise ValueError("outcome_list must contain only 0/1 values")

    if binning not in {"equal_width", "equal_mass"}:
        raise ValueError("binning must be 'equal_width' or 'equal_mass'")

    if binning == "equal_width":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bin_edges = np.quantile(p_hat, np.linspace(0.0, 1.0, n_bins + 1))
        # Ensure edges span [0,1] numerically.
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0

    counts = np.zeros(n_bins, dtype=int)
    mean_pred = np.zeros(n_bins, dtype=float)
    mean_outcome = np.zeros(n_bins, dtype=float)

    for p, o in zip(p_hat, outcome):
        if p < 0.0 or p > 1.0:
            raise ValueError("p_hat values must be in [0,1]")
        # Bin assignment mirrors np.digitize with right=False but with safe upper bound.
        idx = np.searchsorted(bin_edges, p, side="right") - 1
        idx = min(max(idx, 0), n_bins - 1)
        counts[idx] += 1
        mean_pred[idx] += p
        mean_outcome[idx] += float(o)

    for i in range(n_bins):
        if counts[i] > 0:
            mean_pred[i] /= counts[i]
            mean_outcome[i] /= counts[i]

    return {
        "counts": counts,
        "mean_pred": mean_pred,
        "mean_outcome": mean_outcome,
        "bin_edges": bin_edges,
        "binning_mode": binning,
    }


def calibration_ece(
    p_hat_list: List[float],
    outcome_list: List[int],
    n_bins: int = 10,
    binning: str = "equal_width",
    min_nonempty_bins: int = 5,
    allow_fallback: bool = True,
) -> Tuple[float, Dict[str, int]]:
    """Expected Calibration Error with selectable binning.

    Uses equal-width bins by default; if allow_fallback is True and equal-width
    produces fewer than min_nonempty_bins non-empty bins, falls back to
    equal-mass (quantile) binning. ECE formula is unchanged.
    """

    bins = reliability_bins(p_hat_list, outcome_list, n_bins, binning)
    counts = bins["counts"]
    total = int(counts.sum())
    if total == 0:
        return 0.0, {
            "n_unique_predictions": 0,
            "n_nonempty_bins": 0,
            "degenerate_binning": True,
            "binning_mode_used": bins["binning_mode"],
            "binning_mode_requested": binning,
            "fallback_applied": False,
        }

    def _ece_from_bins(bins_dict: Dict[str, np.ndarray]) -> Tuple[float, int]:
        ece_val = 0.0
        nonempty_bins = 0
        for c, mp, mo in zip(
            bins_dict["counts"], bins_dict["mean_pred"], bins_dict["mean_outcome"]
        ):
            if c > 0:
                nonempty_bins += 1
                ece_val += (c / total) * abs(mp - mo)
        return ece_val, nonempty_bins

    ece, nonempty = _ece_from_bins(bins)
    fallback_applied = False
    used_binning = bins["binning_mode"]

    if (
        allow_fallback
        and binning == "equal_width"
        and nonempty < min_nonempty_bins
        and len(p_hat_list) >= min_nonempty_bins
    ):
        bins = reliability_bins(p_hat_list, outcome_list, n_bins, binning="equal_mass")
        ece, nonempty = _ece_from_bins(bins)
        used_binning = "equal_mass"
        fallback_applied = True

    p_hat = np.asarray(p_hat_list, dtype=float)
    unique = int(np.unique(np.round(p_hat, 6)).size)
    degenerate = (nonempty <= 2) or (unique <= 2)
    diagnostics = {
        "n_unique_predictions": unique,
        "n_nonempty_bins": nonempty,
        "degenerate_binning": degenerate,
        "binning_mode_used": used_binning,
        "binning_mode_requested": binning,
        "fallback_applied": fallback_applied,
    }
    return float(ece), diagnostics


def interval_coverage_outcome(alpha: float, beta: float, outcome: int, level: float) -> float:
    """Central Beta credible interval coverage for outcome (0/1)."""

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0,1)")
    tail = 0.5 * (1.0 - level)
    lo = beta_ppf(tail, alpha, beta)
    hi = beta_ppf(1.0 - tail, alpha, beta)
    return 1.0 if lo <= float(outcome) <= hi else 0.0


def interval_coverage_ptrue(alpha: float, beta: float, p_true: float, level: float) -> float:
    """Central Beta credible interval coverage for p_true."""

    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0,1)")
    if p_true < 0.0 or p_true > 1.0:
        raise ValueError("p_true must be in [0,1]")
    lo, hi = credible_intervals(alpha, beta, level)
    return 1.0 if lo <= float(p_true) <= hi else 0.0


def mae_vs_outcome(p_hat_list: List[float], outcome_list: List[int]) -> float:
    """Mean absolute error between probabilities and outcomes (not calibration/ECE)."""

    p_hat = np.asarray(p_hat_list, dtype=float)
    outcome = np.asarray(outcome_list, dtype=int)
    if p_hat.ndim != 1 or outcome.ndim != 1:
        raise ValueError("p_hat_list and outcome_list must be 1D sequences")
    if p_hat.size == 0 or p_hat.size != outcome.size:
        raise ValueError("p_hat_list and outcome_list must be same nonzero length")
    return float(np.mean(np.abs(p_hat - outcome)))


def reliability_table(
    p_hat_list: List[float],
    outcome_list: List[int],
    n_bins: int = 10,
    binning: str = "equal_width",
    min_nonempty_bins: int = 5,
    allow_fallback: bool = True,
) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    """Return reliability rows (bin_low, bin_high, count, mean_pred, mean_outcome)."""

    ece_val, diag = calibration_ece(
        p_hat_list,
        outcome_list,
        n_bins=n_bins,
        binning=binning,
        min_nonempty_bins=min_nonempty_bins,
        allow_fallback=allow_fallback,
    )
    # Use the binning mode actually used for ECE.
    bins = reliability_bins(
        p_hat_list,
        outcome_list,
        n_bins=n_bins,
        binning=diag.get("binning_mode_used", binning),
    )
    rows = []
    edges = bins["bin_edges"]
    for i in range(len(edges) - 1):
        rows.append(
            {
                "bin_low": float(edges[i]),
                "bin_high": float(edges[i + 1]),
                "count": int(bins["counts"][i]),
                "mean_pred": float(bins["mean_pred"][i]),
                "mean_outcome": float(bins["mean_outcome"][i]),
            }
        )
    # ece_val unused but ensures table aligns with calibration_ece binning.
    _ = ece_val
    return rows, diag
