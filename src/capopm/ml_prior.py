"""
Phase 2: Machine-Learning prior component for CAPOPM.

Implements a synthetic p_ML generator and reliability-based pseudo-count sizing
as specified in the Phase 2 hybrid prior construction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def simulate_p_ml(rng: np.random.Generator, ml_cfg: Dict[str, Any]) -> float:
    """Phase 2 synthetic ML prior probability p_ML in (0,1)."""

    base = float(ml_cfg.get("base_prob", 0.5))
    bias = float(ml_cfg.get("bias", 0.0))
    noise_std = float(ml_cfg.get("noise_std", 0.0))
    calibration = float(ml_cfg.get("calibration", 1.0))
    if not 0.0 <= base <= 1.0:
        raise ValueError("base_prob must be in [0,1]")
    if noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative")
    if calibration <= 0.0:
        raise ValueError("calibration must be positive")

    noise = float(rng.normal(0.0, noise_std)) if noise_std > 0.0 else 0.0
    raw = base + bias + noise
    # Calibration contracts/expands deviations around 0.5.
    calibrated = 0.5 + calibration * (raw - 0.5)
    p_ml = min(max(calibrated, 1e-12), 1.0 - 1e-12)
    return p_ml


def compute_r_ml(ml_cfg: Dict[str, Any], diagnostics: Optional[Dict[str, Any]] = None) -> float:
    """Phase 2 ML reliability index r_ML in [0,1], deterministic from config."""

    r_ml = float(ml_cfg.get("r_ml", 1.0))
    if r_ml < 0.0 or r_ml > 1.0:
        raise ValueError("r_ml must be in [0,1]")
    return r_ml


def compute_n_ml(r_ml: float, n_ml_eff: float, n_ml_scale: float = 1.0) -> float:
    """Phase 2 reliability-weighted pseudo sample size n_ML >= 0."""

    if r_ml < 0.0 or r_ml > 1.0:
        raise ValueError("r_ml must be in [0,1]")
    if n_ml_eff < 0.0:
        raise ValueError("n_ml_eff must be nonnegative")
    if n_ml_scale < 0.0:
        raise ValueError("n_ml_scale must be nonnegative")
    return r_ml * n_ml_eff * n_ml_scale
