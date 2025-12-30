"""
Phase 6 Stage 2 linear offsets smoke test.

Run:
  python scripts/smoke_test_stage2_offsets.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# Allow running from repo root without installation.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.posterior import capopm_pipeline


@dataclass(frozen=True)
class Trade:
    side: str
    size: float


def main() -> None:
    trade_tape = [Trade(side="YES", size=1.0) for _ in range(10)] + [
        Trade(side="NO", size=1.0) for _ in range(10)
    ]

    structural_cfg = {"T": 1.0, "K": 1.0, "S0": 1.0, "V0": 0.04}
    ml_cfg = {"base_prob": 0.5, "bias": 0.0, "noise_std": 0.0, "calibration": 1.0, "r_ml": 1.0}
    prior_cfg = {"n_str": 10.0, "n_ml_eff": 0.0, "n_ml_scale": 1.0}

    rng_a = np.random.default_rng(42)
    out_a = capopm_pipeline(
        rng=rng_a,
        trade_tape=trade_tape,
        structural_cfg=structural_cfg,
        ml_cfg=ml_cfg,
        prior_cfg=prior_cfg,
        stage1_cfg={"enabled": False},
        stage2_cfg={"enabled": False, "delta_plus": 0.0, "delta_minus": 0.0},
    )

    rng_b = np.random.default_rng(42)
    out_b = capopm_pipeline(
        rng=rng_b,
        trade_tape=trade_tape,
        structural_cfg=structural_cfg,
        ml_cfg=ml_cfg,
        prior_cfg=prior_cfg,
        stage1_cfg={"enabled": False},
        stage2_cfg={"enabled": True, "delta_plus": 5.0, "delta_minus": 7.0},
    )

    assert out_a["y"] == 10.0
    assert out_a["n"] == 20.0
    assert out_b["y_stage2"] == 15.0
    assert out_b["n_stage2"] == 32.0

    assert (out_b["alpha_post"] - out_a["alpha_post"]) == 5.0
    assert (out_b["beta_post"] - out_a["beta_post"]) == 7.0
    assert abs((out_b["pi_yes"] + out_b["pi_no"]) - 1.0) <= 1e-12

    print("SMOKE TEST PASSED: Stage 2 linear offsets")


if __name__ == "__main__":
    main()
