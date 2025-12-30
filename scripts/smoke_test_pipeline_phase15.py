"""
Phase 1–5 pipeline smoke test (surrogate structural prior).

Run:
  python scripts/smoke_test_pipeline_phase15.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow running from repo root without installation.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.market_simulator import MarketConfig, simulate_market
from src.capopm.posterior import capopm_pipeline
from src.capopm.trader_model import TraderParams, build_traders


def main() -> None:
    seed = 202501
    rng = np.random.default_rng(seed)

    traders = build_traders(
        n_traders=30,
        proportions={"informed": 0.5, "noise": 0.4, "adversarial": 0.1},
        params_by_type={
            "informed": TraderParams(signal_quality=0.7, noise_yes_prob=0.5, herding_intensity=0.0),
            "noise": TraderParams(signal_quality=0.7, noise_yes_prob=0.5, herding_intensity=0.0),
            "adversarial": TraderParams(signal_quality=0.7, noise_yes_prob=0.5, herding_intensity=0.0),
        },
    )

    cfg = MarketConfig(
        n_steps=10,
        arrivals_per_step=2,
        fee_rate=0.0,
        initial_yes_pool=1.0,
        initial_no_pool=1.0,
        signal_model="conditional_on_state",
        use_realized_state_for_signals=True,
        herding_enabled=False,
        size_dist="fixed",
        size_dist_params={"size": 1.0},
    )

    p_true = 0.55
    trade_tape, _ = simulate_market(rng, cfg, traders, p_true)

    structural_cfg = {
        "T": 1.0,
        "K": 1.0,
        "S0": 1.0,
        "V0": 0.04,
        "kappa": 1.0,
        "theta": 0.04,
        "xi": 0.2,
        "rho": -0.3,
        "alpha": 0.7,
        "lambda": 0.1,
    }
    ml_cfg = {"base_prob": 0.5, "bias": 0.0, "noise_std": 0.01, "calibration": 1.0, "r_ml": 0.8}
    prior_cfg = {"n_str": 10.0, "n_ml_eff": 5.0, "n_ml_scale": 1.0}

    out = capopm_pipeline(
        rng=rng,
        trade_tape=trade_tape,
        structural_cfg=structural_cfg,
        ml_cfg=ml_cfg,
        prior_cfg=prior_cfg,
    )

    # Determinism check: re-run with same seed and inputs.
    rng2 = np.random.default_rng(seed)
    trade_tape2, _ = simulate_market(rng2, cfg, traders, p_true)
    out2 = capopm_pipeline(
        rng=rng2,
        trade_tape=trade_tape2,
        structural_cfg=structural_cfg,
        ml_cfg=ml_cfg,
        prior_cfg=prior_cfg,
    )

    assert 0.0 < out["q_str"] < 1.0
    assert 0.0 < out["p_ML"] < 1.0
    assert out["alpha0"] > 0.0 and out["beta0"] > 0.0
    assert out["alpha_post"] > out["alpha0"]
    assert out["beta_post"] > out["beta0"]
    assert abs((out["pi_yes"] + out["pi_no"]) - 1.0) <= 1e-12
    lo90, hi90 = out["credible_intervals_90"]
    assert lo90 <= out["pi_yes"] <= hi90
    lo95, hi95 = out["credible_intervals_95"]
    assert lo95 <= out["pi_yes"] <= hi95

    # Determinism: key outputs should match on rerun with same seed/config.
    keys = ["q_str", "p_ML", "alpha0", "beta0", "alpha_post", "beta_post", "pi_yes", "pi_no"]
    for k in keys:
        assert np.allclose(out[k], out2[k]), f"Determinism failed for key {k}"
    assert np.allclose(out["credible_intervals_90"], out2["credible_intervals_90"])
    assert np.allclose(out["credible_intervals_95"], out2["credible_intervals_95"])

    print("SMOKE TEST PASSED: Phase 1–5 pipeline")


if __name__ == "__main__":
    main()
