"""
Phase 3 smoke test: determinism and invariants for trader + market simulator.

Run:
  python scripts/smoke_test_phase3.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

import numpy as np

# Allow running from repo root without installation.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.trader_model import TraderParams, build_traders
from src.capopm.market_simulator import MarketConfig, simulate_market


def build_scenario_traders() -> List:
    n_traders = 50
    proportions = {"informed": 0.4, "noise": 0.5, "adversarial": 0.1}
    params_by_type = {
        "informed": TraderParams(signal_quality=0.7, noise_yes_prob=0.5, herding_intensity=0.0),
        "noise": TraderParams(signal_quality=0.7, noise_yes_prob=0.5, herding_intensity=0.0),
        "adversarial": TraderParams(signal_quality=0.7, noise_yes_prob=0.5, herding_intensity=0.0),
    }
    return build_traders(n_traders, proportions, params_by_type=params_by_type)


def build_config() -> MarketConfig:
    return MarketConfig(
        n_steps=25,
        arrivals_per_step=3,
        fee_rate=0.0,
        initial_yes_pool=1.0,
        initial_no_pool=1.0,
        signal_model="conditional_on_state",
        use_realized_state_for_signals=True,
        herding_enabled=False,
        size_dist="fixed",
        size_dist_params={"size": 1.0},
    )


def assert_trade_tapes_equal(tape1, tape2, tol=1e-12) -> None:
    assert len(tape1) == len(tape2)
    for a, b in zip(tape1, tape2):
        assert a.t == b.t
        assert a.trader_id == b.trader_id
        assert a.trader_type == b.trader_type
        assert a.side == b.side
        assert abs(a.size - b.size) <= tol
        for field in [
            "yes_pool_before",
            "no_pool_before",
            "yes_pool_after",
            "no_pool_after",
            "implied_yes_before",
            "implied_yes_after",
            "implied_no_before",
            "implied_no_after",
            "odds_yes_before",
            "odds_yes_after",
        ]:
            aval = getattr(a, field)
            bval = getattr(b, field)
            assert abs(aval - bval) <= tol


def assert_pool_paths_equal(path1, path2, tol=1e-12) -> None:
    assert len(path1) == len(path2)
    for a, b in zip(path1, path2):
        assert a[0] == b[0]
        for i in range(1, len(a)):
            assert abs(a[i] - b[i]) <= tol


def assert_invariants(trade_tape, pool_path) -> None:
    for trade in trade_tape:
        assert trade.size > 0
        assert trade.yes_pool_before > 0
        assert trade.no_pool_before > 0
        assert trade.yes_pool_after > 0
        assert trade.no_pool_after > 0
        for val in [
            trade.implied_yes_before,
            trade.implied_yes_after,
            trade.implied_no_before,
            trade.implied_no_after,
        ]:
            assert math.isfinite(val)
            assert 0.0 <= val <= 1.0
        for val in [trade.odds_yes_before, trade.odds_yes_after]:
            assert not math.isnan(val)
    for snap in pool_path:
        _, yes_pool, no_pool, p_yes, p_no = snap
        assert yes_pool >= 0
        assert no_pool >= 0
        assert math.isfinite(p_yes) and 0.0 <= p_yes <= 1.0
        assert math.isfinite(p_no) and 0.0 <= p_no <= 1.0


def main() -> None:
    seed = 12345
    p_true = 0.55
    cfg = build_config()

    rng1 = np.random.default_rng(seed)
    traders1 = build_scenario_traders()
    tape1, path1 = simulate_market(rng1, cfg, traders1, p_true)

    rng2 = np.random.default_rng(seed)
    traders2 = build_scenario_traders()
    tape2, path2 = simulate_market(rng2, cfg, traders2, p_true)

    assert_trade_tapes_equal(tape1, tape2)
    assert_pool_paths_equal(path1, path2)
    assert_invariants(tape1, path1)

    print("SMOKE TEST PASSED: determinism + invariants")


if __name__ == "__main__":
    main()
