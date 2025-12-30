"""
Phase 6 Stage 2 regime-mixture smoke test.

Run:
  python scripts/smoke_test_stage2_mixture.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# Allow running from repo root without installation.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.corrections.stage2_structural import mixture_posterior_params


@dataclass(frozen=True)
class Trade:
    side: str
    size: float


def main() -> None:
    trade_tape = [Trade(side="YES", size=1.0) for _ in range(10)] + [
        Trade(side="NO", size=1.0) for _ in range(10)
    ]
    y = 10.0
    n = 20.0
    s = {
        "y": y,
        "n": n,
        "frac_yes": 0.5,
        "concentration": 0.05,
        "imbalance": 0.0,
    }

    alpha0, beta0 = 2.0, 2.0
    regimes = [
        {"name": "R1", "pi": 0.5, "g_plus_scale": 0.0, "g_minus_scale": 0.0},
        {"name": "R2", "pi": 0.5, "g_plus_scale": 5.0, "g_minus_scale": 0.0},
    ]

    mix = mixture_posterior_params(alpha0, beta0, s, regimes, y, n)
    weights = mix["regime_weights"]
    params = mix["regime_params"]
    assert len(weights) == 2
    assert abs(sum(weights) - 1.0) <= 1e-12
    for a, b in params:
        assert a > 0.0 and b > 0.0
    mean_r = [a / (a + b) for a, b in params]
    assert min(mean_r) <= mix["mixture_mean"] <= max(mean_r)
    assert mix["mixture_mean"] > 0.5

    print("SMOKE TEST PASSED: Stage 2 regime mixture")


if __name__ == "__main__":
    main()
