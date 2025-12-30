"""
Phase 4–5 smoke test: Beta–Binomial update + pricing + Beta PPF sanity.

Run:
  python scripts/smoke_test_phase45.py
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

from src.capopm.likelihood import beta_binomial_update, counts_from_trade_tape
from src.capopm.pricing import beta_ppf, credible_intervals, posterior_prices


@dataclass(frozen=True)
class Trade:
    side: str
    size: float


def main() -> None:
    # 10 YES + 10 NO trades, size=1.
    trade_tape = [Trade(side="YES", size=1.0) for _ in range(10)] + [
        Trade(side="NO", size=1.0) for _ in range(10)
    ]

    y, n = counts_from_trade_tape(trade_tape)
    alpha_post, beta_post = beta_binomial_update(1.0, 1.0, y, n)
    assert alpha_post == 11.0
    assert beta_post == 11.0

    pi_yes, _ = posterior_prices(alpha_post, beta_post)
    assert pi_yes == 0.5

    lo90, hi90 = credible_intervals(11.0, 11.0, level=0.90)
    lo95, hi95 = credible_intervals(11.0, 11.0, level=0.95)
    assert 0.0 < lo90 < 0.5 < hi90 < 1.0
    assert 0.0 < lo95 < 0.5 < hi95 < 1.0
    assert (hi95 - lo95) > (hi90 - lo90)

    # Beta PPF sanity checks
    for q in [0.01, 0.1, 0.5, 0.9, 0.99]:
        assert abs(beta_ppf(q, 1.0, 1.0) - q) <= 1e-8
    assert abs(beta_ppf(0.5, 2.0, 2.0) - 0.5) <= 1e-8

    print("SMOKE TEST PASSED: Phase 4–5 core + Beta PPF sanity")


if __name__ == "__main__":
    main()
