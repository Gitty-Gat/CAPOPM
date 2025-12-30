"""
Dynamic parimutuel market simulator for CAPOPM.

Implements the Phase 3 parimutuel environment and produces a full trade tape
with evolving implied odds (Phase 3.1–3.4; odds definition in eq. (5)).
Supports Phase 7-style herding through trader decision hooks (OFF by default).

Trade tape schema (per transaction):
  - t: int time step index
  - trader_id: int
  - trader_type: str
  - side: "YES" | "NO"
  - size: float (>0)
  - yes_pool_before/no_pool_before: float
  - yes_pool_after/no_pool_after: float
  - implied_yes_before/implied_yes_after: float in [0,1]
  - implied_no_before/implied_no_after: float in [0,1]
  - odds_yes_before/odds_yes_after: float (NO/YES ratio)

Implied probability convention:
- fee_rate represents a house take; implied probabilities sum to 1 - fee_rate.
  That is, p_yes + p_no = 1 - fee_rate (net mass, not renormalized).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np

from .trader_model import Trader, empirical_yes_rate


@dataclass
class MarketConfig:
    """Configuration for a synthetic parimutuel market run."""

    n_steps: int
    arrivals_per_step: int = 1
    fee_rate: float = 0.0
    initial_yes_pool: float = 1.0
    initial_no_pool: float = 1.0
    signal_model: str = "bernoulli_p_true"
    use_realized_state_for_signals: bool = False
    herding_enabled: bool = False
    size_dist: str = "fixed"  # "fixed", "exponential", "lognormal"
    size_dist_params: Optional[Dict[str, float]] = None


@dataclass
class Trade:
    """One transaction-level action for the trade tape."""

    t: int
    trader_id: int
    trader_type: str
    side: str
    size: float
    yes_pool_before: float
    no_pool_before: float
    yes_pool_after: float
    no_pool_after: float
    implied_yes_before: float
    implied_yes_after: float
    implied_no_before: float
    implied_no_after: float
    odds_yes_before: float
    odds_yes_after: float


def simulate_market(
    rng: np.random.Generator,
    cfg: MarketConfig,
    traders: List[Trader],
    p_true: float,
) -> Tuple[List[Trade], List[Tuple[int, float, float, float, float]]]:
    """Simulate a dynamic parimutuel timeline and return trade tape and pool path.

    Returns:
      trade_tape: list of Trade entries (transaction-level actions)
      pool_path: list of (t, yes_pool, no_pool, implied_yes, implied_no) snapshots
    """

    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if cfg.arrivals_per_step <= 0:
        raise ValueError("arrivals_per_step must be positive")
    if cfg.initial_yes_pool <= 0 or cfg.initial_no_pool <= 0:
        raise ValueError("initial pools must be positive")
    if not traders:
        raise ValueError("traders list must be non-empty")

    yes_pool = cfg.initial_yes_pool
    no_pool = cfg.initial_no_pool

    trade_tape: List[Trade] = []
    pool_path: List[Tuple[int, float, float, float]] = []
    history: List[str] = []

    realized_state = None
    if cfg.use_realized_state_for_signals:
        realized_state = 1 if float(rng.random()) < p_true else 0

    if cfg.signal_model == "bernoulli_p_true":
        print(
            "Warning: signal_model=bernoulli_p_true is a Phase 7 simplified model, "
            "not the Phase 3 conditional signal model."
        )

    for t in range(cfg.n_steps):
        for _ in range(cfg.arrivals_per_step):
            trader = traders[int(rng.integers(0, len(traders)))]
            p_hist = empirical_yes_rate(history)
            side = trader.decide(
                rng=rng,
                p_true=p_true,
                p_hist=p_hist,
                signal_model=cfg.signal_model,
                realized_state=realized_state,
                herding_enabled=cfg.herding_enabled,
            )

            trade_size = sample_trade_size(rng, cfg.size_dist, cfg.size_dist_params)
            assert trade_size > 0.0

            yes_pool_before = yes_pool
            no_pool_before = no_pool
            implied_yes_before, implied_no_before = implied_probs(
                yes_pool_before, no_pool_before, cfg.fee_rate
            )
            assert 0.0 <= implied_yes_before <= 1.0
            assert 0.0 <= implied_no_before <= 1.0
            odds_yes_before = parimutuel_odds(yes_pool_before, no_pool_before)

            if side == "YES":
                yes_pool += trade_size
            else:
                no_pool += trade_size
            assert yes_pool >= 0.0 and no_pool >= 0.0

            implied_yes_after, implied_no_after = implied_probs(
                yes_pool, no_pool, cfg.fee_rate
            )
            assert 0.0 <= implied_yes_after <= 1.0
            assert 0.0 <= implied_no_after <= 1.0
            odds_yes_after = parimutuel_odds(yes_pool, no_pool)

            trade_tape.append(
                Trade(
                    t=t,
                    trader_id=trader.trader_id,
                    trader_type=trader.trader_type,
                    side=side,
                    size=trade_size,
                    yes_pool_before=yes_pool_before,
                    no_pool_before=no_pool_before,
                    yes_pool_after=yes_pool,
                    no_pool_after=no_pool,
                    implied_yes_before=implied_yes_before,
                    implied_yes_after=implied_yes_after,
                    implied_no_before=implied_no_before,
                    implied_no_after=implied_no_after,
                    odds_yes_before=odds_yes_before,
                    odds_yes_after=odds_yes_after,
                )
            )

            history.append(side)

        implied_yes, implied_no = implied_probs(yes_pool, no_pool, cfg.fee_rate)
        pool_path.append((t, yes_pool, no_pool, implied_yes, implied_no))

    return trade_tape, pool_path


def implied_probs(
    yes_pool: float, no_pool: float, fee_rate: float
) -> Tuple[float, float]:
    """Implied YES/NO probabilities with optional house take.

    Fee convention: fee_rate represents a house take, so the net probability
    mass is 1 - fee_rate. This means p_yes + p_no = 1 - fee_rate.
    """

    if yes_pool <= 0 or no_pool <= 0:
        raise ValueError("Pool sizes must be positive")
    total = yes_pool + no_pool
    if fee_rate < 0 or fee_rate >= 1:
        raise ValueError("fee_rate must be in [0, 1)")
    net = 1.0 - fee_rate
    p_yes = net * (yes_pool / total)
    p_no = net * (no_pool / total)
    return p_yes, p_no


def parimutuel_odds(yes_pool: float, no_pool: float) -> float:
    """Pool ratio NO/YES as in Phase 3 (O_yes = (n_tot - n_yes) / n_yes)."""

    if yes_pool <= 0:
        return math.inf
    return no_pool / yes_pool


def sample_trade_size(
    rng: np.random.Generator, size_dist: str, params: Optional[Dict[str, float]]
) -> float:
    """Sample trade size according to a simple synthetic distribution."""

    params = params or {}
    if size_dist == "fixed":
        return float(params.get("size", 1.0))
    if size_dist == "exponential":
        lam = float(params.get("lambda", 1.0))
        if lam <= 0:
            raise ValueError("lambda must be positive")
        return float(rng.exponential(1.0 / lam))
    if size_dist == "lognormal":
        mu = float(params.get("mu", 0.0))
        sigma = float(params.get("sigma", 0.25))
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        return float(rng.lognormal(mean=mu, sigma=sigma))

    raise ValueError(f"Unknown size_dist: {size_dist}")
