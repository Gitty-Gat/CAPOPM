"""
Lightweight event-time MCMC extrapolation for incoming trades.

Implements a regime-switching marked point process with:
- Poisson arrivals (rate lambda_r)
- Side Bernoulli(p_buy_r)
- Size ~ LogNormal(mean, sigma)
- Price impact ~ Normal(mean, sigma)

Chains are run sequentially: each chain starts from the end state of the
previous chain to maintain continuity, and emits Databento-form MBO events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class RegimeSpec:
    name: str
    lam: float
    p_buy: float
    size_mean: float
    size_sigma: float
    impact_mean: float
    impact_sigma: float


def load_regime_config(path: str) -> Tuple[List[RegimeSpec], np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    regimes = []
    for r in raw["regimes"]:
        regimes.append(
            RegimeSpec(
                name=r["name"],
                lam=float(r["lambda"]),
                p_buy=float(r["p_buy"]),
                size_mean=float(r["size_lognormal"]["mean"]),
                size_sigma=float(r["size_lognormal"]["sigma"]),
                impact_mean=float(r["price_impact"]["mean"]),
                impact_sigma=float(r["price_impact"]["sigma"]),
            )
        )
    matrix = np.array(raw["transitions"]["matrix"], dtype=np.float64)
    return regimes, matrix


class EventTimeMCMCSampler:
    def __init__(
        self,
        regimes: List[RegimeSpec],
        transition_matrix: np.ndarray,
        n_chains: int,
        warmup: int,
        draws: int,
        max_events_per_chain: int,
        logger: logging.Logger,
        seed: int = 0,
    ):
        self.regimes = regimes
        self.transition = transition_matrix
        self.n_chains = n_chains
        self.warmup = warmup
        self.draws = draws
        self.max_events = max_events_per_chain
        self.log = logger
        self.rng = np.random.default_rng(seed)

        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("transition_matrix must be square")
        if transition_matrix.shape[0] != len(regimes):
            raise ValueError("transition_matrix dimension must match regimes")

    def _step_regime(self, current: int) -> int:
        probs = self.transition[current]
        return int(self.rng.choice(len(self.regimes), p=probs))

    def _simulate_chain(
        self,
        chain_id: int,
        init_regime: int,
        init_mid: float,
        start_ts_event_ns: int,
        start_order_id: int,
        instrument_id: int,
        price_scale: float,
        tick_size: float,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        t = 0.0
        regime = init_regime
        mid = init_mid
        ts_event = int(start_ts_event_ns)
        order_id = int(start_order_id)
        records: List[Dict] = []

        # Warmup discarded samples (affects regime state).
        for _ in range(self.warmup):
            regime = self._step_regime(regime)

        def quantize_price(price_float: float) -> Tuple[int, float]:
            snapped = tick_size * round(price_float / tick_size)
            price_int = int(round(snapped / price_scale))
            return price_int, snapped

        for _ in range(self.max_events):
            spec = self.regimes[regime]
            dt = self.rng.exponential(1.0 / spec.lam)
            t += dt
            ts_event += int(dt * 1e9)
            side_buy = self.rng.random() < spec.p_buy
            side = "B" if side_buy else "S"
            size = float(self.rng.lognormal(mean=spec.size_mean, sigma=spec.size_sigma))
            impact = float(self.rng.normal(loc=spec.impact_mean, scale=spec.impact_sigma))
            mid = mid + impact if side_buy else mid - impact

            trade_price = mid + (tick_size * 0.5 if side_buy else -tick_size * 0.5)
            price_int, price_float = quantize_price(trade_price)
            order_id += 1

            records.append(
                {
                    "ts_event": int(ts_event),
                    "ts_recv": int(ts_event),
                    "instrument_id": int(instrument_id),
                    "action": "F",
                    "side": side,
                    "price": price_int,
                    "size": int(max(1, round(size))),
                    "order_id": order_id,
                    "price_float": price_float,
                    "regime": regime,
                    "chain": chain_id,
                    "mid": mid,
                }
            )

            regime = self._step_regime(regime)
            if len(records) >= self.draws:
                break

        end_state = {
            "regime": regime,
            "mid": mid,
            "ts_event": ts_event,
            "order_id": order_id,
        }
        return pd.DataFrame.from_records(records), end_state

    def run(
        self,
        init_regime: int,
        init_mid: float,
        start_ts_event_ns: int,
        start_order_id: int,
        instrument_id: int,
        price_scale: float,
        tick_size: float,
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame, Dict[str, float]]:
        """Run sequential chains; each chain starts from previous end state."""
        chains: List[pd.DataFrame] = []
        combined = []
        regime = init_regime
        mid = init_mid
        ts_event = start_ts_event_ns
        order_id = start_order_id
        for c in range(self.n_chains):
            chain_df, end_state = self._simulate_chain(
                c,
                regime,
                mid,
                ts_event,
                order_id,
                instrument_id,
                price_scale,
                tick_size,
            )
            chains.append(chain_df)
            combined.append(chain_df)
            regime = int(end_state["regime"])
            mid = float(end_state["mid"])
            ts_event = int(end_state["ts_event"])
            order_id = int(end_state["order_id"])
        combined_df = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
        self.log.info("MCMC chains completed: %d chains", len(chains))
        end_state = {"regime": regime, "mid": mid, "ts_event": ts_event, "order_id": order_id}
        return chains, combined_df, end_state
