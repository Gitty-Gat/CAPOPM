"""
Principled ML prior construction for the flat prior simulator.

The ML model outputs (mu, N_eff) which are mapped to a Beta distribution:
    alpha_ML = N_eff * mu
    beta_ML  = N_eff * (1 - mu)

Uncertainty is estimated from either an ensemble of lightweight logistic
models or Monte-Carlo dropout. Ensemble variance is converted to N_eff using
the Beta variance identity: Var(p) = mu(1-mu)/(N_eff + 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass
class MLPriorConfig:
    model_type: str = "ensemble_logistic"  # or "mlp_mc_dropout"
    n_models: int = 10
    N_eff_min: float = 2.0
    N_eff_max: float = 500.0
    l2: float = 1.0
    lookback_events: int = 500


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_features(
    events: pd.DataFrame,
    lookback_events: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """
    Compute interpretable microstructure features from the last `lookback_events`.

    Features:
    - best_bid, best_ask, mid
    - spread
    - bid_sz1, ask_sz1
    - imbalance
    - order_flow_imbalance (OFI)
    - microprice
    - mid_return_vol
    - event_intensity (events / minute)
    """

    window = events.tail(lookback_events).copy()
    if window.empty:
        zeros = np.zeros(10, dtype=np.float64)
        return zeros, [
            "best_bid",
            "best_ask",
            "mid",
            "spread",
            "bid_sz1",
            "ask_sz1",
            "imbalance",
            "order_flow_imbalance",
            "microprice",
            "mid_return_vol",
            "event_intensity",
        ], {}

    # Reconstruct a lightweight level-1 book.
    bids: Dict[int, Dict[str, float]] = {}
    asks: Dict[int, Dict[str, float]] = {}
    ofi = 0.0
    mid_prices: List[float] = []

    for _, row in window.iterrows():
        action = row["action"]
        side = row["side"]
        oid = int(row["order_id"])
        price = float(row["price"])
        size = float(row["size"])

        book = bids if side == "B" else asks

        if action == "A":  # add
            book[oid] = {"price": price, "size": size}
            ofi += size if side == "B" else -size
        elif action == "M":  # modify
            if oid in book:
                prev = book[oid]
                delta = size - prev["size"]
                book[oid] = {"price": price, "size": size}
                ofi += delta if side == "B" else -delta
        elif action == "C":  # cancel
            if oid in book:
                prev = book.pop(oid)
                ofi -= prev["size"] if side == "B" else -prev["size"]
        elif action == "F":  # fill
            if oid in book:
                prev = book[oid]
                remaining = max(prev["size"] - size, 0.0)
                delta = size
                if remaining <= EPS:
                    book.pop(oid, None)
                else:
                    book[oid]["size"] = remaining
                ofi -= delta if side == "B" else -delta
        # Track mid after each event
        best_bid = max((v["price"] for v in bids.values()), default=np.nan)
        best_ask = min((v["price"] for v in asks.values()), default=np.nan)
        if np.isfinite(best_bid) and np.isfinite(best_ask):
            mid = 0.5 * (best_bid + best_ask)
            mid_prices.append(mid)

    best_bid = max((v["price"] for v in bids.values()), default=np.nan)
    best_ask = min((v["price"] for v in asks.values()), default=np.nan)
    mid = np.nan
    if np.isfinite(best_bid) and np.isfinite(best_ask):
        mid = 0.5 * (best_bid + best_ask)
    spread = (best_ask - best_bid) if np.isfinite(best_bid) and np.isfinite(best_ask) else np.nan
    bid_sz1 = max((v["size"] for v in bids.values()), default=np.nan)
    ask_sz1 = max((v["size"] for v in asks.values()), default=np.nan)
    imbalance = (bid_sz1 - ask_sz1) / (bid_sz1 + ask_sz1 + EPS) if np.isfinite(bid_sz1) and np.isfinite(ask_sz1) else 0.0
    microprice = (
        (best_ask * bid_sz1 + best_bid * ask_sz1) / (bid_sz1 + ask_sz1 + EPS)
        if np.isfinite(best_bid) and np.isfinite(best_ask) and np.isfinite(bid_sz1) and np.isfinite(ask_sz1)
        else np.nan
    )

    mid_return_vol = 0.0
    if len(mid_prices) > 2:
        mid_arr = np.array(mid_prices)
        returns = np.diff(np.log(mid_arr + EPS))
        mid_return_vol = float(np.std(returns))

    duration_ns = float(window["ts_event"].iloc[-1] - window["ts_event"].iloc[0])
    events_per_min = len(window) / (duration_ns / 1e9 / 60.0 + EPS)

    features = np.array(
        [
            best_bid if np.isfinite(best_bid) else 0.0,
            best_ask if np.isfinite(best_ask) else 0.0,
            mid if np.isfinite(mid) else 0.0,
            spread if np.isfinite(spread) else 0.0,
            bid_sz1 if np.isfinite(bid_sz1) else 0.0,
            ask_sz1 if np.isfinite(ask_sz1) else 0.0,
            imbalance,
            ofi,
            microprice if np.isfinite(microprice) else 0.0,
            mid_return_vol,
            events_per_min,
        ],
        dtype=np.float64,
    )

    names = [
        "best_bid",
        "best_ask",
        "mid",
        "spread",
        "bid_sz1",
        "ask_sz1",
        "imbalance",
        "order_flow_imbalance",
        "microprice",
        "mid_return_vol",
        "event_intensity",
    ]

    logger.debug("ML prior features computed: %s", dict(zip(names, features.tolist())))
    diagnostics = {name: float(val) for name, val in zip(names, features)}
    diagnostics["n_events_used"] = int(len(window))
    return features, names, diagnostics


class PrincipledMLPrior:
    """Produces (mu, N_eff) and Beta parameters from interpretable features."""

    def __init__(self, cfg: MLPriorConfig, logger: Optional[logging.Logger] = None, seed: int = 0):
        self.cfg = cfg
        self.log = logger or logging.getLogger(__name__)
        self.rng = np.random.default_rng(seed)

        if cfg.model_type not in {"ensemble_logistic", "mlp_mc_dropout"}:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")

        # Initialize lightweight model parameters.
        input_dim = 11
        self.weights = self.rng.normal(loc=0.0, scale=0.05, size=(cfg.n_models, input_dim))
        self.bias = self.rng.normal(loc=0.0, scale=0.02, size=(cfg.n_models,))

    def _predict_ensemble(self, features: np.ndarray) -> np.ndarray:
        logits = self.weights @ features + self.bias
        return _sigmoid(logits)

    def _predict_mc_dropout(self, features: np.ndarray, p_drop: float = 0.1) -> np.ndarray:
        preds = []
        for i in range(self.cfg.n_models):
            mask = self.rng.binomial(1, 1.0 - p_drop, size=features.shape).astype(np.float64)
            logits = (self.weights[i] * mask) @ features + self.bias[i]
            preds.append(_sigmoid(logits))
        return np.array(preds, dtype=np.float64)

    def predict_beta(self, events: pd.DataFrame) -> Tuple[float, float, Dict[str, float]]:
        """Return (alpha_ml, beta_ml, diagnostics)."""

        features, names, feat_diag = compute_features(events, self.cfg.lookback_events, self.log)

        if self.cfg.model_type == "ensemble_logistic":
            preds = self._predict_ensemble(features)
        else:
            preds = self._predict_mc_dropout(features)

        mu = float(np.mean(preds))
        variance = float(np.var(preds))
        mu = min(max(mu, EPS), 1.0 - EPS)

        if variance < EPS:
            N_eff = self.cfg.N_eff_max
        else:
            implied = mu * (1 - mu) / (variance + EPS) - 1.0
            N_eff = float(np.clip(implied, self.cfg.N_eff_min, self.cfg.N_eff_max))

        alpha_ml = N_eff * mu
        beta_ml = N_eff * (1.0 - mu)

        diagnostics = {
            "mu": mu,
            "variance": variance,
            "N_eff": N_eff,
            "alpha_ml": alpha_ml,
            "beta_ml": beta_ml,
            "model_type": self.cfg.model_type,
            "n_models": self.cfg.n_models,
            "features_used": names,
        }
        diagnostics.update(feat_diag)
        self.log.info("ML prior: mu=%.4f var=%.4f N_eff=%.2f", mu, variance, N_eff)
        return alpha_ml, beta_ml, diagnostics

