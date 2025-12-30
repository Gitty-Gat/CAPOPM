"""
Trader information structure and decision rules for CAPOPM.

Implements Phase 3 (Asymmetric Information Model) from the canonical paper:
- Assumptions A1–A6 (risk neutrality, price taking, common prior, private signals)
- Binary signals and separating strategies (YES on q1, NO on q0)

Optional Phase 7 herding-style dependence (OFF by default):
P(s_t = YES | F_{t-1}) = (1 - lambda) * p_base + lambda * p_hist
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import math


TraderType = str  # "informed", "noise", "adversarial"


@dataclass(frozen=True)
class TraderParams:
    """Parameters controlling a trader's signal quality and behavior.

    signal_quality (rho): P(q1 | theta=1) = rho, P(q1 | theta=0) = 1 - rho (Phase 3).
    noise_yes_prob: P(YES) for noise traders (Phase 7 uses 1/2).
    herding_intensity: lambda in Phase 7 herding formula.
    """

    signal_quality: float = 0.7
    noise_yes_prob: float = 0.5
    herding_intensity: float = 0.0


@dataclass(frozen=True)
class Trader:
    trader_id: int
    trader_type: TraderType
    params: TraderParams

    def decide(
        self,
        rng: np.random.Generator,
        p_true: float,
        p_hist: Optional[float],
        signal_model: str,
        realized_state: Optional[int],
        herding_enabled: bool,
    ) -> str:
        """Return "YES" or "NO" action for this trader.

        signal_model:
          - "bernoulli_p_true": signal ~ Bernoulli(p_true) (Phase 7 simplified)
          - "conditional_on_state": signal depends on realized state via rho (Phase 3)
        """

        base_yes_prob = self._base_yes_probability(
            rng=rng,
            p_true=p_true,
            signal_model=signal_model,
            realized_state=realized_state,
        )

        if herding_enabled and self.params.herding_intensity > 0 and p_hist is not None:
            base_yes_prob = apply_herding(
                base_yes_prob, p_hist, self.params.herding_intensity
            )

        return "YES" if float(rng.random()) < base_yes_prob else "NO"

    def _base_yes_probability(
        self,
        rng: np.random.Generator,
        p_true: float,
        signal_model: str,
        realized_state: Optional[int],
    ) -> float:
        if self.trader_type == "noise":
            return self.params.noise_yes_prob

        if signal_model == "bernoulli_p_true":
            signal = 1 if float(rng.random()) < p_true else 0
        elif signal_model == "conditional_on_state":
            if realized_state is None:
                raise ValueError(
                    "realized_state is required for signal_model=conditional_on_state"
                )
            rho = self.params.signal_quality
            if realized_state == 1:
                signal = 1 if float(rng.random()) < rho else 0
            else:
                signal = 1 if float(rng.random()) < (1.0 - rho) else 0
        else:
            raise ValueError(f"Unknown signal_model: {signal_model}")

        if self.trader_type == "informed":
            return 1.0 if signal == 1 else 0.0
        if self.trader_type == "adversarial":
            return 0.0 if signal == 1 else 1.0

        raise ValueError(f"Unknown trader_type: {self.trader_type}")


def apply_herding(base_prob: float, p_hist: float, intensity: float) -> float:
    """Phase 7 herding rule: mix base belief with empirical order flow."""

    if intensity < 0.0 or intensity > 1.0:
        raise ValueError("herding_intensity must be in [0, 1]")
    return (1.0 - intensity) * base_prob + intensity * p_hist


def build_traders(
    n_traders: int,
    proportions: Dict[TraderType, float],
    params_by_type: Optional[Dict[TraderType, TraderParams]] = None,
) -> List[Trader]:
    """Create a trader population following Phase 7 types (informed/noise/adversarial).

    Allocation is deterministic (no RNG) for reproducibility across runs.
    """

    if n_traders <= 0:
        raise ValueError("n_traders must be positive")

    if params_by_type is None:
        params_by_type = {}

    def get_params(t_type: TraderType) -> TraderParams:
        return params_by_type.get(t_type, TraderParams())

    # Normalize proportions to sum to 1
    total = sum(proportions.values())
    if total <= 0:
        raise ValueError("proportions must sum to a positive value")
    weights = {k: v / total for k, v in proportions.items()}

    # Allocate counts by rounding, then adjust to match n_traders
    counts = {k: int(round(weights[k] * n_traders)) for k in weights}
    delta = n_traders - sum(counts.values())
    if delta != 0:
        # Assign any remainder to the largest proportion
        largest = max(weights.items(), key=lambda kv: kv[1])[0]
        counts[largest] += delta
    if any(v < 0 for v in counts.values()) or sum(counts.values()) != n_traders:
        # Deterministic fallback: floor then distribute remainder to largest weights.
        counts = {k: int(math.floor(weights[k] * n_traders)) for k in weights}
        remainder = n_traders - sum(counts.values())
        if remainder > 0:
            ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
            idx = 0
            while remainder > 0:
                counts[ordered[idx % len(ordered)][0]] += 1
                remainder -= 1
                idx += 1
        assert all(v >= 0 for v in counts.values())
        assert sum(counts.values()) == n_traders

    traders: List[Trader] = []
    trader_id = 0
    for t_type, count in counts.items():
        for _ in range(count):
            traders.append(Trader(trader_id=trader_id, trader_type=t_type, params=get_params(t_type)))
            trader_id += 1

    return traders


def empirical_yes_rate(history: List[str]) -> Optional[float]:
    """Compute empirical YES fraction up to t-1; returns None if no history."""

    if not history:
        return None
    yes = sum(1 for s in history if s == "YES")
    return yes / len(history)
