"""
Hybrid mixture prior combining a principled ML Beta and a flat Beta(1,1).

Posterior updates:
- Each component Beta is updated conjugately with (y, n - y).
- Mixture weights are updated via Bayes rule using the Beta-binomial evidence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np

from .flat_structural_prior import FlatStructuralPrior


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_binom_log_evidence(alpha: float, beta: float, y: int, n: int) -> float:
    return _log_beta(alpha + y, beta + n - y) - _log_beta(alpha, beta)


@dataclass
class MixturePriorState:
    alpha_ml: float
    beta_ml: float
    alpha_flat: float
    beta_flat: float
    weight: float  # weight on ML component

    def normalize_weight(self) -> "MixturePriorState":
        w = min(max(self.weight, 0.0), 1.0)
        return replace(self, weight=w)

    def posterior_mean(self) -> float:
        mean_ml = self.alpha_ml / (self.alpha_ml + self.beta_ml)
        mean_flat = self.alpha_flat / (self.alpha_flat + self.beta_flat)
        return self.weight * mean_ml + (1.0 - self.weight) * mean_flat

    def sample(self, draws: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the mixture posterior."""
        w = self.weight
        component_choices = rng.uniform(size=draws) < w
        samples = np.empty(draws, dtype=np.float64)
        n_ml = component_choices.sum()
        n_flat = draws - n_ml
        if n_ml > 0:
            samples[component_choices] = rng.beta(self.alpha_ml, self.beta_ml, size=n_ml)
        if n_flat > 0:
            samples[~component_choices] = rng.beta(self.alpha_flat, self.beta_flat, size=n_flat)
        return samples


class HybridMixturePrior:
    def __init__(self, weight: float, logger: logging.Logger):
        self.log = logger
        self.w0 = float(weight)
        if self.w0 < 0.0 or self.w0 > 1.0:
            raise ValueError("mixture_weight_w must be in [0,1]")
        self.structural = FlatStructuralPrior()

    def initialize(self, alpha_ml: float, beta_ml: float) -> MixturePriorState:
        alpha_flat, beta_flat = self.structural.params()
        state = MixturePriorState(
            alpha_ml=alpha_ml,
            beta_ml=beta_ml,
            alpha_flat=alpha_flat,
            beta_flat=beta_flat,
            weight=self.w0,
        ).normalize_weight()
        self.log.info(
            "Initialized mixture prior: w=%.2f | ML=(%.2f, %.2f) flat=(%.2f, %.2f)",
            state.weight,
            alpha_ml,
            beta_ml,
            alpha_flat,
            beta_flat,
        )
        return state

    def update(self, state: MixturePriorState, y: int, n: int) -> MixturePriorState:
        """Conjugate updates + Bayes weight update."""
        alpha_ml_post = state.alpha_ml + y
        beta_ml_post = state.beta_ml + (n - y)
        alpha_flat_post = state.alpha_flat + y
        beta_flat_post = state.beta_flat + (n - y)

        log_like_ml = _beta_binom_log_evidence(state.alpha_ml, state.beta_ml, y, n)
        log_like_flat = _beta_binom_log_evidence(state.alpha_flat, state.beta_flat, y, n)

        # Convert to probabilities in log-space.
        max_log = max(log_like_ml, log_like_flat)
        weight_ml = state.weight * math.exp(log_like_ml - max_log)
        weight_flat = (1.0 - state.weight) * math.exp(log_like_flat - max_log)
        denom = weight_ml + weight_flat
        if denom <= 0.0:
            new_w = state.weight
        else:
            new_w = weight_ml / denom

        new_state = MixturePriorState(
            alpha_ml=alpha_ml_post,
            beta_ml=beta_ml_post,
            alpha_flat=alpha_flat_post,
            beta_flat=beta_flat_post,
            weight=new_w,
        ).normalize_weight()

        self.log.debug(
            "Mixture update: y=%d n=%d w=%.3f->%.3f alpha_ml=%.2f beta_ml=%.2f",
            y,
            n,
            state.weight,
            new_w,
            new_state.alpha_ml,
            new_state.beta_ml,
        )
        return new_state

    def diagnostics(self, state: MixturePriorState) -> Dict[str, float]:
        return {
            "weight": state.weight,
            "alpha_ml": state.alpha_ml,
            "beta_ml": state.beta_ml,
            "alpha_flat": state.alpha_flat,
            "beta_flat": state.beta_flat,
            "posterior_mean": state.posterior_mean(),
        }

