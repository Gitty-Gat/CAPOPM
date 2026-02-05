"""
Flat structural prior for the exploratory simulator.

Fixed Beta(1,1) = Uniform prior. Only used inside the new simulator path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FlatStructuralPrior:
    """Beta(1,1) structural prior."""

    alpha: float = 1.0
    beta: float = 1.0

    def params(self) -> Tuple[float, float]:
        return self.alpha, self.beta

