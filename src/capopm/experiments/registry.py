"""
Experiment registry and dispatcher for CAPOPM empirical validation.

Allows running experiments by name without modifying core pipeline logic.
"""

from __future__ import annotations

from typing import Callable, Dict

from .a1_info_efficiency_curves import run_a1_info_efficiency_curves
from .a2_time_to_converge import run_a2_time_to_converge
from .a3_strategic_timing_attack import run_a3_strategic_timing_attack
from .b1_correction_no_regret import run_b1_correction_no_regret
from .b2_asymptotic_rate_check import run_b2_asymptotic_rate_check
from .b3_misspecification_regret_grid import run_b3_misspecification_regret_grid
from .b4_regime_posterior_concentration import run_b4_regime_posterior_concentration
from .b5_arbitrage_projection_impact import run_b5_arbitrage_projection_impact

ExperimentRunner = Callable[..., object]

EXPERIMENT_REGISTRY: Dict[str, ExperimentRunner] = {
    "A1.INFO_EFFICIENCY_CURVES": run_a1_info_efficiency_curves,
    "A1": run_a1_info_efficiency_curves,
    "INFO_EFFICIENCY_CURVES": run_a1_info_efficiency_curves,
    "A2.TIME_TO_CONVERGE": run_a2_time_to_converge,
    "A2": run_a2_time_to_converge,
    "TIME_TO_CONVERGE": run_a2_time_to_converge,
    "A3.STRATEGIC_TIMING_ATTACK": run_a3_strategic_timing_attack,
    "A3": run_a3_strategic_timing_attack,
    "STRATEGIC_TIMING_ATTACK": run_a3_strategic_timing_attack,
    "B1.CORRECTION_NO_REGRET": run_b1_correction_no_regret,
    "B1": run_b1_correction_no_regret,
    "CORRECTION_NO_REGRET": run_b1_correction_no_regret,
    "B2.ASYMPTOTIC_RATE_CHECK": run_b2_asymptotic_rate_check,
    "B2": run_b2_asymptotic_rate_check,
    "ASYMPTOTIC_RATE_CHECK": run_b2_asymptotic_rate_check,
    "B3.MISSPECIFICATION_REGRET_GRID": run_b3_misspecification_regret_grid,
    "B3": run_b3_misspecification_regret_grid,
    "MISSPECIFICATION_REGRET_GRID": run_b3_misspecification_regret_grid,
    "B4.REGIME_POSTERIOR_CONCENTRATION": run_b4_regime_posterior_concentration,
    "B4": run_b4_regime_posterior_concentration,
    "REGIME_POSTERIOR_CONCENTRATION": run_b4_regime_posterior_concentration,
    "B5.ARBITRAGE_PROJECTION_IMPACT": run_b5_arbitrage_projection_impact,
    "B5": run_b5_arbitrage_projection_impact,
    "ARBITRAGE_PROJECTION_IMPACT": run_b5_arbitrage_projection_impact,
}


def get_experiment_runner(name: str) -> ExperimentRunner:
    """Return the runner function for a registered experiment name."""

    key = name.upper()
    if key in EXPERIMENT_REGISTRY:
        return EXPERIMENT_REGISTRY[key]
    # Allow fully qualified experiment_id lookup without altering case.
    if name in EXPERIMENT_REGISTRY:
        return EXPERIMENT_REGISTRY[name]
    raise KeyError(f"Experiment '{name}' is not registered.")
