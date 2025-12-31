"""
Tier A Experiment A1: INFO_EFFICIENCY_CURVES.

Implements the sweep over signal_quality x informed_share x adversarial_share
to evaluate information aggregation efficiency. Uses existing Phase 7 runner
to compute metrics and write standardized artifacts.
"""

from __future__ import annotations

import copy
import itertools
from typing import Dict, Iterable, List, Tuple

from .runner import run_experiment

EXPERIMENT_ID = "A1.INFO_EFFICIENCY_CURVES"
TIER = "A"

# Default sweep grid for signal quality and trader composition.
DEFAULT_SIGNAL_QUALITY = [0.60, 0.75, 0.90]
DEFAULT_INFORMED_SHARE = [0.20, 0.50]
DEFAULT_ADVERSARIAL_SHARE = [0.00, 0.10, 0.20]

# Deterministic seed offset to allow reproducible sweeps.
DEFAULT_BASE_SEED = 202610
# Default number of Phase 7 runs per sweep point (can be overridden by caller).
DEFAULT_N_RUNS = 150


def run_a1_info_efficiency_curves(
    signal_quality_grid: Iterable[float] = DEFAULT_SIGNAL_QUALITY,
    informed_share_grid: Iterable[float] = DEFAULT_INFORMED_SHARE,
    adversarial_share_grid: Iterable[float] = DEFAULT_ADVERSARIAL_SHARE,
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
) -> List[Dict]:
    """Run A1 sweep deterministically across the specified grids."""

    sweep = list(
        itertools.product(
            list(signal_quality_grid),
            list(informed_share_grid),
            list(adversarial_share_grid),
        )
    )

    results = []
    for idx, (rho, inf_share, adv_share) in enumerate(sweep):
        noise_share = 1.0 - inf_share - adv_share
        if noise_share <= 0.0:
            continue  # skip infeasible compositions

        seed = base_seed + idx
        scenario_name = scenario_id(rho, inf_share, adv_share, seed)
        cfg = build_base_config()
        cfg["seed"] = seed
        cfg["n_runs"] = n_runs
        cfg["scenario_name"] = scenario_name
        cfg["experiment_id"] = EXPERIMENT_ID
        cfg["tier"] = TIER
        cfg["sweep_params"] = {
            "signal_quality": rho,
            "informed_share": inf_share,
            "adversarial_share": adv_share,
            "noise_share": noise_share,
        }

        cfg["traders"]["proportions"] = {
            "informed": inf_share,
            "adversarial": adv_share,
            "noise": noise_share,
        }
        cfg["traders"]["params"]["informed"]["signal_quality"] = rho
        cfg["traders"]["params"]["adversarial"]["signal_quality"] = rho

        run_experiment(cfg)
        results.append(
            {
                "scenario_name": scenario_name,
                "seed": seed,
                "sweep_params": cfg["sweep_params"],
            }
        )

    return results


def scenario_id(rho: float, informed: float, adversarial: float, seed: int) -> str:
    """Stable scenario name encoding sweep settings."""

    def pct(x: float) -> int:
        return int(round(100 * x))

    return f"a1_info_eff_q{pct(rho)}_inf{pct(informed)}_adv{pct(adversarial)}_seed{seed}"


def build_base_config() -> Dict:
    """Base configuration for A1 with Stage 1 + Stage 2 enabled (defaults)."""

    cfg = {
        "p_true_dist": {"type": "beta", "a": 2.0, "b": 2.0},
        "models": [
            "capopm",
            "raw_parimutuel",
            "structural_only",
            "ml_only",
            "uncorrected",
            "beta_1_1",
            "beta_0_5_0_5",
        ],
        "traders": {
            "n_traders": 60,
            "proportions": {
                "informed": 0.4,
                "adversarial": 0.1,
                "noise": 0.5,
            },
            "params": {
                "informed": {
                    "signal_quality": 0.7,
                    "noise_yes_prob": 0.5,
                    "herding_intensity": 0.0,
                },
                "adversarial": {
                    "signal_quality": 0.7,
                    "noise_yes_prob": 0.5,
                    "herding_intensity": 0.0,
                },
                "noise": {
                    "signal_quality": 0.5,
                    "noise_yes_prob": 0.5,
                    "herding_intensity": 0.0,
                },
            },
        },
        "market": {
            "n_steps": 12,
            "arrivals_per_step": 5,
            "fee_rate": 0.01,
            "initial_yes_pool": 1.0,
            "initial_no_pool": 1.0,
            "signal_model": "conditional_on_state",
            "use_realized_state_for_signals": True,
            "herding_enabled": False,
            "size_dist": "fixed",
            "size_dist_params": {"size": 1.0},
        },
        "structural_cfg": {
            "T": 1.0,
            "K": 1.0,
            "S0": 1.0,
            "V0": 0.04,
            "kappa": 1.0,
            "theta": 0.04,
            "xi": 0.2,
            "rho": -0.3,
            "alpha": 0.7,
            "lambda": 0.1,
        },
        "ml_cfg": {
            "base_prob": 0.55,
            "bias": -0.02,
            "noise_std": 0.01,
            "calibration": 1.0,
            "r_ml": 0.8,
        },
        "prior_cfg": {"n_str": 10.0, "n_ml_eff": 4.0, "n_ml_scale": 1.0},
        "stage1_cfg": {
            "enabled": True,
            "w_min": 0.25,
            "w_max": 1.25,
            "longshot_ref_p": 0.5,
            "longshot_gamma": 0.8,
            "herding_lambda": 0.15,
            "herding_window": 25,
        },
        "stage2_cfg": {
            "enabled": True,
            "mode": "offsets_mixture",
            "delta_plus": 0.0,
            "delta_minus": 0.0,
            "regimes": [
                {"pi": 0.5, "g_plus_scale": 0.05, "g_minus_scale": 0.05},
                {"pi": 0.5, "g_plus_scale": -0.05, "g_minus_scale": -0.05},
            ],
        },
        "ece_bins": 10,
        "calibration_binning": "equal_width",
        "ece_min_nonempty_bins": 5,
        "coverage_include_outcome": False,
    }

    return copy.deepcopy(cfg)
