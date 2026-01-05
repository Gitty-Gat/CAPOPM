"""
Tier C2: ADVERSARY_MAJORITARIAN

Descriptive stress run with adversarial dominance; uses standard runner outputs.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List

from .runner import run_experiment

EXPERIMENT_ID = "C2.ADVERSARY_MAJORITARIAN"
TIER = "C"
DEFAULT_BASE_SEED = 303000
DEFAULT_N_RUNS = 30


def build_base_config() -> Dict:
    cfg = {
        "p_true_dist": {"type": "beta", "a": 2.0, "b": 2.0},
        "models": [
            "capopm",
            "uncorrected",
            "raw_parimutuel",
            "structural_only",
            "ml_only",
        ],
        "traders": {
            "n_traders": 60,
            "proportions": {
                "informed": 0.1,
                "adversarial": 0.7,
                "noise": 0.2,
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
            "n_steps": 30,
            "arrivals_per_step": 2,
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
            "w_min": 0.1,
            "w_max": 1.25,
            "longshot_ref_p": 0.5,
            "longshot_gamma": 0.8,
            "herding_lambda": 0.8,
            "herding_window": 50,
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


def run_c2_adversary_majoritarian(
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
) -> List[Dict]:
    scenario_name = f"C2_adversary_majoritarian__seed{base_seed}"
    cfg = build_base_config()
    cfg["seed"] = int(base_seed)
    cfg["n_runs"] = int(n_runs)
    cfg["scenario_name"] = scenario_name
    cfg["experiment_id"] = EXPERIMENT_ID
    cfg["tier"] = TIER
    cfg["sweep_params"] = {"adversarial_dominance": 0.7}
    cfg["metadata"] = {"adversarial": True}

    run_experiment(cfg)
    return [
        {
            "scenario_name": scenario_name,
            "seed": base_seed,
            "base_seed": base_seed,
            "run_index": 0,
            "sweep_params": cfg["sweep_params"],
        }
    ]
