"""
Tier C1: ZERO_INFORMATION_MARKET

Descriptive stress run with non-informative trades; uses standard runner outputs.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List

from .runner import run_experiment
from .b1_correction_no_regret import build_base_config as b1_base_config

EXPERIMENT_ID = "C1.ZERO_INFORMATION_MARKET"
TIER = "C"
DEFAULT_BASE_SEED = 302000
DEFAULT_N_RUNS = 30


def build_base_config() -> Dict:
    cfg = b1_base_config()
    cfg["p_true_dist"] = {"type": "fixed", "value": 0.5}
    cfg["models"] = [
        "capopm",
        "raw_parimutuel",
        "uncorrected",
        "structural_only",
        "ml_only",
    ]
    cfg["traders"] = {
        "n_traders": 60,
        "proportions": {
            "informed": 0.0,
            "adversarial": 0.0,
            "noise": 1.0,
        },
        "params": {
            "noise": {
                "signal_quality": 0.5,
                "noise_yes_prob": 0.5,
                "herding_intensity": 0.0,
            },
        },
    }
    cfg["stage2_cfg"] = {"enabled": False}
    cfg["ml_cfg"]["r_ml"] = 0.5
    cfg["ml_cfg"]["bias"] = 0.0
    cfg["ml_cfg"]["base_prob"] = 0.5
    cfg["prior_cfg"]["n_ml_eff"] = 4.0
    cfg["prior_cfg"]["n_ml_scale"] = 1.0
    return copy.deepcopy(cfg)


def run_c1_zero_information_market(
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
) -> List[Dict]:
    scenario_name = f"C1_zero_info__seed{base_seed}"
    cfg = build_base_config()
    cfg["seed"] = int(base_seed)
    cfg["n_runs"] = int(n_runs)
    cfg["scenario_name"] = scenario_name
    cfg["experiment_id"] = EXPERIMENT_ID
    cfg["tier"] = TIER
    cfg["sweep_params"] = {"information_level": "zero"}

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
