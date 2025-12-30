"""
Phase 7 metrics sanity smoke test for calibration ECE and coverage.

Run:
  python scripts/smoke_test_metrics_phase7.py
"""

from __future__ import annotations

import os
import sys

# Allow running from repo root without installation.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.runner import run_experiment


def build_config() -> dict:
    """Minimal experiment config with fixed p_true for calibration sanity."""

    return {
        "seed": 202507,
        "n_runs": 200,
        "p_true_dist": {"type": "fixed", "value": 0.55},
        "scenario_name": "smoke_metrics_phase7",
        "models": [
            "capopm",
            "raw_parimutuel",
            "structural_only",
            "ml_only",
            "uncorrected",
        ],
        "traders": {
            "n_traders": 40,
            "proportions": {"informed": 0.5, "noise": 0.5},
            "params": {
                "informed": {
                    "signal_quality": 0.7,
                    "noise_yes_prob": 0.5,
                    "herding_intensity": 0.0,
                },
                "noise": {
                    "signal_quality": 0.6,
                    "noise_yes_prob": 0.5,
                    "herding_intensity": 0.0,
                },
            },
        },
        "market": {
            "n_steps": 8,
            "arrivals_per_step": 3,
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
            "noise_std": 0.005,
            "calibration": 1.0,
            "r_ml": 0.8,
        },
        "prior_cfg": {"n_str": 10.0, "n_ml_eff": 4.0, "n_ml_scale": 1.0},
        "stage1_cfg": {"enabled": False},
        "stage2_cfg": {"enabled": False},
        "ece_bins": 10,
        "calibration_binning": "equal_width",
        "ece_min_nonempty_bins": 9,
        "coverage_include_outcome": False,
    }


def main() -> None:
    config = build_config()
    results = run_experiment(config)
    capopm_metrics = results["aggregated_metrics"]["capopm"]
    diag = capopm_metrics["calibration_diagnostics"]

    ece = capopm_metrics["calibration_ece"]
    coverage_95 = capopm_metrics["coverage_95"]
    assert "coverage_90_outcome" in capopm_metrics
    assert "coverage_95_outcome" in capopm_metrics
    assert "coverage_90_ptrue" in capopm_metrics
    assert "coverage_95_ptrue" in capopm_metrics
    assert "calib_n_nonempty_bins" in capopm_metrics
    assert "calib_binning_mode_used" in capopm_metrics

    assert ece is not None
    assert ece < 0.2, f"Calibration ECE unexpectedly high: {ece:.4f}"
    assert coverage_95 is not None
    assert 0.0 < coverage_95 < 1.0, f"Coverage_95 out of expected range: {coverage_95:.4f}"
    assert diag.get("binning_mode_used") == "equal_mass", "Fallback to equal_mass binning not triggered"
    assert diag.get("fallback_applied") is True, "Fallback flag missing or False"
    assert diag.get("n_nonempty_bins", 0) >= config["ece_min_nonempty_bins"], "Fallback did not restore bin coverage"

    print("SMOKE TEST PASSED: Phase 7 metrics sanity")
    print(f"ECE (capopm): {ece:.4f}")
    print(f"Coverage_95 (capopm): {coverage_95:.4f}")
    print(f"Binning mode used: {diag.get('binning_mode_used')}")
    # Validate reliability CSV has scenario and model columns
    rel_path = os.path.join("results", config["scenario_name"], "reliability_capopm.csv")
    import csv

    with open(rel_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert "scenario_name" in row and row["scenario_name"] == config["scenario_name"]
        assert "model" in row and row["model"] == "capopm"


if __name__ == "__main__":
    main()
