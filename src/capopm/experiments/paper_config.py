"""
Centralized paper-run thresholds and harness defaults.

These values govern audit gates and paper-ready runs only; no core math
or DGP logic is altered. Adjust here (single source of truth).
"""

from __future__ import annotations

# Minimum runs per scenario to claim paper readiness.
MIN_RUNS_PER_CELL = 30
# Minimum distinct grid points when a claim depends on a sweep.
MIN_GRID_POINTS = 2
# Calibration interpretability gates.
CALIB_MIN_UNIQUE = 5
CALIB_MIN_NONEMPTY_BINS = 5
CALIB_MIN_SAMPLES = 10
# Coverage tolerances (absolute deviation from nominal).
COVERAGE_TOLERANCE = 0.1
# Extreme-p slice definition.
EXTREME_P_THRESHOLD = 0.1
# Paper-ready classification thresholds.
PAPER_READY_MIN_RUNS = 30
PAPER_READY_MIN_GRID = 2
SMOKE_RUN_THRESHOLD = 5
# Bootstrap parameters for effect sizes.
EFFECT_BOOTSTRAP_SAMPLES = 1000
EFFECT_ALPHA = 0.05

# Paper-suite grids (lightweight but >= MIN_GRID_POINTS).
PAPER_GRIDS = {
    "A1": {
        "signal_quality_grid": [0.65, 0.85],
        "informed_share_grid": [0.30, 0.50],
        "adversarial_share_grid": [0.00, 0.20],
        "base_seed": 99101,
    },
    "A2": {
        "arrival_grid": [1, 3],
        "pool_grid": [1.0, 2.0],
        "steps_grid": [25, 40],
        "base_seed": 240000,
    },
    "A3": {
        "attack_strength_grid": [0.0, 1.0],
        "window_grid": [1.0],
        "scale_grid": [1, 3],
        "base_seed": 250000,
    },
    "B1": {
        "longshot_bias_grid": [0, 1],
        "herding_grid": [0, 1],
        "timing_attack_grid": [0.0, 1.0],
        "liquidity_level_grid": ["low", "high"],
        "base_seed": 202640,
    },
    "B2": {
        "base_seed": 260000,
        "n_total_grid": [100, 400],
    },
    "B3": {
        "structural_mis_grid": [0, 10],
        "ml_mis_grid": [0, 5],
        "base_seed": 270000,
    },
    "B4": {
        "evidence_grid": [50, 150],
        "base_seed": 280000,
    },
    "B5": {
        "violation_strength_grid": [0.0, 0.5, 1.0],
        "projection_method": "euclidean",
        "base_seed": 290000,
    },
}

# Reproduction command template for paper suite.
PAPER_REPRO_CMD = "python run_paper_suite.py"
