"""
Audit contracts and theorem/proposition mappings for CAPOPM experiments.

This module encodes **only** audit-level assertions and publication-facing
checklists. It does **not** modify any CAPOPM math, priors, likelihoods, or
simulation logic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

AuditCriterion = Dict[str, object]
AuditContract = Dict[str, object]


def _grid_requirement(description: str, theorem: Optional[str] = None) -> AuditCriterion:
    """Helper to declare that a claim cannot be evaluated without a sweep/grid."""

    return {
        "id": "grid_requirement",
        "requires_grid": True,
        "description": description,
        "theorem": theorem,
    }


AUDIT_CONTRACTS: Dict[str, AuditContract] = {
    "A1.INFO_EFFICIENCY_CURVES": {
        "theorems": [
            {"id": "Proposition 6", "claim": "Information efficiency vs signal quality"},
            {"id": "Lemma 3", "claim": "Aggregation under informed share sweeps"},
        ],
        "criteria": [
            {
                "id": "capopm_dominates_raw_parimutuel_brier",
                "metric_path": ("aggregated_metrics", "capopm", "brier"),
                "comparator_path": ("aggregated_metrics", "raw_parimutuel", "brier"),
                "direction": "<=",
                "description": "CAPOPM Brier should not exceed raw parimutuel (Prop 6 support).",
                "theorem": "Proposition 6",
            },
            _grid_requirement(
                "Monotonic info-efficiency curve requires multi-point signal quality sweep.",
                theorem="Proposition 6",
            ),
        ],
        "reproduction_command": "python run_paper_suite.py --experiment A1",
    },
    "A2.TIME_TO_CONVERGE": {
        "theorems": [
            {"id": "Phase 4 consistency", "claim": "Higher liquidity accelerates stabilization"},
        ],
        "criteria": [
            {
                "id": "variance_decay_negative",
                "metric_path": ("aggregated_metrics", "capopm", "var_decay_slope"),
                "threshold": 0.0,
                "direction": "<=",
                "description": "Posterior variance slope vs steps should be non-positive.",
                "theorem": "Phase 4 consistency",
            },
            _grid_requirement(
                "Convergence comparisons need multiple liquidity/arrival settings.",
                theorem="Phase 4 consistency",
            ),
        ],
        "reproduction_command": "python run_paper_suite.py --experiment A2",
    },
    "A3.STRATEGIC_TIMING_ATTACK": {
        "theorems": [
            {"id": "Theorem 12", "claim": "Stage 1+2 reduces late manipulation"},
            {"id": "Proposition 9", "claim": "Strategic timing mitigation"},
        ],
        "criteria": [
            {
                "id": "regret_log_non_negative",
                "metric_path": ("aggregated_metrics", "capopm", "regret_log"),
                "threshold": 0.0,
                "direction": ">=",
                "description": "Log-score regret vs uncorrected should be non-negative.",
                "theorem": "Theorem 12",
            },
            _grid_requirement(
                "Attack-strength sensitivity requires multi-scenario grid (early vs late attacks).",
                theorem="Proposition 9",
            ),
        ],
        "reproduction_command": "python run_paper_suite.py --experiment A3",
    },
    "B1.CORRECTION_NO_REGRET": {
        "theorems": [
            {"id": "Theorem 14", "claim": "Corrections do not increase regret"},
        ],
        "criteria": [
            {
                "id": "regret_brier_non_positive",
                "metric_path": ("aggregated_metrics", "capopm", "regret_brier"),
                "threshold": 0.0,
                "direction": "<=",
                "description": "Mean Brier regret vs uncorrected should be ≤ 0.",
                "theorem": "Theorem 14",
            },
            {
                "id": "regret_log_bad_non_positive",
                "metric_path": ("aggregated_metrics", "capopm", "regret_log_bad"),
                "threshold": 0.0,
                "direction": "<=",
                "description": "Mean log-score regret (bad if positive) should be ≤ 0.",
                "theorem": "Theorem 14",
            },
        ],
        "reproduction_command": "python run_paper_suite.py --experiment B1",
    },
    "B2.ASYMPTOTIC_RATE_CHECK": {
        "theorems": [
            {"id": "Theorem 7", "claim": "Consistency and variance decay"},
        ],
        "criteria": [
            {
                "id": "variance_slope_negative",
                "metric_path": ("summary", "status", "metrics", "rate_var_vs_n_capopm"),
                "threshold": 0.0,
                "direction": "<=",
                "description": "Variance slope vs n should be negative.",
                "theorem": "Theorem 7",
            },
            {
                "id": "bias_slope_negative",
                "metric_path": ("summary", "status", "metrics", "rate_bias_vs_n_capopm"),
                "threshold": 0.0,
                "direction": "<=",
                "description": "Bias slope vs n should be non-positive.",
                "theorem": "Theorem 7",
            },
            {
                "id": "variance_improves_uncorrected",
                "metric_path": ("summary", "status", "metrics", "rate_var_slope_diff"),
                "threshold": 0.0,
                "direction": "<=",
                "description": "Variance slope difference (capopm - uncorrected) should be ≤ 0.",
                "theorem": "Theorem 7",
            },
        ],
        "reproduction_command": "python run_paper_suite.py --experiment B2",
    },
    "B3.MISSPECIFICATION_REGRET_GRID": {
        "theorems": [
            {"id": "Proposition 8", "claim": "Regret robustness under misspecification"},
        ],
        "criteria": [
            {
                "id": "regret_brier_non_positive",
                "metric_path": ("aggregated_metrics", "capopm", "regret_brier"),
                "threshold": 0.0,
                "direction": "<=",
                "description": "Mean Brier regret vs uncorrected should be ≤ 0 across grid points.",
                "theorem": "Proposition 8",
            },
            {
                "id": "regret_log_bad_non_positive",
                "metric_path": ("aggregated_metrics", "capopm", "regret_log_bad"),
                "threshold": 0.0,
                "direction": "<=",
                "description": "Mean log-score regret (bad if positive) should be ≤ 0 across grid points.",
                "theorem": "Proposition 8",
            },
            _grid_requirement(
                "Full misspecification grid is required to validate robustness surface.",
                theorem="Proposition 8",
            ),
        ],
        "reproduction_command": "python run_paper_suite.py --experiment B3",
    },
    "B4.REGIME_POSTERIOR_CONCENTRATION": {
        "theorems": [
            {"id": "Theorem 15", "claim": "Mixture posterior concentrates with evidence"},
        ],
        "criteria": [
            {
                "id": "regime_entropy_defined",
                "metric_path": ("aggregated_metrics", "capopm", "regime_entropy"),
                "description": "Regime entropy should be finite and defined.",
                "direction": "<=",
                "threshold": 0.0,
                "allow_missing": False,
                "theorem": "Theorem 15",
            },
            _grid_requirement(
                "Need multiple evidence levels to validate entropy/weight movement.",
                theorem="Theorem 15",
            ),
        ],
        "reproduction_command": "python run_paper_suite.py --experiment B4",
    },
    "B5.ARBITRAGE_PROJECTION_IMPACT": {
        "theorems": [
            {"id": "Theorem 13", "claim": "Projection minimally perturbs prices unless violation"},
        ],
        "criteria": [
            {
                "id": "projection_distance_non_negative",
                "metric_path": ("summary", "status", "metrics", "proj_l1"),
                "threshold": 0.0,
                "direction": ">=",
                "description": "Projection distance must be non-negative (coherent if zero).",
                "theorem": "Theorem 13",
            },
            {
                "id": "projection_improves_scores_when_violation",
                "metric_path": ("summary", "status", "metrics", "delta_brier"),
                "threshold": 0.0,
                "direction": "<=",
                "description": "If violation_strength>0, projection should not worsen Brier.",
                "conditional_on_violation": True,
                "theorem": "Theorem 13",
            },
            {
                "id": "projection_improves_log_when_violation",
                "metric_path": ("summary", "status", "metrics", "delta_log_score"),
                "threshold": 0.0,
                "direction": ">=",
                "description": "If violation_strength>0, projection should not reduce log-score.",
                "conditional_on_violation": True,
                "theorem": "Theorem 13",
            },
        ],
        "reproduction_command": "python run_paper_suite.py --experiment B5",
    },
}


DEFAULT_CONTRACT: AuditContract = {
    "theorems": [],
    "criteria": [],
    "reproduction_command": None,
}
