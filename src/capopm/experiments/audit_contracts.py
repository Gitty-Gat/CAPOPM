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
        "criteria": [],
        "reproduction_command": "python run_paper_suite.py --experiment A2",
    },
    "A3.STRATEGIC_TIMING_ATTACK": {
        "theorems": [
            {"id": "Theorem 12", "claim": "Stage 1+2 reduces late manipulation"},
            {"id": "Proposition 9", "claim": "Strategic timing mitigation"},
        ],
        "criteria": [
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
        "criteria": [],
        "reproduction_command": "python run_paper_suite.py --experiment B1",
    },
    "B2.ASYMPTOTIC_RATE_CHECK": {
        "theorems": [
            {"id": "Theorem 7", "claim": "Consistency and variance decay"},
        ],
        "criteria": [],
        "reproduction_command": "python run_paper_suite.py --experiment B2",
    },
    "B3.MISSPECIFICATION_REGRET_GRID": {
        "theorems": [
            {"id": "Proposition 8", "claim": "Regret robustness under misspecification"},
        ],
        "criteria": [
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
        ],
        "reproduction_command": "python run_paper_suite.py --experiment B5",
    },
}


DEFAULT_CONTRACT: AuditContract = {
    "theorems": [],
    "criteria": [],
    "reproduction_command": None,
}
