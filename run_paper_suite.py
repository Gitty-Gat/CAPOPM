"""
Paper-ready harness for CAPOPM experiments (A1–A3, B1–B5).

Enforces paper-level run counts and grid coverage without altering any
core CAPOPM math or DGP logic. Results are written under results/<scenario>/
with Phase-7 artifacts plus audit.json and config snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from src.capopm.experiments.a1_info_efficiency_curves import run_a1_info_efficiency_curves
from src.capopm.experiments.a2_time_to_converge import run_a2_time_to_converge
from src.capopm.experiments.a3_strategic_timing_attack import run_a3_strategic_timing_attack
from src.capopm.experiments.b1_correction_no_regret import run_b1_correction_no_regret
from src.capopm.experiments.b2_asymptotic_rate_check import run_b2_asymptotic_rate_check
from src.capopm.experiments.b3_misspecification_regret_grid import run_b3_misspecification_regret_grid
from src.capopm.experiments.b4_regime_posterior_concentration import run_b4_regime_posterior_concentration
from src.capopm.experiments.b5_arbitrage_projection_impact import run_b5_arbitrage_projection_impact
from src.capopm.experiments.paper_config import MIN_GRID_POINTS, MIN_RUNS_PER_CELL, PAPER_GRIDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready CAPOPM experiment suite.")
    parser.add_argument(
        "--experiment",
        choices=["A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5", "ALL"],
        default="ALL",
        help="Limit to a single experiment or run all.",
    )
    parser.add_argument("--runs", type=int, default=MIN_RUNS_PER_CELL, help="Runs per grid cell.")
    args = parser.parse_args()

    targets = [args.experiment] if args.experiment != "ALL" else ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5"]
    manifest_entries: List[Dict] = []

    for exp in targets:
        grid = PAPER_GRIDS.get(exp, {})
        if exp == "A1":
            assert len(grid["signal_quality_grid"]) >= MIN_GRID_POINTS
            res = run_a1_info_efficiency_curves(
                signal_quality_grid=grid["signal_quality_grid"],
                informed_share_grid=grid["informed_share_grid"],
                adversarial_share_grid=grid["adversarial_share_grid"],
                base_seed=grid["base_seed"],
                n_runs=max(args.runs, MIN_RUNS_PER_CELL),
            )
        elif exp == "A2":
            assert len(grid["arrival_grid"]) >= MIN_GRID_POINTS
            res = run_a2_time_to_converge(
                arrivals_grid=grid["arrival_grid"],
                pool_grid=grid["pool_grid"],
                steps_grid=grid["steps_grid"],
                base_seed=grid["base_seed"],
                n_runs=max(args.runs, MIN_RUNS_PER_CELL),
            )
        elif exp == "A3":
            assert len(grid["attack_strength_grid"]) >= MIN_GRID_POINTS
            res = run_a3_strategic_timing_attack(
                attack_strength_grid=grid["attack_strength_grid"],
                attack_window_grid=grid["window_grid"],
                adversarial_size_scale_grid=grid["scale_grid"],
                base_seed=grid["base_seed"],
                n_runs=max(args.runs, MIN_RUNS_PER_CELL),
            )
        elif exp == "B1":
            assert len(grid["longshot_bias_grid"]) >= MIN_GRID_POINTS
            res = run_b1_correction_no_regret(
                longshot_bias_grid=grid["longshot_bias_grid"],
                herding_grid=grid["herding_grid"],
                timing_attack_grid=grid["timing_attack_grid"],
                liquidity_level_grid=grid["liquidity_level_grid"],
                base_seed=grid["base_seed"],
                n_runs=max(args.runs, MIN_RUNS_PER_CELL),
            )
        elif exp == "B2":
            assert len(grid["n_total_grid"]) >= MIN_GRID_POINTS
            res = run_b2_asymptotic_rate_check(
                n_total_grid=grid["n_total_grid"],
                base_seed=grid["base_seed"],
                n_runs=max(args.runs, MIN_RUNS_PER_CELL),
            )
        elif exp == "B3":
            assert len(grid["structural_mis_grid"]) >= MIN_GRID_POINTS
            res = run_b3_misspecification_regret_grid(
                structural_mis_grid=grid["structural_mis_grid"],
                ml_bias_grid=grid["ml_mis_grid"],
                base_seed=grid["base_seed"],
                n_runs=max(args.runs, MIN_RUNS_PER_CELL),
            )
        elif exp == "B4":
            assert len(grid["evidence_grid"]) >= MIN_GRID_POINTS
            res = run_b4_regime_posterior_concentration(
                evidence_grid=grid["evidence_grid"],
                base_seed=grid["base_seed"],
                n_runs=max(args.runs, MIN_RUNS_PER_CELL),
            )
        elif exp == "B5":
            assert len(grid["violation_strength_grid"]) >= MIN_GRID_POINTS
            res = run_b5_arbitrage_projection_impact(
                violation_strength_grid=grid["violation_strength_grid"],
                projection_method=grid["projection_method"],
                base_seed=grid["base_seed"],
                n_runs=max(args.runs, MIN_RUNS_PER_CELL),
            )
        else:
            continue

        manifest_entries.extend(
            {
                "experiment": exp,
                "scenario_name": r.get("scenario_name"),
                "seed": r.get("seed"),
                "sweep_params": r.get("sweep_params", {}),
                "n_runs": max(args.runs, MIN_RUNS_PER_CELL),
            }
            for r in res
        )

    write_manifest(manifest_entries)


def write_manifest(entries: List[Dict]) -> None:
    """Write PAPER_RUN_MANIFEST.json at repo root."""

    enriched = []
    for entry in entries:
        scenario = entry.get("scenario_name")
        audit_path = os.path.join("results", scenario, "audit.json") if scenario else None
        audit_hash = None
        artifact_hashes = {}
        if audit_path and os.path.exists(audit_path):
            with open(audit_path, "r", encoding="utf-8") as f:
                audit = json.load(f)
            audit_hash = audit.get("audit_hash")
            artifact_hashes = audit.get("reproducibility", {}).get("artifact_hashes", {})
        enriched.append({**entry, "audit_hash": audit_hash, "artifact_hashes": artifact_hashes})

    manifest = {
        "paper_runs": enriched,
        "min_runs_per_cell": MIN_RUNS_PER_CELL,
        "min_grid_points": MIN_GRID_POINTS,
    }
    with open("PAPER_RUN_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote PAPER_RUN_MANIFEST.json with {len(entries)} entries.")


if __name__ == "__main__":
    main()
