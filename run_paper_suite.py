"""
Paper-ready harness for CAPOPM experiments (A1–A3, B1–B5) and Tier C stress runs.

Enforces run counts and grid coverage without altering CAPOPM math or DGP logic.
Results are written under results/<scenario>/ with Phase-7 artifacts plus audit.json
and config snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

from src.capopm.experiments.a1_info_efficiency_curves import run_a1_info_efficiency_curves
from src.capopm.experiments.a2_time_to_converge import run_a2_time_to_converge
from src.capopm.experiments.a3_strategic_timing_attack import run_a3_strategic_timing_attack
from src.capopm.experiments.b1_correction_no_regret import run_b1_correction_no_regret
from src.capopm.experiments.b2_asymptotic_rate_check import run_b2_asymptotic_rate_check
from src.capopm.experiments.b3_misspecification_regret_grid import run_b3_misspecification_regret_grid
from src.capopm.experiments.b4_regime_posterior_concentration import run_b4_regime_posterior_concentration
from src.capopm.experiments.b5_arbitrage_projection_impact import run_b5_arbitrage_projection_impact
from src.capopm.experiments.c1_zero_information_market import run_c1_zero_information_market
from src.capopm.experiments.c2_adversary_majoritarian import run_c2_adversary_majoritarian
from src.capopm.experiments.c3_nonstationary_ptrue_drift import run_c3_nonstationary_ptrue_drift
from src.capopm.experiments.c4_regime_switch_midwindow import run_c4_regime_switch_midwindow
from src.capopm.experiments.c5_liquidity_dropout import run_c5_liquidity_dropout
from src.capopm.experiments.c6_extreme_prior_mislead import run_c6_extreme_prior_mislead
from src.capopm.experiments.paper_config import MIN_GRID_POINTS, MIN_RUNS_PER_CELL, PAPER_GRIDS
from src.capopm.experiments.runner import sanitize_for_json


def _run_ast_gate() -> None:
    checker = [sys.executable, os.path.join("scripts", "forbidden_ast_check.py")]
    result = subprocess.run(checker, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)


def main() -> None:
    _run_ast_gate()
    parser = argparse.ArgumentParser(description="Run paper-ready CAPOPM experiment suite.")
    parser.add_argument(
        "--experiment",
        choices=["A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5", "ALL"],
        default="ALL",
        help="Limit to a single experiment or run all.",
    )
    parser.add_argument("--runs", type=int, default=MIN_RUNS_PER_CELL, help="Runs per grid cell.")
    parser.add_argument(
        "--tier",
        choices=["paper", "C", "all"],
        default="paper",
        help="Select paper experiments (A/B), Tier C stress experiments, or all.",
    )
    args = parser.parse_args()

    targets = [args.experiment] if args.experiment != "ALL" else ["A1", "A2", "A3", "B1", "B2", "B3", "B4", "B5"]
    manifest_entries: List[Dict] = []

    def _add_entry(exp: str, res: List[Dict], grid: Dict, tier: str) -> None:
        for r in res:
            manifest_entries.append(
                {
                    "experiment": exp,
                    "scenario_name": r.get("scenario_name"),
                    "seed": r.get("seed"),
                    "base_seed": r.get("base_seed", grid.get("base_seed") if grid else None),
                    "run_index": r.get("run_index"),
                    "sweep_params": _normalize_sweep_params(r.get("sweep_params", {})),
                    "n_runs": max(args.runs, MIN_RUNS_PER_CELL),
                    "tier": tier,
                }
            )

    if args.tier in {"paper", "all"}:
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
                    evidence_strength_grid=grid["evidence_grid"],
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

            _add_entry(exp, res, grid, tier="A" if exp.startswith("A") else "B")

    if args.tier in {"C", "all"}:
        c_res = run_c1_zero_information_market(base_seed=302000, n_runs=max(args.runs, MIN_RUNS_PER_CELL))
        _add_entry("C1", c_res, {"base_seed": 302000}, tier="C")

        c_res = run_c2_adversary_majoritarian(base_seed=303000, n_runs=max(args.runs, MIN_RUNS_PER_CELL))
        _add_entry("C2", c_res, {"base_seed": 303000}, tier="C")

        c_res = run_c3_nonstationary_ptrue_drift(base_seed=304000, n_runs=max(args.runs, MIN_RUNS_PER_CELL))
        _add_entry("C3", c_res, {"base_seed": 304000}, tier="C")

        c_res = run_c4_regime_switch_midwindow(base_seed=305000, n_runs=max(args.runs, MIN_RUNS_PER_CELL))
        _add_entry("C4", c_res, {"base_seed": 305000}, tier="C")

        c_res = run_c5_liquidity_dropout(base_seed=306000, n_runs=max(args.runs, MIN_RUNS_PER_CELL))
        _add_entry("C5", c_res, {"base_seed": 306000}, tier="C")

        c_res = run_c6_extreme_prior_mislead(base_seed=307000, n_runs=max(args.runs, MIN_RUNS_PER_CELL))
        _add_entry("C6", c_res, {"base_seed": 307000}, tier="C")

    write_manifest(manifest_entries)


def _normalize_sweep_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return sanitize_for_json(params)


def write_manifest(entries: List[Dict]) -> None:
    created_utc = datetime.utcnow().isoformat() + "Z"
    python_version = sys.version
    command = sys.argv
    ast_gate = {
        "checker_path": os.path.join("scripts", "forbidden_ast_check.py"),
        "status": "passed",
    }

    enriched = []
    for entry in entries:
        scenario = entry.get("scenario_name")
        results_dir = os.path.join("results", scenario) if scenario else None
        summary_path = os.path.join(results_dir, "summary.json") if results_dir else None
        audit_path = os.path.join("results", scenario, "audit.json") if scenario else None
        audit_hash = None
        artifact_hashes = {}
        if audit_path and os.path.exists(audit_path):
            with open(audit_path, "r", encoding="utf-8") as f:
                audit = json.load(f)
            audit_hash = audit.get("audit_hash")
            artifact_hashes = audit.get("reproducibility", {}).get("artifact_hashes", {})
        enriched.append(
            {
                **entry,
                "results_dir": results_dir,
                "summary_path": summary_path,
                "audit_path": audit_path if audit_hash is not None else None,
                "audit_hash": audit_hash,
                "artifact_hashes": artifact_hashes if audit_hash is not None else {},
            }
        )

    manifest = {
        "paper_runs": enriched,
        "min_runs_per_cell": MIN_RUNS_PER_CELL,
        "min_grid_points": MIN_GRID_POINTS,
        "created_utc": created_utc,
        "python_version": python_version,
        "command": command,
        "ast_gate": ast_gate,
    }
    # If an existing manifest exists, retain previous entries.
    if os.path.exists("PAPER_RUN_MANIFEST.json"):
        try:
            with open("PAPER_RUN_MANIFEST.json", "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev, dict) and "paper_runs" in prev:
                manifest["paper_runs"] = prev.get("paper_runs", []) + manifest["paper_runs"]
        except Exception:
            pass

    with open("PAPER_RUN_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote PAPER_RUN_MANIFEST.json with {len(manifest['paper_runs'])} entries.")


if __name__ == "__main__":
    main()
