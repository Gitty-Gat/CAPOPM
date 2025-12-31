"""
Smoke test for Tier B experiment B5.ARBITRAGE_PROJECTION_IMPACT.

Validates schema compliance, coherent/violated behaviors, and determinism.
"""

from __future__ import annotations

import csv
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.b5_arbitrage_projection_impact import run_b5_arbitrage_projection_impact
from src.capopm.experiments.schema_contract import (
    validate_capopm_core_metrics,
    validate_metrics_aggregated_csv,
    validate_reliability_csv,
    validate_summary_json,
    validate_tests_csv,
)


def main() -> None:
    strengths = [0.0, 0.5, 1.0]
    results = run_b5_arbitrage_projection_impact(
        violation_strength_grid=strengths,
        projection_method="euclidean",
        base_seed=0,
        n_runs=8,
    )
    assert results, "No scenarios produced for B5 sweep"

    artifact_paths = []
    proj_by_strength = {}
    delta_brier_by_strength = {}
    delta_log_by_strength = {}

    for item in results:
        scenario = item["scenario_name"]
        strength = float(item["sweep_params"]["violation_strength"])
        results_dir = os.path.join("results", scenario)
        agg_path = os.path.join(results_dir, "metrics_aggregated.csv")
        tests_path = os.path.join(results_dir, "tests.csv")
        summary_path = os.path.join(results_dir, "summary.json")

        for path in [agg_path, tests_path, summary_path]:
            assert os.path.exists(path), f"Missing artifact: {path}"

        ok, errors = validate_metrics_aggregated_csv(agg_path)
        assert ok, "metrics_aggregated.csv schema errors:\n" + "\n".join(errors)
        ok, errors = validate_tests_csv(tests_path)
        assert ok, "tests.csv schema errors:\n" + "\n".join(errors)
        ok, errors = validate_summary_json(summary_path)
        assert ok, "summary.json schema errors:\n" + "\n".join(errors)
        ok, errors = validate_capopm_core_metrics(agg_path)
        assert ok, "CAPOPM core metrics invalid:\n" + "\n".join(errors)

        with open(agg_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        models = sorted({r.get("model") for r in rows if r.get("model")})
        for model in models:
            rel_path = os.path.join(results_dir, f"reliability_{model}.csv")
            assert os.path.exists(rel_path), f"Missing artifact: {rel_path}"
            ok, errors = validate_reliability_csv(rel_path)
            assert ok, f"{rel_path} schema errors:\n" + "\n".join(errors)

        for r in rows:
            if r.get("model") != "after_projection":
                continue
            proj_l1 = float(r.get("proj_l1", "nan"))
            delta_brier = float(r.get("delta_brier", "nan"))
            delta_log = float(r.get("delta_log_score", "nan"))
            proj_by_strength[strength] = proj_l1
            delta_brier_by_strength[strength] = delta_brier
            delta_log_by_strength[strength] = delta_log

        artifact_paths.extend(
            [agg_path, tests_path, summary_path]
            + [os.path.join(results_dir, f"reliability_{model}.csv") for model in models]
        )

    assert 0.0 in proj_by_strength, "Coherent regime missing"
    assert strengths[-1] in proj_by_strength, "Violated regime missing"

    # Coherent regime: tiny distances and deltas ~0.
    assert abs(proj_by_strength[0.0]) <= 1e-8, "Coherent regime projection not near zero"
    assert abs(delta_brier_by_strength[0.0]) <= 1e-8, "Coherent delta_brier not near zero"
    assert abs(delta_log_by_strength[0.0]) <= 1e-8, "Coherent delta_log_score not near zero"

    # Violated regime: distances increase and scores improve.
    assert proj_by_strength[strengths[-1]] > proj_by_strength[0.0] + 1e-4, "Projection distance did not increase"
    assert delta_brier_by_strength[strengths[-1]] < 0.0, "delta_brier not improved in violated regime"
    assert delta_log_by_strength[strengths[-1]] > 0.0, "delta_log_score not improved in violated regime"

    first_contents = read_artifacts(artifact_paths)
    _ = run_b5_arbitrage_projection_impact(
        violation_strength_grid=strengths,
        projection_method="euclidean",
        base_seed=0,
        n_runs=8,
    )
    second_contents = read_artifacts(artifact_paths)
    assert first_contents == second_contents, "B5 outputs are not deterministic across runs"

    print("SMOKE TEST PASSED: B5.ARBITRAGE_PROJECTION_IMPACT artifacts, trends, and determinism validated.")


def read_artifacts(paths):
    contents = {}
    for path in paths:
        with open(path, "rb") as f:
            contents[path] = f.read()
    return contents


if __name__ == "__main__":
    main()
