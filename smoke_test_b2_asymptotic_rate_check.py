"""
Smoke test for Tier B experiment B2.ASYMPTOTIC_RATE_CHECK.

Validates artifact creation, schema compliance, determinism, and slope sanity.
"""

from __future__ import annotations

import csv
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.b2_asymptotic_rate_check import run_b2_asymptotic_rate_check
from src.capopm.experiments.schema_contract import (
    validate_capopm_core_metrics,
    validate_metrics_aggregated_csv,
    validate_reliability_csv,
    validate_summary_json,
    validate_tests_csv,
)


def main() -> None:
    results = run_b2_asymptotic_rate_check(
        n_total_grid=[50, 100, 200],
        base_seed=0,
        n_runs=5,
    )
    assert results, "No scenario produced for B2"
    scenario = results[0]["scenario_name"]
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
        reader = csv.DictReader(f)
        rows = list(reader)
    models = sorted({r.get("model") for r in rows if r.get("model")})
    for model in models:
        rel_path = os.path.join(results_dir, f"reliability_{model}.csv")
        assert os.path.exists(rel_path), f"Missing artifact: {rel_path}"
        ok, errors = validate_reliability_csv(rel_path)
        assert ok, f"{rel_path} schema errors:\n" + "\n".join(errors)

    assert is_finite_metric(rows, "capopm", "posterior_variance"), (
        "CAPOPM posterior_variance not finite"
    )
    assert is_finite_metric(rows, "uncorrected", "posterior_variance"), (
        "Uncorrected posterior_variance not finite"
    )

    slope_ok = False
    for row in rows:
        if row.get("seed") == "-1" and row.get("rate_var_vs_n"):
            model = row.get("model")
            slope = float(row.get("rate_var_vs_n", "nan"))
            if model in {"capopm", "uncorrected"} and math.isfinite(slope) and slope < 0.0:
                slope_ok = True
    assert slope_ok, "No negative rate_var_vs_n slope found for capopm/uncorrected"

    artifact_paths = [agg_path, tests_path, summary_path] + [
        os.path.join(results_dir, f"reliability_{model}.csv") for model in models
    ]
    first_contents = read_artifacts(artifact_paths)
    _ = run_b2_asymptotic_rate_check(
        n_total_grid=[50, 100, 200],
        base_seed=0,
        n_runs=5,
    )
    second_contents = read_artifacts(artifact_paths)
    assert first_contents == second_contents, "B2 outputs are not deterministic across runs"

    print("SMOKE TEST PASSED: B2.ASYMPTOTIC_RATE_CHECK artifacts and determinism validated.")


def is_finite_metric(rows, model, metric) -> bool:
    for row in rows:
        if row.get("model") != model:
            continue
        if row.get("seed") == "-1":
            continue
        val = row.get(metric)
        if val is None or str(val).strip() == "":
            continue
        try:
            parsed = float(val)
        except ValueError:
            continue
        if math.isfinite(parsed):
            return True
    return False


def read_artifacts(paths):
    contents = {}
    for path in paths:
        with open(path, "rb") as f:
            contents[path] = f.read()
    return contents


if __name__ == "__main__":
    main()
