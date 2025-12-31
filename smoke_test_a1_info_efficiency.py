"""
Smoke test for Tier A experiment A1.INFO_EFFICIENCY_CURVES.

Validates artifact creation, schema columns, absence of NaNs for CAPOPM
core metrics, and determinism across repeated runs with identical seeds.
"""

from __future__ import annotations

import csv
import os
import sys

# Allow running from repo root without installation.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.a1_info_efficiency_curves import run_a1_info_efficiency_curves
from src.capopm.experiments.schema_contract import (
    validate_capopm_core_metrics,
    validate_metrics_aggregated_csv,
    validate_reliability_csv,
    validate_summary_json,
    validate_tests_csv,
)


def main() -> None:
    grid = {
        "signal_quality_grid": [0.65],
        "informed_share_grid": [0.30],
        "adversarial_share_grid": [0.10],
    }
    base_seed = 99101
    n_runs = 4

    # First run
    results = run_a1_info_efficiency_curves(
        signal_quality_grid=grid["signal_quality_grid"],
        informed_share_grid=grid["informed_share_grid"],
        adversarial_share_grid=grid["adversarial_share_grid"],
        base_seed=base_seed,
        n_runs=n_runs,
    )
    assert results, "No scenarios produced for A1 sweep"
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
    reliability_paths = []
    for model in models:
        path = os.path.join(results_dir, f"reliability_{model}.csv")
        reliability_paths.append(path)
        assert os.path.exists(path), f"Missing artifact: {path}"
        ok, errors = validate_reliability_csv(path)
        assert ok, f"{path} schema errors:\n" + "\n".join(errors)

    # Determinism check: capture contents, rerun, and compare.
    artifact_paths = [agg_path, tests_path, summary_path] + reliability_paths
    first_contents = read_artifacts(artifact_paths)

    _ = run_a1_info_efficiency_curves(
        signal_quality_grid=grid["signal_quality_grid"],
        informed_share_grid=grid["informed_share_grid"],
        adversarial_share_grid=grid["adversarial_share_grid"],
        base_seed=base_seed,
        n_runs=n_runs,
    )
    second_contents = read_artifacts(artifact_paths)
    assert first_contents == second_contents, "A1 outputs are not deterministic across runs"

    print("SMOKE TEST PASSED: A1.INFO_EFFICIENCY_CURVES artifacts and determinism validated.")


def read_artifacts(paths):
    """Read artifacts as raw bytes for deterministic comparisons."""

    contents = {}
    for path in paths:
        with open(path, "rb") as f:
            contents[path] = f.read()
    return contents


if __name__ == "__main__":
    main()
