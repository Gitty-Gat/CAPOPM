"""
Smoke test for Tier B experiment B3.MISSPECIFICATION_REGRET_GRID.

Validates artifacts, schema compliance, determinism, and regret fields.
"""

from __future__ import annotations

import csv
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.b3_misspecification_regret_grid import run_b3_misspecification_regret_grid
from src.capopm.experiments.schema_contract import (
    validate_capopm_core_metrics,
    validate_metrics_aggregated_csv,
    validate_reliability_csv,
    validate_summary_json,
    validate_tests_csv,
)


def main() -> None:
    results = run_b3_misspecification_regret_grid(
        structural_mis_grid=[0.0, 0.1],
        ml_bias_grid=[0.0, 0.05],
        base_seed=0,
        n_runs=3,
    )
    assert results, "No scenarios produced for B3 sweep"

    artifact_paths = []
    regret_finite = False
    grid_cols_present = False

    for item in results:
        scenario = item["scenario_name"]
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

        for r in rows:
            if r.get("model") == "capopm":
                for key in ["regret_brier", "regret_log_bad", "regret_abs_error"]:
                    val = float(r.get(key, "nan"))
                    if math.isfinite(val):
                        regret_finite = True
                if "structural_shift" in r and "ml_bias" in r:
                    grid_cols_present = True

        artifact_paths.extend(
            [agg_path, tests_path, summary_path]
            + [os.path.join(results_dir, f"reliability_{model}.csv") for model in models]
        )

    assert regret_finite, "No finite regret metrics for capopm"
    assert grid_cols_present, "Grid columns missing in metrics_aggregated"

    first_contents = read_artifacts(artifact_paths)
    _ = run_b3_misspecification_regret_grid(
        structural_mis_grid=[0.0, 0.1],
        ml_bias_grid=[0.0, 0.05],
        base_seed=0,
        n_runs=3,
    )
    second_contents = read_artifacts(artifact_paths)
    assert first_contents == second_contents, "B3 outputs are not deterministic across runs"

    print("SMOKE TEST PASSED: B3.MISSPECIFICATION_REGRET_GRID artifacts and determinism validated.")


def read_artifacts(paths):
    contents = {}
    for path in paths:
        with open(path, "rb") as f:
            contents[path] = f.read()
    return contents


if __name__ == "__main__":
    main()
