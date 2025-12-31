"""
Smoke test for Tier B experiment B1.CORRECTION_NO_REGRET.

Validates artifact creation, schema compliance, determinism, and regret sanity.
"""

from __future__ import annotations

import csv
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.b1_correction_no_regret import run_b1_correction_no_regret
from src.capopm.experiments.schema_contract import (
    validate_capopm_core_metrics,
    validate_metrics_aggregated_csv,
    validate_reliability_csv,
    validate_summary_json,
    validate_tests_csv,
)


def main() -> None:
    results = run_b1_correction_no_regret(
        longshot_bias_grid=[0],
        herding_grid=[0],
        timing_attack_grid=[0.0, 1.0],
        liquidity_level_grid=["low", "high"],
        base_seed=0,
        n_runs=3,
    )
    assert results, "No scenarios produced for B1 sweep"

    artifact_paths = []
    regret_ok = False

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

        cap_row = next((r for r in rows if r.get("model") == "capopm"), None)
        stg_row = next((r for r in rows if r.get("model") == "capopm_stage1_only"), None)
        assert cap_row is not None and stg_row is not None, "Missing capopm rows"

        for row in [cap_row, stg_row]:
            for key in ["regret_brier", "regret_log_bad", "regret_abs_error"]:
                val = float(row.get(key, "nan"))
                assert math.isfinite(val), f"{key} is not finite for {row.get('model')}"

        cap_regret = float(cap_row.get("regret_brier", "nan"))
        if math.isfinite(cap_regret) and cap_regret <= 0.0:
            regret_ok = True

        artifact_paths.extend(
            [
                agg_path,
                tests_path,
                summary_path,
            ]
            + [os.path.join(results_dir, f"reliability_{model}.csv") for model in models]
        )

    assert regret_ok, "No setting with capopm regret_brier <= 0"

    first_contents = read_artifacts(artifact_paths)
    _ = run_b1_correction_no_regret(
        longshot_bias_grid=[0],
        herding_grid=[0],
        timing_attack_grid=[0.0, 1.0],
        liquidity_level_grid=["low", "high"],
        base_seed=0,
        n_runs=3,
    )
    second_contents = read_artifacts(artifact_paths)
    assert first_contents == second_contents, "B1 outputs are not deterministic across runs"

    print("SMOKE TEST PASSED: B1.CORRECTION_NO_REGRET artifacts and determinism validated.")


def read_artifacts(paths):
    contents = {}
    for path in paths:
        with open(path, "rb") as f:
            contents[path] = f.read()
    return contents


if __name__ == "__main__":
    main()
