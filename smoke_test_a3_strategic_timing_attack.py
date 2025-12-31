"""
Smoke test for Tier A experiment A3.STRATEGIC_TIMING_ATTACK.

Validates artifact creation, schema compliance, determinism, and basic
robustness sanity under attack.
"""

from __future__ import annotations

import csv
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.a3_strategic_timing_attack import run_a3_strategic_timing_attack
from src.capopm.experiments.schema_contract import (
    validate_capopm_core_metrics,
    validate_metrics_aggregated_csv,
    validate_reliability_csv,
    validate_summary_json,
    validate_tests_csv,
)


def main() -> None:
    results = run_a3_strategic_timing_attack(
        attack_strength_grid=[0.0, 1.0],
        attack_window_grid=[0.2],
        adversarial_size_scale_grid=[1, 3],
        base_seed=0,
        n_runs=5,
    )
    assert results, "No scenarios produced for A3 sweep"

    artifact_paths = []
    improved_cases = 0
    total_attack_cases = 0

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
        unc_row = next((r for r in rows if r.get("model") == "uncorrected"), None)
        assert cap_row is not None and unc_row is not None, "Missing capopm or uncorrected rows"

        for key in ["regime_entropy", "regime_max_weight"]:
            val = float(cap_row.get(key, "nan"))
            assert math.isfinite(val), f"{key} is not finite for capopm"

        attack_strength = float(cap_row.get("attack_strength", "0"))
        if attack_strength >= 1.0:
            total_attack_cases += 1
            cap_mae = float(cap_row.get("mae_prob", "nan"))
            unc_mae = float(unc_row.get("mae_prob", "nan"))
            if math.isfinite(cap_mae) and math.isfinite(unc_mae) and cap_mae < unc_mae:
                improved_cases += 1

        artifact_paths.extend(
            [
                agg_path,
                tests_path,
                summary_path,
            ]
            + [os.path.join(results_dir, f"reliability_{model}.csv") for model in models]
        )

    assert total_attack_cases > 0, "No attack_strength=1.0 cases found"
    assert improved_cases >= 1, "CAPOPM did not beat uncorrected on bias in any attack case"

    first_contents = read_artifacts(artifact_paths)
    _ = run_a3_strategic_timing_attack(
        attack_strength_grid=[0.0, 1.0],
        attack_window_grid=[0.2],
        adversarial_size_scale_grid=[1, 3],
        base_seed=0,
        n_runs=5,
    )
    second_contents = read_artifacts(artifact_paths)
    assert first_contents == second_contents, "A3 outputs are not deterministic across runs"

    print("SMOKE TEST PASSED: A3.STRATEGIC_TIMING_ATTACK artifacts and determinism validated.")


def read_artifacts(paths):
    contents = {}
    for path in paths:
        with open(path, "rb") as f:
            contents[path] = f.read()
    return contents


if __name__ == "__main__":
    main()
