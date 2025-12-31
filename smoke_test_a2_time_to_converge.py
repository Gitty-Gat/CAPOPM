"""
Smoke test for Tier A experiment A2.TIME_TO_CONVERGE.

Validates artifact creation, schema compliance, time-to-eps fields, and
determinism across repeated runs.
"""

from __future__ import annotations

import csv
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.a2_time_to_converge import run_a2_time_to_converge
from src.capopm.experiments.schema_contract import (
    validate_capopm_core_metrics,
    validate_metrics_aggregated_csv,
    validate_reliability_csv,
    validate_summary_json,
    validate_tests_csv,
)


def main() -> None:
    results = run_a2_time_to_converge(
        arrivals_grid=[1, 3],
        pool_grid=[1.0],
        steps_grid=[25],
        base_seed=0,
        n_runs=5,
    )
    assert results, "No scenarios produced for A2 sweep"

    artifact_paths = []
    time_to_eps_capopm = {"time_to_eps_0.05": [], "time_to_eps_0.10": []}
    time_to_eps_raw = {"time_to_eps_0.05": [], "time_to_eps_0.10": []}

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

        cap_rows = [r for r in rows if r.get("model") == "capopm"]
        raw_rows = [r for r in rows if r.get("model") == "raw_parimutuel"]
        if cap_rows:
            time_to_eps_capopm["time_to_eps_0.05"].append(cap_rows[0].get("time_to_eps_0.05"))
            time_to_eps_capopm["time_to_eps_0.10"].append(cap_rows[0].get("time_to_eps_0.10"))
        if raw_rows:
            time_to_eps_raw["time_to_eps_0.05"].append(raw_rows[0].get("time_to_eps_0.05"))
            time_to_eps_raw["time_to_eps_0.10"].append(raw_rows[0].get("time_to_eps_0.10"))

        artifact_paths.extend(
            [
                agg_path,
                tests_path,
                summary_path,
            ]
            + [os.path.join(results_dir, f"reliability_{model}.csv") for model in models]
        )

    assert any(is_finite_number(v) for v in time_to_eps_capopm["time_to_eps_0.05"]), (
        "CAPOPM time_to_eps_0.05 never finite"
    )
    assert any(is_finite_number(v) for v in time_to_eps_raw["time_to_eps_0.05"]), (
        "raw_parimutuel time_to_eps_0.05 never finite"
    )

    # Determinism check: capture contents, rerun, and compare.
    first_contents = read_artifacts(artifact_paths)
    _ = run_a2_time_to_converge(
        arrivals_grid=[1, 3],
        pool_grid=[1.0],
        steps_grid=[25],
        base_seed=0,
        n_runs=5,
    )
    second_contents = read_artifacts(artifact_paths)
    assert first_contents == second_contents, "A2 outputs are not deterministic across runs"

    print("SMOKE TEST PASSED: A2.TIME_TO_CONVERGE artifacts and determinism validated.")


def is_finite_number(val) -> bool:
    if val is None:
        return False
    try:
        parsed = float(val)
    except ValueError:
        return False
    return math.isfinite(parsed)


def read_artifacts(paths):
    contents = {}
    for path in paths:
        with open(path, "rb") as f:
            contents[path] = f.read()
    return contents


if __name__ == "__main__":
    main()
