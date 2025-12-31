"""
Smoke test for Tier B experiment B4.REGIME_POSTERIOR_CONCENTRATION.

Validates artifacts, schema compliance, determinism, and entropy/max-weight trends.
"""

from __future__ import annotations

import csv
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.b4_regime_posterior_concentration import run_b4_regime_posterior_concentration
from src.capopm.experiments.schema_contract import (
    validate_capopm_core_metrics,
    validate_metrics_aggregated_csv,
    validate_reliability_csv,
    validate_summary_json,
    validate_tests_csv,
)


def main() -> None:
    results = run_b4_regime_posterior_concentration(
        evidence_strength_grid=[0.5, 1.5],
        base_seed=0,
        n_runs=3,
    )
    assert results, "No scenarios produced for B4 sweep"

    artifact_paths = []
    entropies = []
    max_weights = []

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
            rows = list(csv.DictReader(f))
        models = sorted({r.get("model") for r in rows if r.get("model")})
        for model in models:
            rel_path = os.path.join(results_dir, f"reliability_{model}.csv")
            assert os.path.exists(rel_path), f"Missing artifact: {rel_path}"
            ok, errors = validate_reliability_csv(rel_path)
            assert ok, f"{rel_path} schema errors:\n" + "\n".join(errors)

        for r in rows:
            if r.get("model") == "capopm":
                ent = float(r.get("regime_entropy", "nan"))
                mw = float(r.get("regime_max_weight", "nan"))
                assert math.isfinite(ent), "Non-finite regime_entropy for capopm"
                assert math.isfinite(mw), "Non-finite regime_max_weight for capopm"
                entropies.append((float(r.get("evidence_strength", item["sweep_params"]["evidence_strength"])), ent))
                max_weights.append((float(r.get("evidence_strength", item["sweep_params"]["evidence_strength"])), mw))

        artifact_paths.extend(
            [agg_path, tests_path, summary_path]
            + [os.path.join(results_dir, f"reliability_{model}.csv") for model in models]
        )

    # Trend check: higher evidence_strength should lower entropy and raise max weight on average.
    entropies.sort(key=lambda x: x[0])
    max_weights.sort(key=lambda x: x[0])
    assert entropies[0][0] < entropies[-1][0], "Entropy sweep strengths not ordered"
    assert max_weights[0][0] < max_weights[-1][0], "Max-weight sweep strengths not ordered"
    assert entropies[-1][1] <= entropies[0][1] + 1e-9, "Entropy did not decrease with evidence_strength"
    assert max_weights[-1][1] >= max_weights[0][1] - 1e-9, "Max weight did not increase with evidence_strength"

    first_contents = read_artifacts(artifact_paths)
    _ = run_b4_regime_posterior_concentration(
        evidence_strength_grid=[0.5, 1.5],
        base_seed=0,
        n_runs=3,
    )
    second_contents = read_artifacts(artifact_paths)
    assert first_contents == second_contents, "B4 outputs are not deterministic across runs"

    print("SMOKE TEST PASSED: B4.REGIME_POSTERIOR_CONCENTRATION artifacts, determinism, and trend validated.")


def read_artifacts(paths):
    contents = {}
    for path in paths:
        with open(path, "rb") as f:
            contents[path] = f.read()
    return contents


if __name__ == "__main__":
    main()

