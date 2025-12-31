"""
Smoke test for the CAPOPM audit layer.

Validates that audit.json is written, hashes are present, and degenerate
calibration cases are flagged on an existing deterministic scenario.
"""

from __future__ import annotations

import json
import os
import sys

# Allow running from repo root without installation.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_audit_layer import audit_scenario  # noqa: E402


def main() -> None:
    scenario_dir = os.path.join(
        REPO_ROOT, "results", "a1_info_eff_q65_inf30_adv10_seed99101"
    )
    assert os.path.isdir(scenario_dir), "Scenario directory missing for audit smoke test."

    audit_path = audit_scenario(scenario_dir)
    assert os.path.exists(audit_path), "audit.json was not written."

    with open(audit_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert "audit_hash" in report, "Audit hash missing."
    cap_cal = report.get("calibration", {}).get("capopm", {})
    assert cap_cal, "Calibration block missing for capopm."
    assert cap_cal.get("ece_interpretable") is False, "Degenerate calibration not flagged."
    assert cap_cal.get("metric_used") == "discrete_calibration", "Discrete fallback not used."
    coverage = report.get("coverage", {}).get("capopm", {}).get("overall", {})
    assert coverage.get("warning_low_n") is True, "Low-n coverage warning missing."

    print("SMOKE TEST PASSED: Audit layer flags degenerate calibration and low-n coverage.")


if __name__ == "__main__":
    main()
