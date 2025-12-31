"""
Run the CAPOPM audit layer over existing experiment outputs.

This script is reporting-only: it reads completed artifacts and writes
`audit.json` alongside each `summary.json`. No math or DGP logic is touched.
"""

from __future__ import annotations

import glob
import json
import os
import sys

# Allow running from repo root without installation.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from src.capopm.experiments.audit import run_audit_for_results  # noqa: E402


def audit_scenario(results_dir: str) -> str:
    """Run audit for a single scenario directory."""

    summary_path = os.path.join(results_dir, "summary.json")
    metrics_path = os.path.join(results_dir, "metrics_aggregated.csv")
    tests_path = os.path.join(results_dir, "tests.csv")
    reliability_paths = sorted(glob.glob(os.path.join(results_dir, "reliability_*.csv")))

    audit_report = run_audit_for_results(
        scenario_name=os.path.basename(results_dir),
        summary_path=summary_path,
        metrics_path=metrics_path if os.path.exists(metrics_path) else None,
        tests_path=tests_path if os.path.exists(tests_path) else None,
        reliability_paths=reliability_paths,
        registry_root=os.path.dirname(results_dir),
    )
    audit_path = os.path.join(results_dir, "audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    return audit_path


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PARENT, "results")
    summary_paths = sorted(glob.glob(os.path.join(root, "*/summary.json")))
    if not summary_paths:
        print(f"No summary.json files found under {root}")
        return
    for summary_path in summary_paths:
        results_dir = os.path.dirname(summary_path)
        audit_path = audit_scenario(results_dir)
        print(f"Wrote audit: {audit_path}")


if __name__ == "__main__":
    main()
