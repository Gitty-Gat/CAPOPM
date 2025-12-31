"""
Build PAPER_RUN_MANIFEST.json from existing audit.json files.

This is reporting-only and does not rerun experiments.
"""

from __future__ import annotations

import glob
import json
import os
import sys

# Allow running from repo root without installation.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.capopm.experiments.paper_config import MIN_GRID_POINTS, MIN_RUNS_PER_CELL


def main() -> None:
    entries = []
    for summary_path in sorted(glob.glob("results/*/summary.json")):
        scenario = os.path.basename(os.path.dirname(summary_path))
        audit_path = os.path.join(os.path.dirname(summary_path), "audit.json")
        if not os.path.exists(audit_path):
            continue
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
        meta = audit.get("metadata", {})
        entries.append(
            {
                "scenario_name": scenario,
                "experiment": audit.get("experiment_id"),
                "tier": audit.get("tier"),
                "seed": meta.get("seed"),
                "sweep_params": meta,
                "audit_hash": audit.get("audit_hash"),
                "artifact_hashes": audit.get("reproducibility", {}).get("artifact_hashes", {}),
                "paper_ready": audit.get("seed_grid_coverage", {}).get("paper_ready"),
                "n_runs": audit.get("seed_grid_coverage", {}).get("n_runs"),
            }
        )

    manifest = {
        "paper_runs": entries,
        "min_runs_per_cell": MIN_RUNS_PER_CELL,
        "min_grid_points": MIN_GRID_POINTS,
    }
    with open("PAPER_RUN_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote PAPER_RUN_MANIFEST.json with {len(entries)} entries.")


if __name__ == "__main__":
    main()
