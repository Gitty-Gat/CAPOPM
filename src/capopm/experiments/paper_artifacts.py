"""
Paper artifact generators (tables/figures) built from Phase 7 + audit outputs.

Reporting-only; does not alter any CAPOPM math or DGP logic.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List


def build_all(output_dir: str = "results/paper_artifacts") -> None:
    os.makedirs(output_dir, exist_ok=True)
    audits = _load_audits()
    _write_borderline_atlas(audits, output_dir)
    _write_claim_table(audits, output_dir)


def _load_audits() -> List[Dict]:
    audits = []
    for path in sorted(glob.glob("results/*/audit.json")):
        with open(path, "r", encoding="utf-8") as f:
            audits.append(json.load(f))
    return audits


def _write_borderline_atlas(audits: List[Dict], output_dir: str) -> None:
    rows = ["| Scenario | Experiment | Low-n | Deg Cal Models | Coverage Flags | Criteria Pass |",
            "|---|---|---|---|---|---|"]
    for audit in audits:
        flags = audit.get("borderline_flags", {})
        rows.append(
            "| {scenario} | {exp} | {low_n} | {deg} | {cov} | {crit} |".format(
                scenario=audit.get("metadata", {}).get("scenario_name"),
                exp=audit.get("experiment_id"),
                low_n=flags.get("low_n_runs"),
                deg=", ".join(flags.get("degenerate_calibration_models", [])),
                cov=", ".join(flags.get("coverage_flags", [])),
                crit=audit.get("criteria_evaluation", {}).get("overall_pass"),
            )
        )
    path = os.path.join(output_dir, "borderline_atlas.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


def _write_claim_table(audits: List[Dict], output_dir: str) -> None:
    rows = ["| Experiment | Scenario | Overall Pass | Semantics Mismatch | Theorem Status |",
            "|---|---|---|---|---|"]
    for audit in audits:
        checklist = audit.get("theorem_checklist", [])
        theorem_status = "; ".join(f"{c.get('theorem')}:{c.get('status')}" for c in checklist)
        rows.append(
            "| {exp} | {scenario} | {overall} | {mismatch} | {thm} |".format(
                exp=audit.get("experiment_id"),
                scenario=audit.get("metadata", {}).get("scenario_name"),
                overall=audit.get("criteria_evaluation", {}).get("overall_pass"),
                mismatch=audit.get("criteria_evaluation", {}).get("criteria_semantics_mismatch"),
                thm=theorem_status,
            )
        )
    path = os.path.join(output_dir, "claim_table.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))


if __name__ == "__main__":
    build_all()
