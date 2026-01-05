"""
Post-experiment audit layer for CAPOPM synthetic studies (Phases 1–7).

This module is **reporting-only**: it does not modify priors, likelihoods,
pricing, corrections, or data-generating processes. It evaluates completed
experiments for statistical interpretability, reproducibility, and theorem/
proposition coverage, emitting an `audit.json` artifact per scenario.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..stats.tests import bootstrap_ci
from .audit_contracts import AUDIT_CONTRACTS, DEFAULT_CONTRACT, AuditContract
from .paper_config import (
    CALIB_MIN_NONEMPTY_BINS,
    CALIB_MIN_SAMPLES,
    CALIB_MIN_UNIQUE,
    COVERAGE_TOLERANCE,
    EFFECT_ALPHA,
    EFFECT_BOOTSTRAP_SAMPLES,
    EXTREME_P_THRESHOLD,
    PAPER_GRIDS,
    PAPER_READY_MIN_GRID,
    PAPER_READY_MIN_RUNS,
    SMOKE_RUN_THRESHOLD,
)


AUDIT_VERSION = "audit_v1"


@dataclass
class AuditThresholds:
    """Audit threshold defaults tuned for small synthetic runs."""

    min_unique_predictions: int = CALIB_MIN_UNIQUE
    min_nonempty_bins: int = CALIB_MIN_NONEMPTY_BINS
    min_calibration_samples: int = CALIB_MIN_SAMPLES
    min_runs_for_coverage: int = PAPER_READY_MIN_RUNS
    extreme_p: float = EXTREME_P_THRESHOLD
    coverage_tolerance: float = COVERAGE_TOLERANCE
    paper_ready_min_runs: int = PAPER_READY_MIN_RUNS
    paper_ready_min_grid: int = PAPER_READY_MIN_GRID
    smoke_run_threshold: int = SMOKE_RUN_THRESHOLD
    bootstrap_samples: int = EFFECT_BOOTSTRAP_SAMPLES
    bootstrap_alpha: float = EFFECT_ALPHA


def run_audit_for_results(
    scenario_name: str,
    summary_path: str,
    metrics_path: Optional[str],
    tests_path: Optional[str],
    reliability_paths: Optional[List[str]],
    registry_root: str = "results",
    thresholds: Optional[AuditThresholds] = None,
) -> Dict:
    """
    Compute audit report for an already-written scenario.

    Args:
        scenario_name: scenario directory name.
        summary_path: path to summary.json (already written).
        metrics_path: optional path to metrics_aggregated.csv.
        tests_path: optional path to tests.csv.
        reliability_paths: optional list of reliability CSVs.
        registry_root: root directory for scenario registry snapshot.
        thresholds: optional override of audit thresholds.
    """

    thresholds = thresholds or AuditThresholds()
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metadata = summary.get("metadata", {}) or {}
    experiment_id = metadata.get("experiment_id")
    contract = AUDIT_CONTRACTS.get(experiment_id, DEFAULT_CONTRACT)
    sweep_params = summary.get("sweep_params", {})

    per_run_metrics = summary.get("per_run_metrics", [])
    aggregated = summary.get("aggregated_metrics", {}) or {}
    model_names = list(aggregated.keys())
    p_hat_lists, outcome_lists = _collect_preds_outcomes(per_run_metrics, model_names)

    calibration = _audit_calibration(
        p_hat_lists=p_hat_lists,
        outcome_lists=outcome_lists,
        aggregated=aggregated,
        thresholds=thresholds,
    )
    coverage = _audit_coverage(per_run_metrics, model_names, thresholds)
    grid_snapshot = _grid_snapshot(experiment_id, sweep_params, registry_root, thresholds)
    criteria_eval = _evaluate_criteria(
        contract=contract,
        summary=summary,
        aggregated=aggregated,
        sweep_params=sweep_params,
        thresholds=thresholds,
        grid_snapshot=grid_snapshot,
    )
    theorem_checklist = _build_theorem_checklist(contract, criteria_eval)
    effect_sizes = _effect_sizes(per_run_metrics, model_names, thresholds)
    seed_grid = _seed_and_grid_coverage(per_run_metrics, sweep_params, thresholds, grid_snapshot)
    reproducibility = _reproducibility_layer(
        summary_path=summary_path,
        metrics_path=metrics_path,
        tests_path=tests_path,
        reliability_paths=reliability_paths or [],
        registry_root=registry_root,
        reproduction_command=contract.get("reproduction_command"),
    )
    borderline = _borderline_flags(
        calibration=calibration,
        coverage=coverage,
        seed_grid=seed_grid,
        criteria=criteria_eval,
        thresholds=thresholds,
    )

    audit_report = {
        "audit_version": AUDIT_VERSION,
        "scenario_name": scenario_name,
        "experiment_id": experiment_id,
        "tier": metadata.get("tier"),
        "seed": metadata.get("seed"),
        "metadata": metadata,
        "calibration": calibration,
        "coverage": coverage,
        "criteria_evaluation": criteria_eval,
        "theorem_checklist": theorem_checklist,
        "seed_grid_coverage": seed_grid,
        "effect_sizes": effect_sizes,
        "reproducibility": reproducibility,
        "borderline_flags": borderline,
        "notes": [
            "Audit is reporting-only; no CAPOPM math/DGP changed.",
            "Calibration alternative uses discrete predictions when ECE is not interpretable.",
        ],
    }
    audit_report["audit_hash"] = _hash_json(audit_report)
    return audit_report


def _collect_preds_outcomes(per_run_metrics: List[Dict], model_names: List[str]) -> Tuple[Dict[str, List[float]], Dict[str, List[int]]]:
    """Rebuild p_hat and outcome lists from per-run metrics."""

    p_hat_lists = {m: [] for m in model_names}
    outcome_lists = {m: [] for m in model_names}
    for run in per_run_metrics:
        outcome = int(run.get("outcome", run.get("y", 0)))
        for model in model_names:
            metrics = run.get("metrics", {}).get(model, {})
            if not metrics:
                continue
            p_hat = metrics.get("p_hat")
            if p_hat is None or (isinstance(p_hat, float) and not math.isfinite(p_hat)):
                continue
            p_hat_lists[model].append(float(p_hat))
            outcome_lists[model].append(outcome)
    return p_hat_lists, outcome_lists


def _audit_calibration(
    p_hat_lists: Dict[str, List[float]],
    outcome_lists: Dict[str, List[int]],
    aggregated: Dict,
    thresholds: AuditThresholds,
) -> Dict[str, Dict]:
    """Assess calibration interpretability and provide discrete fallback."""

    results: Dict[str, Dict] = {}
    for model, preds in p_hat_lists.items():
        outcomes = outcome_lists.get(model, [])
        diag = aggregated.get(model, {}).get("calibration_diagnostics", {}) or {}
        n_unique = int(diag.get("n_unique_predictions") or len(set(round(float(p), 6) for p in preds)))
        n_bins = int(diag.get("n_nonempty_bins") or 0)
        n_samples = len(preds)
        ece_val = aggregated.get(model, {}).get("calibration_ece")
        degenerate = bool(diag.get("degenerate_binning")) or n_unique <= 2

        reasons: List[str] = []
        if n_samples < thresholds.min_calibration_samples:
            reasons.append(f"insufficient_samples<{thresholds.min_calibration_samples}")
        if n_unique < thresholds.min_unique_predictions:
            reasons.append(f"unique_predictions_below_{thresholds.min_unique_predictions}")
        if n_bins < thresholds.min_nonempty_bins:
            reasons.append(f"nonempty_bins_below_{thresholds.min_nonempty_bins}")
        if degenerate:
            reasons.append("degenerate_binning_detected")

        ece_interpretable = len(reasons) == 0
        discrete_cal, discrete_bins = _discrete_calibration(preds, outcomes)

        results[model] = {
            "n_samples": n_samples,
            "n_unique_predictions": n_unique,
            "n_nonempty_bins": n_bins,
            "ece_value": ece_val,
            "ece_interpretable": ece_interpretable,
            "ece_reasons": reasons,
            "ece_status": "OK" if ece_interpretable else "NOT_INTERPRETABLE",
            "discrete_calibration": {
                "value": discrete_cal,
                "n_unique_support": discrete_bins,
                "metric": "mean_abs_bin_error_by_prediction",
            },
            "metric_used": "ece" if ece_interpretable else "discrete_calibration",
        }
    return results


def _discrete_calibration(preds: Sequence[float], outcomes: Sequence[int]) -> Tuple[float, int]:
    """Calibration error grouped by unique predicted probabilities."""

    if not preds:
        return float("nan"), 0
    buckets: Dict[float, Dict[str, float]] = {}
    for p, o in zip(preds, outcomes):
        key = round(float(p), 6)
        if key not in buckets:
            buckets[key] = {"count": 0.0, "sum_pred": 0.0, "sum_out": 0.0}
        buckets[key]["count"] += 1.0
        buckets[key]["sum_pred"] += float(p)
        buckets[key]["sum_out"] += float(o)

    total = sum(b["count"] for b in buckets.values())
    if total == 0:
        return float("nan"), len(buckets)

    error = 0.0
    for bucket in buckets.values():
        if bucket["count"] <= 0:
            continue
        mean_pred = bucket["sum_pred"] / bucket["count"]
        mean_out = bucket["sum_out"] / bucket["count"]
        error += (bucket["count"] / total) * abs(mean_pred - mean_out)
    return float(error), len(buckets)


def _audit_coverage(
    per_run_metrics: List[Dict],
    model_names: List[str],
    thresholds: AuditThresholds,
) -> Dict[str, Dict]:
    """Compute coverage slices overall and for extreme probabilities."""

    coverage: Dict[str, Dict] = {}
    for model in model_names:
        cov90, cov95 = [], []
        cov90_ext, cov95_ext = [], []
        p_true_ext = []
        for run in per_run_metrics:
            metrics = run.get("metrics", {}).get(model, {})
            p_true = run.get("p_true")
            cov90_val = metrics.get("coverage_90")
            cov95_val = metrics.get("coverage_95")
            if isinstance(cov90_val, float) and math.isfinite(cov90_val):
                cov90.append(cov90_val)
            if isinstance(cov95_val, float) and math.isfinite(cov95_val):
                cov95.append(cov95_val)
            if p_true is not None and (
                float(p_true) <= thresholds.extreme_p or float(p_true) >= 1.0 - thresholds.extreme_p
            ):
                p_true_ext.append(float(p_true))
                if isinstance(cov90_val, float) and math.isfinite(cov90_val):
                    cov90_ext.append(cov90_val)
                if isinstance(cov95_val, float) and math.isfinite(cov95_val):
                    cov95_ext.append(cov95_val)

        def _mean_or_none(vals: List[float]) -> Optional[float]:
            return float(np.mean(vals)) if vals else None

        overall_90 = _mean_or_none(cov90)
        overall_95 = _mean_or_none(cov95)
        extreme_90 = _mean_or_none(cov90_ext)
        extreme_95 = _mean_or_none(cov95_ext)

        coverage[model] = {
            "method": "beta_central_interval",
            "provenance": "interval_coverage_ptrue",
            "overall": {
                "n": len(cov90),
                "coverage_90": overall_90,
                "coverage_95": overall_95,
                "warning_low_n": len(cov90) < thresholds.min_runs_for_coverage,
            },
            "extreme_p": {
                "n": len(p_true_ext),
                "coverage_90": extreme_90,
                "coverage_95": extreme_95,
                "p_true_support": p_true_ext,
                "warning_low_support": len(p_true_ext) < max(1, thresholds.smoke_run_threshold),
            },
        }
        coverage[model]["flags"] = _coverage_flags(
            overall_90,
            overall_95,
            extreme_90,
            extreme_95,
            thresholds,
        )
    return coverage


def _coverage_flags(
    overall_90: Optional[float],
    overall_95: Optional[float],
    extreme_90: Optional[float],
    extreme_95: Optional[float],
    thresholds: AuditThresholds,
) -> List[str]:
    """Identify coverage deviations."""

    flags: List[str] = []
    if overall_90 is not None and abs(overall_90 - 0.90) > thresholds.coverage_tolerance:
        flags.append("overall_90_off_nominal")
    if overall_95 is not None and abs(overall_95 - 0.95) > thresholds.coverage_tolerance:
        flags.append("overall_95_off_nominal")
    if extreme_90 is not None and abs(extreme_90 - 0.90) > thresholds.coverage_tolerance:
        flags.append("extreme_p_90_off_nominal")
    if extreme_95 is not None and abs(extreme_95 - 0.95) > thresholds.coverage_tolerance:
        flags.append("extreme_p_95_off_nominal")
    return flags


def _evaluate_criteria(
    contract: AuditContract,
    summary: Dict,
    aggregated: Dict,
    sweep_params: Dict,
    thresholds: AuditThresholds,
    grid_snapshot: Optional[Dict] = None,
) -> Dict:
    """Programmatically evaluate pass/fail criteria with full trace."""

    criteria_results: List[Dict] = []
    grid_points_observed = (grid_snapshot or {}).get("observed_count", 0)
    recorded_status = summary.get("status", {})
    recorded_pass = recorded_status.get("pass")
    for crit in contract.get("criteria", []):
        requires_grid = bool(crit.get("requires_grid", False))
        conditional_on_violation = bool(crit.get("conditional_on_violation", False))
        violation_strength = sweep_params.get("violation_strength") or summary.get("metadata", {}).get(
            "violation_strength", 0.0
        )
        evaluated = True
        reason = None
        passed = None

        if requires_grid and grid_points_observed < thresholds.paper_ready_min_grid:
            evaluated = False
            passed = None
            reason = "grid_missing_for_claim"
        elif conditional_on_violation and (not violation_strength or float(violation_strength) <= 0.0):
            evaluated = False
            passed = True
            reason = "no_violation_triggered"

        metric_val = _resolve_metric(crit.get("metric_path"), summary, aggregated)
        comparator_val = _resolve_metric(crit.get("comparator_path"), summary, aggregated)
        threshold = crit.get("threshold", comparator_val)
        direction = crit.get("direction")
        if evaluated and metric_val is None:
            evaluated = False
            reason = "metric_missing"
            passed = None

        if evaluated:
            passed = _compare(metric_val, threshold, direction)
            if passed is None:
                evaluated = False
                reason = "comparison_not_defined"

        criteria_results.append(
            {
                "id": crit.get("id"),
                "description": crit.get("description"),
                "theorem": crit.get("theorem"),
                "metric_path": crit.get("metric_path"),
                "comparator_path": crit.get("comparator_path"),
                "direction": direction,
                "threshold": threshold,
                "metric_value": metric_val,
                "evaluated": evaluated,
                "pass": passed,
                "reason": reason,
            }
        )

    evaluated_passes = [c for c in criteria_results if c["evaluated"] and c["pass"] is True]
    evaluated_fails = [c for c in criteria_results if c["evaluated"] and c["pass"] is False]
    if evaluated_fails:
        overall = False
    elif evaluated_passes:
        overall = True
    else:
        overall = None
    semantics_mismatch = False
    if recorded_pass is True and overall is not True:
        semantics_mismatch = True
    # Additional guard: if recorded status claims pass but any evaluated criterion failed.
    if recorded_pass is True and evaluated_fails:
        semantics_mismatch = True
    if semantics_mismatch and overall is True:
        overall = False
    return {
        "overall_pass": overall,
        "criteria": criteria_results,
        "recorded_status": recorded_status,
        "status_mismatch": (recorded_pass is not None and overall is not None and bool(recorded_pass) != bool(overall)),
        "criteria_semantics_mismatch": semantics_mismatch,
        "source_of_truth": "criteria_evaluation",
        "grid_snapshot": grid_snapshot,
    }


def _resolve_metric(path: Optional[Tuple[str, ...]], summary: Dict, aggregated: Dict):
    """Resolve a nested metric path from summary or aggregated data."""

    if not path:
        return None
    segments = list(path)
    if segments and segments[0] == "summary":
        node: object = summary
        segments = segments[1:]
    elif segments and segments[0] == "aggregated_metrics":
        node = aggregated
        segments = segments[1:]
    else:
        node = aggregated
    for key in segments:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None
    if isinstance(node, (int, float)):
        return float(node)
    return node


def _compare(value, threshold, direction: Optional[str]):
    """Compare a value to a threshold with direction; returns bool or None."""

    if value is None or threshold is None or direction is None:
        return None
    try:
        v = float(value)
        t = float(threshold)
    except Exception:
        return None
    if direction == "<=":
        return v <= t
    if direction == ">=":
        return v >= t
    return None


def _build_theorem_checklist(contract: AuditContract, criteria_eval: Dict) -> List[Dict]:
    """Summarize theorem/proposition coverage based on evaluated criteria."""

    theorems = contract.get("theorems", []) or []
    checklist: List[Dict] = []
    criteria = criteria_eval.get("criteria", [])
    for theorem in theorems:
        related = [c for c in criteria if c.get("theorem") == theorem.get("id")]
        evaluated = [c for c in related if c.get("evaluated")]
        failed = [c for c in evaluated if c.get("pass") is False]
        passed = [c for c in evaluated if c.get("pass") is True]
        if failed:
            status = "failed"
        elif passed:
            status = "validated"
        elif related:
            status = "indeterminate"
        else:
            status = "not_covered"
        checklist.append(
            {
                "theorem": theorem.get("id"),
                "claim": theorem.get("claim"),
                "status": status,
                "criteria_ids": [c.get("id") for c in related],
            }
        )
    return checklist


def _effect_sizes(
    per_run_metrics: List[Dict],
    model_names: List[str],
    thresholds: AuditThresholds,
) -> Dict[str, Dict]:
    """Compute effect sizes (capopm vs baselines) with bootstrap CIs and Holm correction."""

    baselines = ["uncorrected", "raw_parimutuel"]
    metrics = ["brier", "log_score"]
    diffs = []
    effect_rows: List[Dict] = []
    for baseline in baselines:
        if baseline not in model_names or "capopm" not in model_names:
            continue
        for metric in metrics:
            cap_vals, base_vals = [], []
            for run in per_run_metrics:
                cap = run.get("metrics", {}).get("capopm", {}).get(metric)
                base = run.get("metrics", {}).get(baseline, {}).get(metric)
                if cap is None or base is None:
                    continue
                if isinstance(cap, float) and isinstance(base, float) and math.isfinite(cap) and math.isfinite(base):
                    cap_vals.append(float(cap))
                    base_vals.append(float(base))
            if not cap_vals or not base_vals or len(cap_vals) != len(base_vals):
                continue
            diff = np.asarray(cap_vals) - np.asarray(base_vals)
            ci_lo, ci_hi = bootstrap_ci(diff, n_boot=thresholds.bootstrap_samples, alpha=thresholds.bootstrap_alpha)
            diff_mean = float(np.mean(diff))
            effect_rows.append(
                {
                    "key": f"{metric}_vs_{baseline}",
                    "metric": metric,
                    "baseline": baseline,
                    "diff_mean": diff_mean,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "n": int(diff.size),
                    "better_if_negative": metric == "brier",
                    "raw_diff_series": diff,
                }
            )
            diffs.append(diff)

    # Holm correction across all comparisons using paired t p-values.
    p_values = []
    for row in effect_rows:
        diff = row["raw_diff_series"]
        if diff.size < 2:
            p_values.append(1.0)
        else:
            mean = diff.mean()
            std = diff.std(ddof=1)
            if std == 0:
                p_values.append(1.0)
            else:
                t_stat = mean / (std / math.sqrt(diff.size))
                # two-sided normal approx consistent with stats.tests.
                p = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2.0))))
                p_values.append(p)
    holm = _holm_correction(p_values)

    results: Dict[str, Dict] = {}
    for row, p_corr in zip(effect_rows, holm):
        row_out = dict(row)
        row_out.pop("raw_diff_series", None)
        row_out["paired_t_p_holm"] = p_corr
        results[row["key"]] = row_out
    return results


def _holm_correction(p_values: List[float]) -> List[float]:
    """Holm correction (duplicated to keep audit self-contained)."""

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    corrected = [0.0] * m
    for i, (idx, p) in enumerate(indexed):
        corrected[idx] = min((m - i) * p, 1.0)
    return corrected


def _grid_snapshot(experiment_id: Optional[str], sweep_params: Dict, registry_root: str, thresholds: AuditThresholds) -> Dict:
    """Aggregate grid coverage across scenarios for the same experiment."""
    # B1-CHG-01: cross-scenario grid aggregation semantics.

    exp_prefix = experiment_id.split(".")[0] if experiment_id else None
    expected_cfg = PAPER_GRIDS.get(exp_prefix, {})
    expected_axes = [k for k, v in expected_cfg.items() if isinstance(v, list)]
    if (not expected_axes) or (expected_axes and not any(k in sweep_params for k in expected_axes)):
        expected_axes = sorted(list(sweep_params.keys()))
    observed_points = set()
    if os.path.exists(registry_root):
        for name in sorted(os.listdir(registry_root)):
            summary_path = os.path.join(registry_root, name, "summary.json")
            if not os.path.exists(summary_path):
                continue
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summ = json.load(f)
            except Exception:
                continue
            meta = summ.get("metadata", {}) or {}
            if meta.get("experiment_id") != experiment_id:
                continue
            sp = summ.get("sweep_params", {})
            if expected_axes:
                raw_vals = [sp.get(ax) for ax in expected_axes]
                if any(isinstance(v, list) for v in raw_vals):
                    grids_local = [v if isinstance(v, list) else [v] for v in raw_vals]
                    for combo in product(*grids_local):
                        if all(p is not None for p in combo):
                            observed_points.add(tuple(combo))
                else:
                    point = tuple(raw_vals)
                    if all(p is not None for p in point):
                        observed_points.add(point)
    expected_points: List[tuple] = []
    if expected_axes:
        grids = [expected_cfg.get(ax, []) for ax in expected_axes]
        if grids and all(isinstance(g, list) for g in grids):
            for combo in product(*grids):
                expected_points.append(tuple(combo))
    missing_points = [pt for pt in expected_points if pt not in observed_points]
    observed_count = len(observed_points) if observed_points else 0
    if observed_count == 0:
        grid_status = "empty_registry"
    elif missing_points:
        grid_status = "incomplete_grid"
    else:
        grid_status = "observed_grid"
    return {
        "expected_axes": expected_axes,
        "expected_points": expected_points,
        "observed_points": list(observed_points),
        "observed_count": observed_count,
        "missing_points": missing_points,
        "registry_root": registry_root,
        "threshold_min_grid": thresholds.paper_ready_min_grid,
        "grid_status": grid_status,
    }


def _seed_and_grid_coverage(
    per_run_metrics: List[Dict],
    sweep_params: Dict,
    thresholds: AuditThresholds,
    grid_snapshot: Optional[Dict],
) -> Dict:
    """Classify seed/grid coverage for reproducibility audits."""

    n_runs = len(per_run_metrics)
    grid_axes = sorted(list(sweep_params.keys()))
    observed_count = (grid_snapshot or {}).get("observed_count", 0)
    flags: List[str] = []
    if n_runs < thresholds.paper_ready_min_runs:
        flags.append("seed_count_below_paper_ready")
    if observed_count < thresholds.paper_ready_min_grid and grid_axes:
        flags.append("grid_points_below_paper_ready")
    if n_runs <= thresholds.smoke_run_threshold:
        coverage_class = "smoke_only"
    elif observed_count >= thresholds.paper_ready_min_grid:
        coverage_class = "multi_grid"
    else:
        coverage_class = "multi_seed"

    paper_ready = (n_runs >= thresholds.paper_ready_min_runs) and (
        observed_count >= thresholds.paper_ready_min_grid or not grid_axes
    )

    return {
        "n_runs": n_runs,
        "grid_axes": grid_axes,
        "grid_points_observed": observed_count,
        "coverage_class": coverage_class,
        "paper_ready": paper_ready,
        "flags": flags,
        "grid_snapshot": grid_snapshot,
    }


def _reproducibility_layer(
    summary_path: str,
    metrics_path: Optional[str],
    tests_path: Optional[str],
    reliability_paths: List[str],
    registry_root: str,
    reproduction_command: Optional[str],
) -> Dict:
    """Compute hashes and registry snapshot for reproducibility."""

    scenario_dir = os.path.dirname(summary_path)
    artifact_hashes = {}
    for path in [summary_path, metrics_path, tests_path] + list(reliability_paths):
        if path and os.path.exists(path):
            artifact_hashes[os.path.basename(path)] = _hash_file(path)

    registry_snapshot = _registry_snapshot(registry_root)
    registry_hash = _hash_json(registry_snapshot)
    registry_file = os.path.join("src", "capopm", "experiments", "registry.py")
    registry_code_hash = _hash_file(registry_file) if os.path.exists(registry_file) else None
    config_snapshot_path = os.path.join(scenario_dir, "config_snapshot.json")
    config_hash = _hash_file(config_snapshot_path) if os.path.exists(config_snapshot_path) else None

    return {
        "artifact_hashes": artifact_hashes,
        "scenario_registry_snapshot": registry_snapshot,
        "registry_hash": registry_hash,
        "registry_code_hash": registry_code_hash,
        "config_hash": config_hash,
        "config_snapshot_path": config_snapshot_path if os.path.exists(config_snapshot_path) else None,
        "reproduction_command": reproduction_command,
        "reproduce_line": f"{reproduction_command} --scenario {os.path.basename(scenario_dir)}"
        if reproduction_command
        else None,
    }


def _registry_snapshot(root: str) -> Dict:
    """Snapshot available scenarios for reproducibility."""

    entries = []
    if os.path.exists(root):
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if os.path.isdir(full):
                entries.append(name)
    return {"root": root, "scenarios": entries}


def _hash_file(path: str) -> str:
    """SHA256 hash of a file."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_json(obj: Dict) -> str:
    """Stable SHA256 of JSON-serializable object."""

    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _borderline_flags(
    calibration: Dict,
    coverage: Dict,
    seed_grid: Dict,
    criteria: Dict,
    thresholds: AuditThresholds,
) -> Dict:
    """Aggregate borderline/edge-case warnings for audit summary."""

    deg_cal = [m for m, v in calibration.items() if not v.get("ece_interpretable", False)]
    low_n = seed_grid.get("n_runs", 0) < thresholds.paper_ready_min_runs
    coverage_flags = []
    marginal_failures = []
    for model, cov in coverage.items():
        coverage_flags.extend(cov.get("flags", []))
        overall = cov.get("overall", {})
        cov90 = overall.get("coverage_90")
        cov95 = overall.get("coverage_95")
        if cov90 is not None and abs(cov90 - 0.90) > thresholds.coverage_tolerance:
            marginal_failures.append(f"{model}_coverage90")
        if cov95 is not None and abs(cov95 - 0.95) > thresholds.coverage_tolerance:
            marginal_failures.append(f"{model}_coverage95")
    mismatched = criteria.get("status_mismatch", False)
    semantics_mismatch = criteria.get("criteria_semantics_mismatch", False)

    return {
        "degenerate_calibration_models": deg_cal,
        "low_n_runs": low_n,
        "coverage_flags": coverage_flags,
        "marginal_coverage_failures": marginal_failures,
        "status_mismatch_detected": mismatched,
        "criteria_semantics_mismatch": semantics_mismatch,
    }
