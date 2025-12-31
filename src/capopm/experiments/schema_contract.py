"""
Schema contract validators for empirical experiment artifacts.

Validators are pandas-free and return (ok, errors) tuples.
"""

from __future__ import annotations

import csv
import json
import math
from typing import Dict, List, Tuple


REQUIRED_METRICS_COLUMNS = [
    "scenario_name",
    "experiment_id",
    "tier",
    "seed",
    "model",
    "brier",
    "log_score",
    "mae_prob",
    "calibration_ece",
    "p_hat",
]

REQUIRED_RELIABILITY_COLUMNS = [
    "scenario_name",
    "experiment_id",
    "tier",
    "seed",
    "model",
    "bin_low",
    "bin_high",
    "count",
    "mean_pred",
    "mean_outcome",
]

REQUIRED_TESTS_COLUMNS = [
    "scenario_name",
    "experiment_id",
    "tier",
    "seed",
    "model",
    "metric",
    "test",
    "stat",
    "p_value",
    "p_value_str",
    "p_holm",
    "p_holm_str",
    "p_bonferroni",
    "p_bonferroni_str",
    "diff_mean",
    "better_if_negative",
    "ci_low",
    "ci_high",
]

REQUIRED_SUMMARY_KEYS = [
    "aggregated_metrics",
    "tests",
    "warnings",
    "note",
    "metadata",
    "reporting_version",
]

REQUIRED_SUMMARY_METADATA_KEYS = ["scenario_name", "experiment_id", "tier", "seed"]

CAPOPM_CORE_FIELDS = ["brier", "log_score", "mae_prob", "calibration_ece", "p_hat"]


def validate_metrics_aggregated_csv(path: str) -> Tuple[bool, List[str]]:
    rows, errors = _read_csv_rows(path)
    if errors:
        return False, errors
    header = rows[0].keys() if rows else []
    errors.extend(_missing_columns(header, REQUIRED_METRICS_COLUMNS))
    if errors:
        return False, errors
    if not rows:
        return False, [f"{path}: no rows found"]
    for idx, row in enumerate(rows):
        errors.extend(_require_nonempty(row, ["scenario_name", "experiment_id", "tier", "model"], path, idx))
        errors.extend(_require_int(row, ["seed"], path, idx))
        errors.extend(_require_float(row, ["brier", "log_score", "mae_prob", "calibration_ece", "p_hat"], path, idx))
    return len(errors) == 0, errors


def validate_reliability_csv(path: str) -> Tuple[bool, List[str]]:
    rows, errors = _read_csv_rows(path)
    if errors:
        return False, errors
    header = rows[0].keys() if rows else []
    errors.extend(_missing_columns(header, REQUIRED_RELIABILITY_COLUMNS))
    if errors:
        return False, errors
    if not rows:
        return False, [f"{path}: no rows found"]
    for idx, row in enumerate(rows):
        errors.extend(_require_nonempty(row, ["scenario_name", "experiment_id", "tier", "model"], path, idx))
        errors.extend(_require_int(row, ["seed", "count"], path, idx))
        errors.extend(_require_float(row, ["bin_low", "bin_high", "mean_pred", "mean_outcome"], path, idx))
    return len(errors) == 0, errors


def validate_tests_csv(path: str) -> Tuple[bool, List[str]]:
    rows, errors = _read_csv_rows(path)
    if errors:
        return False, errors
    header = rows[0].keys() if rows else []
    errors.extend(_missing_columns(header, REQUIRED_TESTS_COLUMNS))
    if errors:
        return False, errors
    if not rows:
        return False, [f"{path}: no rows found"]
    for idx, row in enumerate(rows):
        errors.extend(_require_nonempty(row, ["scenario_name", "experiment_id", "tier", "model", "metric", "test"], path, idx))
        errors.extend(_require_int(row, ["seed"], path, idx))
        errors.extend(_require_bool(row, ["better_if_negative"], path, idx))
        errors.extend(_require_float_optional(row, ["stat", "p_value", "diff_mean", "ci_low", "ci_high"], path, idx))
        errors.extend(_require_float_optional(row, ["p_holm", "p_bonferroni"], path, idx))
    return len(errors) == 0, errors


def validate_summary_json(path: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        return False, [f"{path}: failed to parse JSON: {exc}"]
    if not isinstance(payload, dict):
        return False, [f"{path}: summary.json must be a JSON object"]
    errors.extend(_missing_keys(payload, REQUIRED_SUMMARY_KEYS, path))
    if errors:
        return False, errors
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False, [f"{path}: metadata must be an object"]
    errors.extend(_missing_keys(metadata, REQUIRED_SUMMARY_METADATA_KEYS, path + " metadata"))
    if not isinstance(payload.get("reporting_version"), str):
        errors.append(f"{path}: reporting_version must be a string")
    return len(errors) == 0, errors


def validate_capopm_core_metrics(metrics_csv_path: str) -> Tuple[bool, List[str]]:
    rows, errors = _read_csv_rows(metrics_csv_path)
    if errors:
        return False, errors
    cap_rows = [r for r in rows if r.get("model") == "capopm"]
    if not cap_rows:
        return False, [f"{metrics_csv_path}: no capopm rows found"]
    for idx, row in enumerate(cap_rows):
        for field in CAPOPM_CORE_FIELDS:
            val = row.get(field)
            if val is None or str(val).strip() == "":
                errors.append(f"{metrics_csv_path}: capopm {field} missing at row {idx}")
                continue
            try:
                parsed = float(val)
            except ValueError:
                errors.append(f"{metrics_csv_path}: capopm {field} not float at row {idx}")
                continue
            if not math.isfinite(parsed):
                errors.append(f"{metrics_csv_path}: capopm {field} not finite at row {idx}")
    return len(errors) == 0, errors


def _read_csv_rows(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    errors: List[str] = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as exc:
        return [], [f"{path}: failed to read CSV: {exc}"]
    if reader.fieldnames is None:
        errors.append(f"{path}: missing header row")
    return rows, errors


def _missing_columns(header, required: List[str]) -> List[str]:
    if not header:
        return ["missing CSV header"]
    return [f"missing column: {col}" for col in required if col not in header]


def _missing_keys(obj: Dict, required: List[str], path: str) -> List[str]:
    return [f"{path}: missing key '{k}'" for k in required if k not in obj]


def _require_nonempty(row: Dict[str, str], fields: List[str], path: str, idx: int) -> List[str]:
    errors: List[str] = []
    for field in fields:
        val = row.get(field)
        if val is None or str(val).strip() == "":
            errors.append(f"{path}: row {idx} missing or empty '{field}'")
    return errors


def _require_int(row: Dict[str, str], fields: List[str], path: str, idx: int) -> List[str]:
    errors: List[str] = []
    for field in fields:
        val = row.get(field)
        if val is None or str(val).strip() == "":
            errors.append(f"{path}: row {idx} missing int '{field}'")
            continue
        try:
            int(str(val).strip())
        except ValueError:
            errors.append(f"{path}: row {idx} invalid int '{field}'")
    return errors


def _require_float(row: Dict[str, str], fields: List[str], path: str, idx: int) -> List[str]:
    errors: List[str] = []
    for field in fields:
        val = row.get(field)
        if val is None or str(val).strip() == "":
            errors.append(f"{path}: row {idx} missing float '{field}'")
            continue
        try:
            float(str(val).strip())
        except ValueError:
            errors.append(f"{path}: row {idx} invalid float '{field}'")
    return errors


def _require_float_optional(row: Dict[str, str], fields: List[str], path: str, idx: int) -> List[str]:
    errors: List[str] = []
    for field in fields:
        val = row.get(field)
        if val is None or str(val).strip() == "":
            continue
        try:
            float(str(val).strip())
        except ValueError:
            errors.append(f"{path}: row {idx} invalid float '{field}'")
    return errors


def _require_bool(row: Dict[str, str], fields: List[str], path: str, idx: int) -> List[str]:
    errors: List[str] = []
    for field in fields:
        val = row.get(field)
        if val is None or str(val).strip() == "":
            errors.append(f"{path}: row {idx} missing bool '{field}'")
            continue
        normalized = str(val).strip().lower()
        if normalized not in {"true", "false", "1", "0"}:
            errors.append(f"{path}: row {idx} invalid bool '{field}'")
    return errors
