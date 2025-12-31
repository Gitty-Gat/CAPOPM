"""
Phase 7 master experiment runner (synthetic, deterministic by seed).

Produces per-run and aggregated metrics plus paired statistical tests.
Results are descriptive only and represent partial validation under
controlled synthetic assumptions.
"""

from __future__ import annotations

import csv
import json
import os
import hashlib
from typing import Dict, List, Optional

import numpy as np

from ..market_simulator import MarketConfig, simulate_market
from ..posterior import capopm_pipeline
from ..trader_model import TraderParams, build_traders
from ..likelihood import beta_binomial_update, counts_from_trade_tape, posterior_moments
from ..pricing import posterior_prices
from ..metrics.scoring import brier, log_score, mae_prob
from ..metrics.calibration import (
    calibration_ece,
    interval_coverage_outcome,
    interval_coverage_ptrue,
    reliability_table,
)
from ..metrics.distributional import (
    posterior_mean_bias,
    posterior_variance_ratio,
    mad_posterior_median,
    wasserstein_distance_beta,
)
from ..stats.tests import (
    bootstrap_ci,
    bonferroni_correction,
    holm_correction,
    paired_t_test,
    wilcoxon_signed_rank,
)
from .audit import AUDIT_VERSION, run_audit_for_results

REPORTING_VERSION = "phase7_v1"


def run_experiment(config: Dict) -> Dict:
    """Run a synthetic experiment end-to-end with Phase 7 metrics and tests.

    Notes:
    - ECE is computed on the full set of predictions/outcomes after all runs.
    - coverage_90/coverage_95 refer to p_true coverage by default; outcome
      coverage is optional and separately named.
    """

    seed = int(config.get("seed", 123))
    experiment_id = config.get("experiment_id")
    tier = config.get("tier")
    reporting_version = str(config.get("reporting_version", REPORTING_VERSION))
    extra_metadata = config.get("sweep_params", config.get("metadata", {})) or {}
    n_runs = int(config.get("n_runs", 5))
    calibration_binning = config.get("calibration_binning", config.get("ece_binning", "equal_width"))
    min_nonempty_bins = int(config.get("ece_min_nonempty_bins", 5))
    n_bins = int(config.get("ece_bins", 10))
    rng_master = np.random.default_rng(seed)

    per_run_metrics = []
    model_names = config.get(
        "models",
        [
            "capopm",
            "raw_parimutuel",
            "structural_only",
            "ml_only",
            "uncorrected",
            "beta_1_1",
            "beta_0_5_0_5",
        ],
    )

    # Collect per-model p_hat/outcome for calibration
    p_hat_lists = {m: [] for m in model_names}
    outcome_lists = {m: [] for m in model_names}

    for run_idx in range(n_runs):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

        p_true = draw_p_true(rng, config.get("p_true_dist", {"type": "fixed", "value": 0.5}))
        outcome = 1 if float(rng.random()) < p_true else 0

        traders = build_traders(
            n_traders=int(config["traders"]["n_traders"]),
            proportions=config["traders"]["proportions"],
            params_by_type=build_trader_params(config["traders"].get("params", {})),
        )

        market_cfg = MarketConfig(**config["market"])
        trade_tape, pool_path = simulate_market(rng, market_cfg, traders, p_true)
        y_raw, n_raw = counts_from_trade_tape(trade_tape)

        # Uncorrected posterior (hybrid prior, no Stage 1/2).
        base_out = capopm_pipeline(
            rng=rng,
            trade_tape=trade_tape,
            structural_cfg=config["structural_cfg"],
            ml_cfg=config["ml_cfg"],
            prior_cfg=config["prior_cfg"],
            stage1_cfg={"enabled": False},
            stage2_cfg={"enabled": False},
        )
        var_independent = posterior_moments(base_out["alpha_post"], base_out["beta_post"])[1]

        model_outputs = {}
        for model in model_names:
            if model == "capopm":
                out = capopm_pipeline(
                    rng=rng,
                    trade_tape=trade_tape,
                    structural_cfg=config["structural_cfg"],
                    ml_cfg=config["ml_cfg"],
                    prior_cfg=config["prior_cfg"],
                    stage1_cfg=config.get("stage1_cfg", {"enabled": False}),
                    stage2_cfg=config.get("stage2_cfg", {"enabled": False}),
                )
                p_hat = out["mixture_mean"] if out.get("mixture_enabled") else out["pi_yes"]
                alpha = out["alpha_post"]
                beta = out["beta_post"]
            elif model == "raw_parimutuel":
                p_hat = raw_parimutuel_prob(trade_tape, pool_path)
                alpha = None
                beta = None
            elif model == "structural_only":
                out = capopm_pipeline(
                    rng=rng,
                    trade_tape=trade_tape,
                    structural_cfg=config["structural_cfg"],
                    ml_cfg={**config["ml_cfg"], "r_ml": 0.0, "noise_std": 0.0},
                    prior_cfg={**config["prior_cfg"], "n_ml_eff": 0.0},
                    stage1_cfg={"enabled": False},
                    stage2_cfg={"enabled": False},
                )
                p_hat = out["pi_yes"]
                alpha = out["alpha_post"]
                beta = out["beta_post"]
            elif model == "ml_only":
                out = capopm_pipeline(
                    rng=rng,
                    trade_tape=trade_tape,
                    structural_cfg=config["structural_cfg"],
                    ml_cfg=config["ml_cfg"],
                    prior_cfg={**config["prior_cfg"], "n_str": 0.0},
                    stage1_cfg={"enabled": False},
                    stage2_cfg={"enabled": False},
                )
                p_hat = out["pi_yes"]
                alpha = out["alpha_post"]
                beta = out["beta_post"]
            elif model == "uncorrected":
                p_hat = base_out["pi_yes"]
                alpha = base_out["alpha_post"]
                beta = base_out["beta_post"]
            elif model == "beta_1_1":
                alpha0, beta0 = 1.0, 1.0
                alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
                p_hat, _ = posterior_prices(alpha, beta)
            elif model == "beta_0_5_0_5":
                alpha0, beta0 = 0.5, 0.5
                alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
                p_hat, _ = posterior_prices(alpha, beta)
            else:
                raise ValueError(f"Unknown model: {model}")

            model_outputs[model] = {
                "p_hat": p_hat,
                "alpha": alpha,
                "beta": beta,
            }
            p_hat_lists[model].append(p_hat)
            outcome_lists[model].append(int(outcome))

        metrics = {}
        coverage_include_outcome = bool(
            config.get("coverage_include_outcome", config.get("include_outcome_coverage", False))
        )
        for model, out in model_outputs.items():
            p_hat = out["p_hat"]
            alpha = out["alpha"]
            beta = out["beta"]
            metrics[model] = {
                "brier": brier(p_true, p_hat),
                "log_score": log_score(p_hat, outcome),
                "mae_prob": mae_prob(p_true, p_hat),
                "abs_error_outcome": abs(p_hat - outcome),
                "calibration_ece": None,
                "calibration_diagnostics": {},
                "posterior_mean_bias": posterior_mean_bias(p_hat, p_true),
                "p_hat": p_hat,
            }
            if alpha is not None and beta is not None:
                cov90 = interval_coverage_ptrue(alpha, beta, p_true, 0.90)
                cov95 = interval_coverage_ptrue(alpha, beta, p_true, 0.95)
                _, var_adj = posterior_moments(alpha, beta)
                metrics[model].update(
                    {
                        "posterior_variance_ratio": posterior_variance_ratio(var_adj, var_independent),
                        "mad_posterior_median": mad_posterior_median(alpha, beta, p_true),
                        "wasserstein_distance_beta": wasserstein_distance_beta(alpha, beta, p_true),
                        "coverage_90": cov90,
                        "coverage_95": cov95,
                        "coverage_90_outcome": np.nan,
                        "coverage_95_outcome": np.nan,
                        "coverage_outcome_90": np.nan,
                        "coverage_outcome_95": np.nan,
                        # Backward-compatible aliases
                        "coverage_90_ptrue": cov90,
                        "coverage_95_ptrue": cov95,
                    }
                )
                if coverage_include_outcome:
                    cov_out_90 = interval_coverage_outcome(alpha, beta, outcome, 0.90)
                    cov_out_95 = interval_coverage_outcome(alpha, beta, outcome, 0.95)
                    metrics[model].update(
                        {
                            "coverage_90_outcome": cov_out_90,
                            "coverage_95_outcome": cov_out_95,
                            # Explicit outcome aliases for clarity
                            "coverage_outcome_90": cov_out_90,
                            "coverage_outcome_95": cov_out_95,
                        }
                    )
            else:
                metrics[model].update(
                    {
                        "posterior_variance_ratio": None,
                        "mad_posterior_median": None,
                        "wasserstein_distance_beta": None,
                        "coverage_90": np.nan,
                        "coverage_95": np.nan,
                        "coverage_90_ptrue": np.nan,
                        "coverage_95_ptrue": np.nan,
                        "coverage_90_outcome": np.nan if coverage_include_outcome else np.nan,
                        "coverage_95_outcome": np.nan if coverage_include_outcome else np.nan,
                        "coverage_outcome_90": np.nan if coverage_include_outcome else np.nan,
                        "coverage_outcome_95": np.nan if coverage_include_outcome else np.nan,
                    }
                )
                if coverage_include_outcome:
                    metrics[model].update(
                        {
                            "coverage_90_outcome": None,
                            "coverage_95_outcome": None,
                            "coverage_outcome_90": None,
                            "coverage_outcome_95": None,
                        }
                    )

        per_run_metrics.append(
            {"run": run_idx, "p_true": p_true, "outcome": outcome, "metrics": metrics}
        )

    aggregated = aggregate_metrics(per_run_metrics, model_names)
    # Full-sample calibration ECE computed across all runs per model.
    for m in model_names:
        if len(p_hat_lists[m]) != len(outcome_lists[m]) or len(p_hat_lists[m]) < 2:
            raise ValueError("ECE requires at least 2 matched predictions/outcomes per model")
        unique_outcomes = set(outcome_lists[m])
        if not unique_outcomes.issubset({0, 1}):
            raise ValueError(f"Outcome list contains invalid values: {sorted(unique_outcomes)}")
        ece, diag = calibration_ece(
            p_hat_lists[m],
            outcome_lists[m],
            n_bins=n_bins,
            binning=calibration_binning,
            min_nonempty_bins=min_nonempty_bins,
            allow_fallback=True,
        )
        aggregated[m]["calibration_ece"] = ece
        aggregated[m]["calibration_diagnostics"] = diag
        aggregated[m]["calib_n_unique_predictions"] = diag.get("n_unique_predictions")
        aggregated[m]["calib_n_nonempty_bins"] = diag.get("n_nonempty_bins")
        aggregated[m]["calib_degenerate_binning"] = diag.get("degenerate_binning")
        aggregated[m]["calib_binning_mode_used"] = diag.get("binning_mode_used")
        aggregated[m]["calib_binning_mode_requested"] = diag.get("binning_mode_requested")
        aggregated[m]["calib_fallback_applied"] = diag.get("fallback_applied")
    tests = run_tests(per_run_metrics, model_names)

    warnings = []
    for m in model_names:
        diag = aggregated[m].get("calibration_diagnostics", {})
        if diag and diag.get("degenerate_binning"):
            warnings.append(
                f"Calibration diagnostics: degenerate binning detected for model '{m}'."
            )

    scenario_name = derive_scenario_name(config)
    meta = {
        "scenario_name": scenario_name,
        "experiment_id": experiment_id,
        "tier": tier,
        "seed": seed,
    }
    extra_metadata = filter_extra_metadata(extra_metadata, meta)
    meta.update(extra_metadata)
    results = {
        "per_run_metrics": per_run_metrics,
        "aggregated_metrics": aggregated,
        "tests": tests,
        "warnings": warnings,
        "note": "Partial validation under controlled synthetic assumptions.",
        "metadata": meta,
        "reporting_version": reporting_version,
    }
    write_scenario_outputs(
        scenario_name=scenario_name,
        results=results,
        p_hat_lists=p_hat_lists,
        outcome_lists=outcome_lists,
        n_bins=n_bins,
        binning=calibration_binning,
        min_nonempty_bins=min_nonempty_bins,
        experiment_id=experiment_id,
        tier=tier,
        seed=seed,
        extra_metadata=extra_metadata,
        config_snapshot=sanitize_for_json(config),
    )
    return results


def draw_p_true(rng: np.random.Generator, cfg: Dict) -> float:
    """Draw p_true from a configured distribution."""

    dist_type = cfg.get("type", "fixed")
    if dist_type == "fixed":
        return float(cfg.get("value", 0.5))
    if dist_type == "uniform":
        return float(rng.random())
    if dist_type == "beta":
        a = float(cfg.get("a", 2.0))
        b = float(cfg.get("b", 2.0))
        return float(rng.beta(a, b))
    if dist_type == "two_point":
        p1 = float(cfg.get("p1", 0.3))
        p2 = float(cfg.get("p2", 0.7))
        return p1 if float(rng.random()) < 0.5 else p2
    raise ValueError(f"Unknown p_true_dist type: {dist_type}")


def build_trader_params(cfg: Dict) -> Dict:
    """Build TraderParams mapping from config."""

    params = {}
    for t_type, t_cfg in cfg.items():
        params[t_type] = TraderParams(
            signal_quality=float(t_cfg.get("signal_quality", 0.7)),
            noise_yes_prob=float(t_cfg.get("noise_yes_prob", 0.5)),
            herding_intensity=float(t_cfg.get("herding_intensity", 0.0)),
        )
    return params


def raw_parimutuel_prob(trade_tape, pool_path) -> float:
    """Raw parimutuel implied YES probability from the latest pool state."""

    if pool_path:
        return float(pool_path[-1][3])
    if trade_tape:
        return float(trade_tape[-1].implied_yes_after)
    return 0.5


def aggregate_metrics(per_run_metrics: List[Dict], model_names: List[str]) -> Dict:
    """Aggregate metrics across runs by mean (NaN-safe)."""

    agg = {m: {} for m in model_names}
    for m in model_names:
        metric_lists = {}
        for run in per_run_metrics:
            for k, v in run["metrics"][m].items():
                if v is not None:
                    metric_lists.setdefault(k, []).append(v)
        for k, vals in metric_lists.items():
            if k == "calibration_diagnostics":
                continue
            if not vals:
                agg[m][k] = None
                continue
            vals_arr = np.asarray(vals, dtype=float)
            if vals_arr.size == 0:
                agg[m][k] = None
            elif np.all(np.isnan(vals_arr)):
                agg[m][k] = np.nan
            else:
                agg[m][k] = float(np.nanmean(vals_arr))
        # Ensure legacy coverage aliases exist even if missing.
        if "coverage_90_ptrue" in agg[m]:
            agg[m]["coverage_90"] = agg[m].get("coverage_90_ptrue", np.nan)
        if "coverage_95_ptrue" in agg[m]:
            agg[m]["coverage_95"] = agg[m].get("coverage_95_ptrue", np.nan)
        # Ensure coverage columns are present even if NaN.
        for cov_key in [
            "coverage_90_outcome",
            "coverage_95_outcome",
            "coverage_90_ptrue",
            "coverage_95_ptrue",
            "coverage_90",
            "coverage_95",
        ]:
            agg[m].setdefault(cov_key, np.nan)
    return agg


def run_tests(per_run_metrics: List[Dict], model_names: List[str]) -> Dict:
    """Paired tests comparing CAPOPM vs each baseline on Brier score."""

    capopm_brier = [r["metrics"]["capopm"]["brier"] for r in per_run_metrics]
    capopm_mean = float(np.nanmean(capopm_brier)) if capopm_brier else np.nan
    results = {}
    p_values = []
    keys = []
    for m in model_names:
        if m == "capopm":
            continue
        brier_m = [r["metrics"][m]["brier"] for r in per_run_metrics]
        brier_mean = float(np.nanmean(brier_m)) if brier_m else np.nan
        diff_mean = capopm_mean - brier_mean
        t_stat, p_t = paired_t_test(capopm_brier, brier_m)
        w_stat, p_w = wilcoxon_signed_rank(capopm_brier, brier_m)
        ci_lo, ci_hi = bootstrap_ci(np.asarray(capopm_brier) - np.asarray(brier_m))
        results[m] = {
            "paired_t": {
                "stat": t_stat,
                "p_value": p_t,
                "p_value_str": format_p_value(p_t),
            },
            "wilcoxon": {
                "stat": w_stat,
                "p_value": p_w,
                "p_value_str": format_p_value(p_w),
            },
            "bootstrap_ci": {"low": ci_lo, "high": ci_hi},
            "metric": "brier",
            "diff_mean": diff_mean,
            "better_if_negative": True,
        }
        p_values.append(p_t)
        keys.append(m)

    holm = holm_correction(p_values)
    bonf = bonferroni_correction(p_values)
    for k, p_h, p_b in zip(keys, holm, bonf):
        results[k]["paired_t"]["p_holm"] = p_h
        results[k]["paired_t"]["p_bonferroni"] = p_b
        results[k]["paired_t"]["p_holm_str"] = format_p_value(p_h)
        results[k]["paired_t"]["p_bonferroni_str"] = format_p_value(p_b)
    return results


def format_p_value(p: float) -> str:
    """Format p-values to avoid zeros and improve readability."""

    p_safe = max(float(p), 1e-300)
    return f"{p_safe:.3e}"


def derive_scenario_name(config: Dict) -> str:
    """Derive a stable scenario name if not provided."""

    if config.get("scenario_name"):
        return str(config["scenario_name"])
    seed = config.get("seed", 123)
    suffix = f"seed{seed}"
    base = "baseline"
    return f"{base}_{suffix}"


def write_scenario_outputs(
    scenario_name: str,
    results: Dict,
    p_hat_lists: Dict[str, List[float]],
    outcome_lists: Dict[str, List[int]],
    n_bins: int,
    binning: str,
    min_nonempty_bins: int,
    experiment_id: Optional[str] = None,
    tier: Optional[str] = None,
    seed: Optional[int] = None,
    extra_metadata: Optional[Dict] = None,
    config_snapshot: Optional[Dict] = None,
    run_audit: bool = True,
) -> None:
    """Persist scenario-level artifacts for interpretability."""

    results_dir = os.path.join("results", scenario_name)
    os.makedirs(results_dir, exist_ok=True)

    serializable_results = sanitize_for_json(results)
    serializable_results["audit_file"] = "audit.json"
    serializable_results["audit_version"] = AUDIT_VERSION

    aggregated = serializable_results.get("aggregated_metrics", {})
    meta_columns = {
        "scenario_name": scenario_name,
        "experiment_id": experiment_id,
        "tier": tier,
        "seed": seed,
    }
    extra_metadata = filter_extra_metadata(extra_metadata, meta_columns)
    extra_keys = sorted(extra_metadata.keys())
    agg_path = None
    if aggregated:
        base_metric_columns = [
            "scenario_name",
            "experiment_id",
            "tier",
            "seed",
            "model",
            "abs_error_outcome",
            "brier",
            "log_score",
            "mae_prob",
            "calibration_ece",
            "calib_n_unique_predictions",
            "calib_n_nonempty_bins",
            "calib_degenerate_binning",
            "calib_binning_mode_used",
            "calib_binning_mode_requested",
            "calib_fallback_applied",
            "posterior_mean_bias",
            "posterior_variance_ratio",
            "mad_posterior_median",
            "wasserstein_distance_beta",
            "coverage_90_outcome",
            "coverage_95_outcome",
            "coverage_90_ptrue",
            "coverage_95_ptrue",
            # Legacy aliases at end
            "coverage_90",
            "coverage_95",
            "p_hat",
        ]
        known_metric_keys = set(base_metric_columns) | {"calibration_diagnostics"}
        extra_metric_keys = sorted(
            {
                key
                for vals in aggregated.values()
                for key in (vals or {}).keys()
                if key not in known_metric_keys
            }
        )
        metric_columns = base_metric_columns + extra_metric_keys + ["calibration_diagnostics"]
        metric_columns.extend(extra_keys)
        agg_path = os.path.join(results_dir, "metrics_aggregated.csv")
        rows = []
        for model_name, vals in aggregated.items():
            row = []
            for k in metric_columns:
                if k == "model":
                    row.append(model_name)
                elif k in meta_columns:
                    row.append(meta_columns[k])
                elif k in extra_metadata:
                    row.append(extra_metadata[k])
                elif k == "calibration_diagnostics":
                    row.append(json.dumps(vals.get(k, {})))
                else:
                    row.append(vals.get(k))
            rows.append(row)

        # Guard against empty/NaN model entries.
        model_idx = metric_columns.index("model")
        if all(
            (r[model_idx] is None)
            or (isinstance(r[model_idx], str) and r[model_idx].strip() == "")
            or (isinstance(r[model_idx], float) and np.isnan(r[model_idx]))
            for r in rows
        ):
            raise ValueError("metrics_aggregated.csv rows have empty model values; cannot write CSV.")

        with open(agg_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(metric_columns)
            writer.writerows(rows)

    tests = serializable_results.get("tests", {})
    tests_path = None
    if tests:
        tests_path = os.path.join(results_dir, "tests.csv")
        with open(tests_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            test_columns = [
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
            test_columns.extend(extra_keys)
            writer.writerow(test_columns)
            if isinstance(tests, list):
                for res in tests:
                    row = [
                        scenario_name,
                        experiment_id,
                        tier,
                        seed,
                        res.get("model"),
                        res.get("metric"),
                        res.get("test"),
                        res.get("stat"),
                        res.get("p_value"),
                        res.get("p_value_str"),
                        res.get("p_holm"),
                        res.get("p_holm_str"),
                        res.get("p_bonferroni"),
                        res.get("p_bonferroni_str"),
                        res.get("diff_mean"),
                        res.get("better_if_negative"),
                        res.get("ci_low"),
                        res.get("ci_high"),
                    ]
                    row.extend([extra_metadata[k] for k in extra_keys])
                    writer.writerow(row)
            else:
                for model, res in tests.items():
                    metric_name = res.get("metric")
                    diff_mean = res.get("diff_mean")
                    better_if_negative = res.get("better_if_negative")
                    safe = lambda v: max(v, 1e-300) if isinstance(v, (int, float)) else v
                    # paired t-test row
                    pt = res.get("paired_t", {})
                    row_base = [
                        scenario_name,
                        experiment_id,
                        tier,
                        seed,
                        model,
                        metric_name,
                        "paired_t",
                        pt.get("stat"),
                        safe(pt.get("p_value")),
                        pt.get("p_value_str"),
                        safe(pt.get("p_holm")),
                        pt.get("p_holm_str"),
                        safe(pt.get("p_bonferroni")),
                        pt.get("p_bonferroni_str"),
                        diff_mean,
                        better_if_negative,
                        None,
                        None,
                    ]
                    row_base.extend([extra_metadata[k] for k in extra_keys])
                    writer.writerow(row_base)
                    # wilcoxon row
                    wil = res.get("wilcoxon", {})
                    row_wil = [
                        scenario_name,
                        experiment_id,
                        tier,
                        seed,
                        model,
                        metric_name,
                        "wilcoxon",
                        wil.get("stat"),
                        safe(wil.get("p_value")),
                        wil.get("p_value_str"),
                        None,
                        None,
                        None,
                        None,
                        diff_mean,
                        better_if_negative,
                        None,
                        None,
                    ]
                    row_wil.extend([extra_metadata[k] for k in extra_keys])
                    writer.writerow(row_wil)
                    # bootstrap CI row
                    ci = res.get("bootstrap_ci", {})
                    row_ci = [
                        scenario_name,
                        experiment_id,
                        tier,
                        seed,
                        model,
                        metric_name,
                        "bootstrap_ci",
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        diff_mean,
                        better_if_negative,
                        ci.get("low"),
                        ci.get("high"),
                    ]
                    row_ci.extend([extra_metadata[k] for k in extra_keys])
                    writer.writerow(row_ci)

    # Reliability tables per model using the binning actually used for ECE.
    aggregated_diags = {
        m: aggregated.get(m, {}).get("calibration_diagnostics", {}) for m in aggregated.keys()
    }
    reliability_paths: List[str] = []
    for model, p_list in p_hat_lists.items():
        outcomes = outcome_lists.get(model, [])
        diag = aggregated_diags.get(model, {})
        binning_mode = diag.get("binning_mode_used", binning)
        rows, _ = reliability_table(
            p_list,
            outcomes,
            n_bins=n_bins,
            binning=binning_mode,
            min_nonempty_bins=min_nonempty_bins,
            allow_fallback=False,
        )
        rel_path = os.path.join(results_dir, f"reliability_{model}.csv")
        with open(rel_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            rel_columns = [
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
            rel_columns.extend(extra_keys)
            writer.writerow(rel_columns)
            for row in rows:
                rel_row = [
                    scenario_name,
                    experiment_id,
                    tier,
                    seed,
                    model,
                    row["bin_low"],
                    row["bin_high"],
                    row["count"],
                    row["mean_pred"],
                    row["mean_outcome"],
                ]
                rel_row.extend([extra_metadata[k] for k in extra_keys])
                writer.writerow(rel_row)
        reliability_paths.append(rel_path)

    config_hash = None
    if config_snapshot:
        config_hash = _write_config_snapshot(results_dir, config_snapshot)
        serializable_results.setdefault("metadata", {})["config_hash"] = config_hash

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    if run_audit:
        audit_report = run_audit_for_results(
            scenario_name=scenario_name,
            summary_path=summary_path,
            metrics_path=agg_path if aggregated else None,
            tests_path=tests_path,
            reliability_paths=reliability_paths,
            registry_root="results",
        )
        audit_path = os.path.join(results_dir, "audit.json")
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)


def sanitize_for_json(obj):
    """Convert numpy scalars/arrays to JSON-serializable Python types."""

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return [sanitize_for_json(v) for v in obj.tolist()]
    except Exception:
        pass
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    return obj


def filter_extra_metadata(extra_metadata: Optional[Dict], meta_columns: Dict) -> Dict:
    """Remove keys that collide with core metadata columns."""

    if not extra_metadata:
        return {}
    reserved = set(meta_columns.keys()) | {
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
        "bin_low",
        "bin_high",
        "count",
        "mean_pred",
        "mean_outcome",
        "calibration_diagnostics",
    }
    return {k: v for k, v in extra_metadata.items() if k not in reserved}


def _write_config_snapshot(results_dir: str, cfg: Dict) -> str:
    """Write config snapshot and return hash (reporting-only)."""

    cfg_clean = sanitize_for_json(cfg)
    snapshot_path = os.path.join(results_dir, "config_snapshot.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(cfg_clean, f, indent=2)
    h = hashlib.sha256()
    h.update(json.dumps(cfg_clean, sort_keys=True).encode("utf-8"))
    return h.hexdigest()
