
"""
Tier B Experiment B2: ASYMPTOTIC_RATE_CHECK.

Empirically checks variance and bias decay with increasing effective counts.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..likelihood import beta_binomial_update, counts_from_trade_tape, posterior_moments
from ..market_simulator import MarketConfig, Trade, simulate_market
from ..metrics.calibration import (
    calibration_ece,
    interval_coverage_outcome,
    interval_coverage_ptrue,
)
from ..metrics.distributional import (
    mad_posterior_median,
    posterior_mean_bias,
    posterior_variance_ratio,
    wasserstein_distance_beta,
)
from ..metrics.scoring import brier, log_score, mae_prob
from ..posterior import capopm_pipeline
from ..pricing import posterior_prices
from ..trader_model import build_traders
from ..stats.tests import (
    bootstrap_ci,
    bonferroni_correction,
    holm_correction,
    paired_t_test,
    wilcoxon_signed_rank,
)
from .runner import (
    REPORTING_VERSION,
    aggregate_metrics,
    build_trader_params,
    draw_p_true,
    format_p_value,
    write_scenario_outputs,
)
from ..invariant_runtime import (
    InvariantContext,
    current_context,
    require_invariant,
    reset_invariant_context,
    set_invariant_context,
    stable_config_hash,
)

EXPERIMENT_ID = "B2.ASYMPTOTIC_RATE_CHECK"
TIER = "B"

DEFAULT_N_TOTAL = [50, 100, 200, 400, 800, 1600]
DEFAULT_BASE_SEED = 202650
DEFAULT_N_RUNS = 30

LOG_EPS = 1e-18
BIAS_EPS = 1e-6


def run_b2_asymptotic_rate_check(
    n_total_grid: Iterable[int] = DEFAULT_N_TOTAL,
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
) -> List[Dict]:
    """Run B2 sweep in a single scenario directory."""

    scenario_name = f"B2_asymptotic_rate__seed{base_seed}"
    cfg = build_base_config()
    cfg["seed"] = int(base_seed)
    cfg["n_runs"] = int(n_runs)
    cfg["scenario_name"] = scenario_name
    cfg["experiment_id"] = EXPERIMENT_ID
    cfg["tier"] = TIER
    cfg["n_total_grid"] = [int(n) for n in n_total_grid]

    run_b2_scenario(cfg)
    return [{"scenario_name": scenario_name, "seed": base_seed}]


def scenario_id(base_seed: int) -> str:
    """Stable scenario name for B2."""

    return f"B2_asymptotic_rate__seed{base_seed}"


def compute_n_steps_for_total(n_total: int, arrivals_per_step: int) -> int:
    """Compute n_steps to reach or exceed target total trades."""

    if n_total <= 0:
        raise ValueError("n_total must be positive")
    if arrivals_per_step <= 0:
        raise ValueError("arrivals_per_step must be positive")
    return int(math.ceil(n_total / arrivals_per_step))

def run_b2_scenario(config: Dict) -> Dict:
    """Run B2 scenario across n_total levels and write artifacts."""

    cfg = copy.deepcopy(config)
    scenario_name = cfg.get("scenario_name")
    base_seed = int(cfg.get("seed", 123))
    n_runs = int(cfg.get("n_runs", 5))
    calibration_binning = cfg.get("calibration_binning", cfg.get("ece_binning", "equal_width"))
    min_nonempty_bins = int(cfg.get("ece_min_nonempty_bins", 5))
    n_bins = int(cfg.get("ece_bins", 10))

    model_names = cfg.get(
        "models",
        [
            "capopm",
            "capopm_stage1_only",
            "uncorrected",
            "raw_parimutuel",
        ],
    )

    per_run_metrics = []
    per_run_rows = []
    p_hat_lists = {m: [] for m in model_names}
    outcome_lists = {m: [] for m in model_names}
    config_hash = stable_config_hash(cfg)

    n_total_grid = cfg.get("n_total_grid", DEFAULT_N_TOTAL)
    base_arrivals = int(cfg["market"]["arrivals_per_step"])

    fit_data: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = {
        m: {} for m in model_names
    }

    for level_idx, n_total_target in enumerate(n_total_grid):
        arrivals = base_arrivals
        n_steps = compute_n_steps_for_total(int(n_total_target), arrivals)
        n_total = int(n_steps * arrivals)
        level_seed = base_seed + level_idx * 1000
        rng_level = np.random.default_rng(level_seed)

        for run_idx in range(n_runs):
            run_seed = int(rng_level.integers(0, 2**32 - 1))
            rng = np.random.default_rng(run_seed)

            ctx_token = set_invariant_context(
                InvariantContext(
                    experiment_id=cfg.get("experiment_id"),
                    scenario_name=cfg.get("scenario_name"),
                    run_seed=run_seed,
                    config_hash=config_hash,
                )
            )
            try:
                p_true = draw_p_true(rng, cfg.get("p_true_dist", {"type": "fixed", "value": 0.55}))
                outcome = 1 if float(rng.random()) < p_true else 0

                traders = build_traders(
                    n_traders=int(cfg["traders"]["n_traders"]),
                    proportions=cfg["traders"]["proportions"],
                    params_by_type=build_trader_params(cfg["traders"].get("params", {})),
                )

                market_cfg = MarketConfig(**{**cfg["market"], "n_steps": n_steps, "arrivals_per_step": arrivals})
                trade_tape, pool_path = simulate_market(rng, market_cfg, traders, p_true)
                y_raw, n_raw = counts_from_trade_tape(trade_tape)

                base_out = capopm_pipeline(
                    rng=rng,
                    trade_tape=trade_tape,
                    structural_cfg=cfg["structural_cfg"],
                    ml_cfg=cfg["ml_cfg"],
                    prior_cfg=cfg["prior_cfg"],
                    stage1_cfg={"enabled": False},
                    stage2_cfg={"enabled": False},
                )
                var_independent = posterior_moments(base_out["alpha_post"], base_out["beta_post"])[1]

                model_outputs = build_model_outputs(rng, trade_tape, pool_path, cfg, base_out, y_raw, n_raw)

                metrics = {}
                coverage_include_outcome = bool(
                    cfg.get("coverage_include_outcome", cfg.get("include_outcome_coverage", False))
                )
                for model, out in model_outputs.items():
                    p_hat = out["p_hat"]
                    alpha = out["alpha"]
                    beta = out["beta"]
                    n_eff = out.get("n_effective", n_raw)
                    metrics[model] = {
                        "brier": brier(p_true, p_hat),
                        "log_score": log_score(p_hat, outcome),
                        "mae_prob": mae_prob(p_true, p_hat),
                        "abs_error_outcome": abs(p_hat - outcome),
                        "calibration_ece": None,
                        "calibration_diagnostics": {},
                        "posterior_mean_bias": posterior_mean_bias(p_hat, p_true),
                        "p_hat": p_hat,
                        "posterior_variance": float("nan"),
                        "log_variance": float("nan"),
                        "n_effective": n_eff,
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
                                "coverage_90_outcome": float("nan"),
                                "coverage_95_outcome": float("nan"),
                                "coverage_outcome_90": float("nan"),
                                "coverage_outcome_95": float("nan"),
                                # Backward-compatible aliases
                                "coverage_90_ptrue": cov90,
                                "coverage_95_ptrue": cov95,
                                "posterior_variance": var_adj,
                                "log_variance": math.log(max(var_adj, LOG_EPS)),
                            }
                        )
                        if coverage_include_outcome:
                            cov_out_90 = interval_coverage_outcome(alpha, beta, outcome, 0.90)
                            cov_out_95 = interval_coverage_outcome(alpha, beta, outcome, 0.95)
                            metrics[model].update(
                                {
                                    "coverage_90_outcome": cov_out_90,
                                    "coverage_95_outcome": cov_out_95,
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
                                "coverage_90": float("nan"),
                                "coverage_95": float("nan"),
                                "coverage_90_ptrue": float("nan"),
                                "coverage_95_ptrue": float("nan"),
                                "coverage_90_outcome": float("nan") if coverage_include_outcome else float("nan"),
                                "coverage_95_outcome": float("nan") if coverage_include_outcome else float("nan"),
                                "coverage_outcome_90": float("nan") if coverage_include_outcome else float("nan"),
                                "coverage_outcome_95": float("nan") if coverage_include_outcome else float("nan"),
                            }
                        )

                    p_hat_lists[model].append(p_hat)
                    outcome_lists[model].append(int(outcome))

                    per_run_rows.append(
                        {
                            "scenario_name": cfg.get("scenario_name"),
                            "experiment_id": cfg.get("experiment_id"),
                            "tier": cfg.get("tier"),
                            "model": model,
                            "seed": run_seed,
                            "n_total_level": n_total,
                            "n_steps": n_steps,
                            "arrivals_per_step": arrivals,
                            "n_effective": n_eff,
                            **metrics[model],
                        }
                    )

                    if model in fit_data:
                        bias_abs = abs(metrics[model]["posterior_mean_bias"])
                        fit_data[model].setdefault(n_total, []).append(
                            (float(n_eff), float(metrics[model]["posterior_variance"]), float(bias_abs))
                        )

                per_run_metrics.append(
                    {
                        "run": run_idx,
                        "p_true": p_true,
                        "outcome": outcome,
                        "metrics": metrics,
                        "invariants": [vars(rec) for rec in (current_context().invariant_log if current_context() else [])],
                        "fallbacks": [vars(rec) for rec in (current_context().fallback_log if current_context() else [])],
                    }
                )
            finally:
                reset_invariant_context(ctx_token)

    aggregated = aggregate_metrics(per_run_metrics, model_names)
    scenario_ctx = InvariantContext(
        experiment_id=cfg.get("experiment_id"),
        scenario_name=scenario_name,
        run_seed=base_seed,
        config_hash=config_hash,
    )
    scenario_token = set_invariant_context(scenario_ctx)
    try:
        ece_by_model = {}
        if "capopm" in aggregated and "uncorrected" in aggregated:
            cap_brier = aggregated["capopm"].get("brier")
            base_brier = aggregated["uncorrected"].get("brier")
            if cap_brier is not None and base_brier is not None:
                require_invariant(
                    cap_brier <= base_brier + 1e-8,
                    invariant_id="M-3",
                    message="Expected-loss non-worsening (Brier, synthetic)",
                    tolerance=1e-8,
                    data={"capopm_brier": cap_brier, "uncorrected_brier": base_brier},
                )
            cap_log = aggregated["capopm"].get("log_score")
            base_log = aggregated["uncorrected"].get("log_score")
            if cap_log is not None and base_log is not None:
                require_invariant(
                    cap_log + 1e-8 >= base_log,
                    invariant_id="M-3",
                    message="Expected-loss non-worsening (log_score, synthetic)",
                    tolerance=1e-8,
                    data={"capopm_log": cap_log, "uncorrected_log": base_log},
                )
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
            ece_by_model[m] = ece

        fit_rows, fit_summary = build_fit_rows(aggregated, fit_data, cfg)
        metrics_rows = finalize_metrics_rows(per_run_rows, fit_rows, ece_by_model)

        tests = run_b2_tests(per_run_metrics)

        warnings = []
        for m in model_names:
            diag = aggregated[m].get("calibration_diagnostics", {})
            if diag and diag.get("degenerate_binning"):
                warnings.append(
                    f"Calibration diagnostics: degenerate binning detected for model '{m}'."
                )
        scenario_invariant_log = [vars(rec) for rec in scenario_ctx.invariant_log]
        scenario_fallback_log = [vars(rec) for rec in scenario_ctx.fallback_log]
    finally:
        reset_invariant_context(scenario_token)

    summary = build_summary(fit_summary)
    scenario_name = cfg.get("scenario_name")
    results = {
        "per_run_metrics": per_run_metrics,
        "aggregated_metrics": aggregated,
        "tests": tests,
        "warnings": warnings,
        "note": "Rate fits use log-log regression on median variance/bias by n_total.",
        "metadata": build_metadata(cfg),
        "reporting_version": REPORTING_VERSION,
        "status": summary["status"],
        "sweep_params": {"n_total_grid": n_total_grid},
        "summary": summary,
        "scenario_invariants": scenario_invariant_log,
        "scenario_fallbacks": scenario_fallback_log,
        "grid_status": {
            "grid_axes": ["n_total_grid"],
            "grid_point": {"n_total_grid": n_total_grid},
        },
    }
    results["metadata"]["config_hash"] = config_hash

    write_scenario_outputs(
        scenario_name=scenario_name,
        results=results,
        p_hat_lists=p_hat_lists,
        outcome_lists=outcome_lists,
        n_bins=n_bins,
        binning=calibration_binning,
        min_nonempty_bins=min_nonempty_bins,
        experiment_id=cfg.get("experiment_id"),
        tier=cfg.get("tier"),
        seed=base_seed,
        extra_metadata={"n_total_grid": ",".join(str(n) for n in n_total_grid)},
        config_snapshot=cfg,
    )

    write_metrics_aggregated(cfg, metrics_rows)
    return results

def build_fit_rows(aggregated: Dict, fit_data: Dict, cfg: Dict) -> Tuple[List[Dict], Dict]:
    """Compute log-log slopes and build fit rows per model."""

    fit_rows = []
    fit_summary = {}
    for model, level_data in fit_data.items():
        n_vals = []
        var_vals = []
        bias_vals = []
        for level, entries in sorted(level_data.items()):
            n_list = [e[0] for e in entries if e[0] > 0 and math.isfinite(e[0])]
            var_list = [e[1] for e in entries if e[1] > 0 and math.isfinite(e[1])]
            bias_list = [e[2] for e in entries if e[2] >= 0 and math.isfinite(e[2])]
            if not n_list or not var_list or not bias_list:
                continue
            n_vals.append(float(np.median(n_list)))
            var_vals.append(float(np.median(var_list)))
            bias_vals.append(float(np.median(bias_list)))

        rate_var, r2_var = fit_loglog(n_vals, var_vals, LOG_EPS)
        rate_bias, r2_bias = fit_loglog(n_vals, bias_vals, BIAS_EPS)

        fit_summary[model] = {
            "rate_var_vs_n": rate_var,
            "r2_var": r2_var,
            "rate_bias_vs_n": rate_bias,
            "r2_bias": r2_bias,
        }

        agg = aggregated.get(model, {})
        fit_rows.append(
            {
                "scenario_name": cfg.get("scenario_name"),
                "experiment_id": cfg.get("experiment_id"),
                "tier": cfg.get("tier"),
                "model": model,
                "seed": -1,
                "n_total_level": -1,
                "n_steps": -1,
                "arrivals_per_step": -1,
                "n_effective": agg.get("n_effective", np.nan),
                "brier": agg.get("brier", np.nan),
                "log_score": agg.get("log_score", np.nan),
                "mae_prob": agg.get("mae_prob", np.nan),
                "abs_error_outcome": agg.get("abs_error_outcome", np.nan),
                "calibration_ece": agg.get("calibration_ece", np.nan),
                "posterior_mean_bias": agg.get("posterior_mean_bias", np.nan),
                "posterior_variance_ratio": agg.get("posterior_variance_ratio", np.nan),
                "posterior_variance": agg.get("posterior_variance", np.nan),
                "log_variance": agg.get("log_variance", np.nan),
                "coverage_90_ptrue": agg.get("coverage_90_ptrue", np.nan),
                "coverage_95_ptrue": agg.get("coverage_95_ptrue", np.nan),
                "p_hat": agg.get("p_hat", np.nan),
                "rate_var_vs_n": rate_var,
                "rate_bias_vs_n": rate_bias,
                "r2_var": r2_var,
                "r2_bias": r2_bias,
            }
        )

    return fit_rows, fit_summary


def fit_loglog(n_vals: List[float], y_vals: List[float], eps: float) -> Tuple[float, float]:
    """Fit slope and r2 for log(y+eps) ~ a + b log(n)."""

    if len(n_vals) < 2 or len(y_vals) < 2:
        return float("nan"), float("nan")
    x = np.log(np.asarray(n_vals, dtype=float))
    y = np.log(np.asarray(y_vals, dtype=float) + eps)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return slope, r2


def finalize_metrics_rows(
    per_run_rows: List[Dict],
    fit_rows: List[Dict],
    ece_by_model: Dict[str, float],
) -> List[Dict]:
    """Attach calibration ECE to per-run rows and append fit rows."""

    rows = []
    for row in per_run_rows:
        model = row.get("model")
        row = dict(row)
        row["calibration_ece"] = ece_by_model.get(model)
        row.setdefault("rate_var_vs_n", float("nan"))
        row.setdefault("rate_bias_vs_n", float("nan"))
        row.setdefault("r2_var", float("nan"))
        row.setdefault("r2_bias", float("nan"))
        rows.append(row)
    rows.extend(fit_rows)
    return rows


def write_metrics_aggregated(cfg: Dict, rows: List[Dict]) -> None:
    """Write metrics_aggregated.csv with per-seed rows plus fit rows."""

    import csv
    import os

    results_dir = os.path.join("results", cfg.get("scenario_name"))
    os.makedirs(results_dir, exist_ok=True)

    columns = [
        "scenario_name",
        "experiment_id",
        "tier",
        "model",
        "seed",
        "n_total_level",
        "n_steps",
        "arrivals_per_step",
        "n_effective",
        "brier",
        "log_score",
        "mae_prob",
        "abs_error_outcome",
        "calibration_ece",
        "posterior_mean_bias",
        "posterior_variance_ratio",
        "posterior_variance",
        "log_variance",
        "coverage_90_ptrue",
        "coverage_95_ptrue",
        "p_hat",
        "rate_var_vs_n",
        "rate_bias_vs_n",
        "r2_var",
        "r2_bias",
    ]

    def sort_key(r):
        return (
            int(r.get("n_total_level", -1)),
            str(r.get("model", "")),
            int(r.get("seed", -1)),
        )

    sorted_rows = sorted(rows, key=sort_key)
    path = os.path.join(results_dir, "metrics_aggregated.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in sorted_rows:
            writer.writerow([row.get(col) for col in columns])


def build_model_outputs(
    rng: np.random.Generator,
    trade_tape: List[Trade],
    pool_path,
    config: Dict,
    base_out: Dict,
    y_raw: float,
    n_raw: float,
) -> Dict[str, Dict]:
    """Compute final outputs and effective counts per model."""

    model_outputs = {}
    model_names = config.get("models", [])
    stage1_full = config.get("stage1_cfg", {"enabled": True})
    stage2_full = config.get("stage2_cfg", {"enabled": True})

    for model in model_names:
        if model == "capopm":
            out = capopm_pipeline(
                rng=rng,
                trade_tape=trade_tape,
                structural_cfg=config["structural_cfg"],
                ml_cfg=config["ml_cfg"],
                prior_cfg=config["prior_cfg"],
                stage1_cfg=stage1_full,
                stage2_cfg=stage2_full,
            )
            model_outputs[model] = {
                "p_hat": out["mixture_mean"] if out.get("mixture_enabled") else out["pi_yes"],
                "alpha": out["alpha_post"],
                "beta": out["beta_post"],
                "n_effective": out.get("n_stage2") or out.get("n_stage1") or out.get("n"),
            }
        elif model == "capopm_stage1_only":
            out = capopm_pipeline(
                rng=rng,
                trade_tape=trade_tape,
                structural_cfg=config["structural_cfg"],
                ml_cfg=config["ml_cfg"],
                prior_cfg=config["prior_cfg"],
                stage1_cfg=stage1_full,
                stage2_cfg={"enabled": False},
            )
            model_outputs[model] = {
                "p_hat": out["pi_yes"],
                "alpha": out["alpha_post"],
                "beta": out["beta_post"],
                "n_effective": out.get("n_stage1") or out.get("n"),
            }
        elif model == "uncorrected":
            model_outputs[model] = {
                "p_hat": base_out["pi_yes"],
                "alpha": base_out["alpha_post"],
                "beta": base_out["beta_post"],
                "n_effective": base_out.get("n"),
            }
        elif model == "raw_parimutuel":
            model_outputs[model] = {
                "p_hat": raw_parimutuel_prob(trade_tape, pool_path),
                "alpha": None,
                "beta": None,
                "n_effective": n_raw,
            }
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
            model_outputs[model] = {
                "p_hat": out["pi_yes"],
                "alpha": out["alpha_post"],
                "beta": out["beta_post"],
                "n_effective": out.get("n"),
            }
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
            model_outputs[model] = {
                "p_hat": out["pi_yes"],
                "alpha": out["alpha_post"],
                "beta": out["beta_post"],
                "n_effective": out.get("n"),
            }
        elif model == "beta_1_1":
            alpha0, beta0 = 1.0, 1.0
            alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
            p_hat, _ = posterior_prices(alpha, beta)
            model_outputs[model] = {
                "p_hat": p_hat,
                "alpha": alpha,
                "beta": beta,
                "n_effective": n_raw,
            }
        elif model == "beta_0_5_0_5":
            alpha0, beta0 = 0.5, 0.5
            alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
            p_hat, _ = posterior_prices(alpha, beta)
            model_outputs[model] = {
                "p_hat": p_hat,
                "alpha": alpha,
                "beta": beta,
                "n_effective": n_raw,
            }
        else:
            raise ValueError(f"Unknown model: {model}")

        require_invariant(
            0.0 <= model_outputs[model]["p_hat"] <= 1.0,
            invariant_id="M-1",
            message="Model price in [0,1]",
            tolerance=1e-8,
            data={"model": model, "p_hat": float(model_outputs[model]["p_hat"])},
        )

    return model_outputs


def run_b2_tests(per_run_metrics: List[Dict]) -> List[Dict]:
    """Paired tests for posterior variance and bias (CAPOPM vs uncorrected)."""

    metrics = [
        {"name": "posterior_variance", "better_if_negative": True},
        {"name": "posterior_mean_bias", "better_if_negative": True},
    ]
    rows = []
    for metric in metrics:
        metric_name = metric["name"]
        better_if_negative = metric["better_if_negative"]
        cap_vals, base_vals = paired_finite_values(per_run_metrics, metric_name)
        if len(cap_vals) < 2:
            continue
        t_stat, p_t = paired_t_test(cap_vals, base_vals)
        w_stat, p_w = wilcoxon_signed_rank(cap_vals, base_vals)
        diff = np.asarray(cap_vals) - np.asarray(base_vals)
        ci_lo, ci_hi = bootstrap_ci(diff)
        diff_mean = float(np.nanmean(diff)) if diff.size > 0 else float("nan")
        p_h = holm_correction([p_t])[0]
        p_b = bonferroni_correction([p_t])[0]

        rows.append(
            {
                "model": "capopm",
                "metric": metric_name,
                "test": "paired_t",
                "stat": t_stat,
                "p_value": max(p_t, 1e-300),
                "p_value_str": format_p_value(p_t),
                "p_holm": max(p_h, 1e-300),
                "p_holm_str": format_p_value(p_h),
                "p_bonferroni": max(p_b, 1e-300),
                "p_bonferroni_str": format_p_value(p_b),
                "diff_mean": diff_mean,
                "better_if_negative": better_if_negative,
                "ci_low": None,
                "ci_high": None,
            }
        )
        rows.append(
            {
                "model": "capopm",
                "metric": metric_name,
                "test": "wilcoxon",
                "stat": w_stat,
                "p_value": max(p_w, 1e-300),
                "p_value_str": format_p_value(p_w),
                "p_holm": None,
                "p_holm_str": None,
                "p_bonferroni": None,
                "p_bonferroni_str": None,
                "diff_mean": diff_mean,
                "better_if_negative": better_if_negative,
                "ci_low": None,
                "ci_high": None,
            }
        )
        rows.append(
            {
                "model": "capopm",
                "metric": metric_name,
                "test": "bootstrap_ci",
                "stat": None,
                "p_value": None,
                "p_value_str": None,
                "p_holm": None,
                "p_holm_str": None,
                "p_bonferroni": None,
                "p_bonferroni_str": None,
                "diff_mean": diff_mean,
                "better_if_negative": better_if_negative,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
            }
        )

    return rows


def paired_finite_values(per_run_metrics: List[Dict], metric_name: str) -> Tuple[List[float], List[float]]:
    """Collect paired values for CAPOPM vs uncorrected for a metric."""

    cap_vals = []
    base_vals = []
    for run in per_run_metrics:
        cap = run["metrics"]["capopm"].get(metric_name)
        base = run["metrics"]["uncorrected"].get(metric_name)
        if cap is None or base is None:
            continue
        if isinstance(cap, float) and np.isnan(cap):
            continue
        if isinstance(base, float) and np.isnan(base):
            continue
        cap_vals.append(float(cap))
        base_vals.append(float(base))
    return cap_vals, base_vals


def build_summary(fit_summary: Dict) -> Dict:
    """Build summary with slope diagnostics and pass flags."""

    cap = fit_summary.get("capopm", {})
    unc = fit_summary.get("uncorrected", {})
    rate_var_cap = cap.get("rate_var_vs_n", float("nan"))
    rate_var_unc = unc.get("rate_var_vs_n", float("nan"))
    rate_bias_cap = cap.get("rate_bias_vs_n", float("nan"))
    pass_var = rate_var_cap < 0.0 if math.isfinite(rate_var_cap) else False
    pass_bias = rate_bias_cap < 0.0 if math.isfinite(rate_bias_cap) else False
    slope_diff = rate_var_cap - rate_var_unc if math.isfinite(rate_var_cap) and math.isfinite(rate_var_unc) else float("nan")

    return {
        "status": {
            "pass": bool(pass_var and pass_bias),
            "criteria": {
                "rate_var_vs_n_negative": True,
                "rate_bias_vs_n_negative": True,
            },
            "metrics": {
                "rate_var_vs_n_capopm": rate_var_cap,
                "rate_var_vs_n_uncorrected": rate_var_unc,
                "rate_bias_vs_n_capopm": rate_bias_cap,
                "rate_var_slope_diff": slope_diff,
            },
        }
    }


def build_metadata(config: Dict) -> Dict:
    """Build summary metadata."""

    return {
        "scenario_name": config.get("scenario_name"),
        "experiment_id": config.get("experiment_id"),
        "tier": config.get("tier"),
        "seed": int(config.get("seed", 0)),
        "structural_prior_mode": config.get("structural_cfg", {}).get("mode", "surrogate_heston"),
    }


def raw_parimutuel_prob(trade_tape, pool_path) -> float:
    """Raw parimutuel implied YES probability from the latest pool state."""

    if pool_path:
        return float(pool_path[-1][3])
    if trade_tape:
        return float(trade_tape[-1].implied_yes_after)
    return 0.5


def build_base_config() -> Dict:
    """Base configuration for B2 with fixed p_true and stable market."""

    cfg = {
        "p_true_dist": {"type": "fixed", "value": 0.55},
        "models": [
            "capopm",
            "capopm_stage1_only",
            "uncorrected",
            "raw_parimutuel",
        ],
        "traders": {
            "n_traders": 60,
            "proportions": {
                "informed": 0.5,
                "adversarial": 0.0,
                "noise": 0.5,
            },
            "params": {
                "informed": {
                    "signal_quality": 0.7,
                    "noise_yes_prob": 0.5,
                    "herding_intensity": 0.0,
                },
                "adversarial": {
                    "signal_quality": 0.7,
                    "noise_yes_prob": 0.5,
                    "herding_intensity": 0.0,
                },
                "noise": {
                    "signal_quality": 0.5,
                    "noise_yes_prob": 0.5,
                    "herding_intensity": 0.0,
                },
            },
        },
        "market": {
            "n_steps": 20,
            "arrivals_per_step": 5,
            "fee_rate": 0.01,
            "initial_yes_pool": 1.0,
            "initial_no_pool": 1.0,
            "signal_model": "conditional_on_state",
            "use_realized_state_for_signals": True,
            "herding_enabled": False,
            "size_dist": "fixed",
            "size_dist_params": {"size": 1.0},
        },
        "structural_cfg": {
            "T": 1.0,
            "K": 1.0,
            "S0": 1.0,
            "V0": 0.04,
            "kappa": 1.0,
            "theta": 0.04,
            "xi": 0.2,
            "rho": -0.3,
            "alpha": 0.7,
            "lambda": 0.1,
        },
        "ml_cfg": {
            "base_prob": 0.55,
            "bias": -0.02,
            "noise_std": 0.01,
            "calibration": 1.0,
            "r_ml": 0.8,
        },
        "prior_cfg": {"n_str": 10.0, "n_ml_eff": 4.0, "n_ml_scale": 1.0},
        "stage1_cfg": {
            "enabled": True,
            "w_min": 0.1,
            "w_max": 1.25,
            "longshot_ref_p": 0.5,
            "longshot_gamma": 0.8,
            "herding_lambda": 0.0,
            "herding_window": 50,
        },
        "stage2_cfg": {
            "enabled": True,
            "mode": "offsets",
            "delta_plus": 0.0,
            "delta_minus": 0.0,
        },
        "ece_bins": 10,
        "calibration_binning": "equal_width",
        "ece_min_nonempty_bins": 5,
        "coverage_include_outcome": False,
    }

    return copy.deepcopy(cfg)
