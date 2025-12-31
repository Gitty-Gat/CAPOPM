
"""
Tier B Experiment B3: MISSPECIFICATION_REGRET_GRID.

Evaluates regret under structural and ML prior misspecification grid.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List

import numpy as np

from ..likelihood import counts_from_trade_tape, posterior_moments
from ..market_simulator import MarketConfig, simulate_market
from ..metrics.calibration import calibration_ece, interval_coverage_outcome, interval_coverage_ptrue
from ..metrics.distributional import mad_posterior_median, posterior_mean_bias, posterior_variance_ratio, wasserstein_distance_beta
from ..metrics.scoring import brier, log_score, mae_prob
from ..posterior import capopm_pipeline
from ..trader_model import build_traders
from ..stats.tests import bootstrap_ci, bonferroni_correction, holm_correction, paired_t_test, wilcoxon_signed_rank
from .runner import (
    REPORTING_VERSION,
    aggregate_metrics,
    build_trader_params,
    draw_p_true,
    format_p_value,
    write_scenario_outputs,
)

EXPERIMENT_ID = "B3.MISSPECIFICATION_REGRET_GRID"
TIER = "B"

DEFAULT_STRUCT_MIS = [0.0, 0.2]
DEFAULT_ML_MIS = [0.0, 0.1]
DEFAULT_BASE_SEED = 202660
DEFAULT_N_RUNS = 60


def run_b3_misspecification_regret_grid(
    structural_mis_grid: Iterable[float] = DEFAULT_STRUCT_MIS,
    ml_bias_grid: Iterable[float] = DEFAULT_ML_MIS,
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
) -> List[Dict]:
    results = []
    for i, struct_shift in enumerate(structural_mis_grid):
        for j, ml_bias in enumerate(ml_bias_grid):
            seed = base_seed + i * 100 + j
            scenario_name = scenario_id(struct_shift, ml_bias, seed)
            cfg = build_base_config()
            cfg["seed"] = seed
            cfg["n_runs"] = n_runs
            cfg["scenario_name"] = scenario_name
            cfg["experiment_id"] = EXPERIMENT_ID
            cfg["tier"] = TIER
            cfg["sweep_params"] = {
                "structural_shift": float(struct_shift),
                "ml_bias": float(ml_bias),
            }
            apply_structural_mis(cfg, struct_shift)
            apply_ml_mis(cfg, ml_bias)
            run_b3_scenario(cfg)
            results.append(
                {
                    "scenario_name": scenario_name,
                    "seed": seed,
                    "sweep_params": cfg["sweep_params"],
                }
            )
    return results


def run_b3_scenario(config: Dict) -> Dict:
    cfg = copy.deepcopy(config)
    seed = int(cfg.get("seed", 123))
    n_runs = int(cfg.get("n_runs", 5))
    calibration_binning = cfg.get("calibration_binning", cfg.get("ece_binning", "equal_width"))
    min_nonempty_bins = int(cfg.get("ece_min_nonempty_bins", 5))
    n_bins = int(cfg.get("ece_bins", 10))
    rng_master = np.random.default_rng(seed)

    model_names = cfg.get(
        "models",
        ["capopm", "structural_only", "ml_only", "uncorrected", "raw_parimutuel"],
    )

    per_run_metrics = []
    p_hat_lists = {m: [] for m in model_names}
    outcome_lists = {m: [] for m in model_names}

    for run_idx in range(n_runs):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        p_true = draw_p_true(rng, cfg.get("p_true_dist", {"type": "fixed", "value": 0.55}))
        outcome = 1 if float(rng.random()) < p_true else 0

        traders = build_traders(
            n_traders=int(cfg["traders"]["n_traders"]),
            proportions=cfg["traders"]["proportions"],
            params_by_type=build_trader_params(cfg["traders"].get("params", {})),
        )

        market_cfg = MarketConfig(**cfg["market"])
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
        coverage_include_outcome = bool(cfg.get("coverage_include_outcome", cfg.get("include_outcome_coverage", False)))
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
                "regret_brier": None,
                "regret_log_bad": None,
                "regret_abs_error": None,
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

        uncorrected = metrics["uncorrected"]
        for model in metrics:
            metrics[model]["regret_brier"] = metrics[model]["brier"] - uncorrected["brier"]
            metrics[model]["regret_log_bad"] = uncorrected["log_score"] - metrics[model]["log_score"]
            metrics[model]["regret_abs_error"] = metrics[model]["abs_error_outcome"] - uncorrected["abs_error_outcome"]

        per_run_metrics.append({"run": run_idx, "p_true": p_true, "outcome": outcome, "metrics": metrics})

    aggregated = aggregate_metrics(per_run_metrics, model_names)
    for m in model_names:
        if len(p_hat_lists[m]) != len(outcome_lists[m]) or len(p_hat_lists[m]) < 2:
            raise ValueError("ECE requires at least 2 matched predictions/outcomes per model")
        ece, diag = calibration_ece(
            p_hat_lists[m], outcome_lists[m], n_bins=n_bins, binning=calibration_binning, min_nonempty_bins=min_nonempty_bins, allow_fallback=True
        )
        aggregated[m]["calibration_ece"] = ece
        aggregated[m]["calibration_diagnostics"] = diag
        aggregated[m]["calib_n_unique_predictions"] = diag.get("n_unique_predictions")
        aggregated[m]["calib_n_nonempty_bins"] = diag.get("n_nonempty_bins")
        aggregated[m]["calib_degenerate_binning"] = diag.get("degenerate_binning")
        aggregated[m]["calib_binning_mode_used"] = diag.get("binning_mode_used")
        aggregated[m]["calib_binning_mode_requested"] = diag.get("binning_mode_requested")
        aggregated[m]["calib_fallback_applied"] = diag.get("fallback_applied")

    tests = run_b3_tests(per_run_metrics)

    warnings = []
    for m in model_names:
        diag = aggregated[m].get("calibration_diagnostics", {})
        if diag and diag.get("degenerate_binning"):
            warnings.append(f"Calibration diagnostics: degenerate binning detected for model '{m}'.")

    summary = build_summary(aggregated)
    scenario_name = cfg.get("scenario_name")
    results = {
        "per_run_metrics": per_run_metrics,
        "aggregated_metrics": aggregated,
        "tests": tests,
        "warnings": warnings,
        "note": "Regret fields are relative to uncorrected (bad if positive).",
        "metadata": build_metadata(cfg),
        "reporting_version": REPORTING_VERSION,
        "status": summary["status"],
        "sweep_params": cfg.get("sweep_params", {}),
        "summary": summary,
    }

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
        seed=seed,
        extra_metadata=cfg.get("sweep_params", {}),
        config_snapshot=cfg,
    )
    return results


def build_model_outputs(
    rng: np.random.Generator,
    trade_tape,
    pool_path,
    config: Dict,
    base_out: Dict,
    y_raw: float,
    n_raw: float,
) -> Dict[str, Dict]:
    """Compute final outputs and priors per model."""

    model_outputs = {}
    model_names = config.get("models", [])
    stage1_full = config.get("stage1_cfg", {"enabled": True})
    stage2_full = config.get("stage2_cfg", {"enabled": False})

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
                "p_hat": out["pi_yes"] if not out.get("mixture_enabled") else out["mixture_mean"],
                "alpha": out["alpha_post"],
                "beta": out["beta_post"],
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
            }
        elif model == "uncorrected":
            model_outputs[model] = {
                "p_hat": base_out["pi_yes"],
                "alpha": base_out["alpha_post"],
                "beta": base_out["beta_post"],
            }
        elif model == "raw_parimutuel":
            model_outputs[model] = {
                "p_hat": raw_parimutuel_prob(trade_tape, pool_path),
                "alpha": None,
                "beta": None,
            }
        elif model == "beta_1_1":
            alpha0, beta0 = 1.0, 1.0
            alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
            p_hat, _ = posterior_prices(alpha, beta)
            model_outputs[model] = {"p_hat": p_hat, "alpha": alpha, "beta": beta}
        elif model == "beta_0_5_0_5":
            alpha0, beta0 = 0.5, 0.5
            alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
            p_hat, _ = posterior_prices(alpha, beta)
            model_outputs[model] = {"p_hat": p_hat, "alpha": alpha, "beta": beta}
        else:
            raise ValueError(f"Unknown model: {model}")

    return model_outputs

def apply_structural_mis(config: Dict, shift: float) -> None:
    """Apply structural misspecification via S0 scaling."""

    config["structural_cfg"] = dict(config.get("structural_cfg", {}))
    config["structural_cfg"]["S0"] = float(config["structural_cfg"].get("S0", 1.0)) * (1.0 + float(shift))


def apply_ml_mis(config: Dict, bias: float) -> None:
    """Apply ML miscalibration via bias knob."""

    config["ml_cfg"] = dict(config.get("ml_cfg", {}))
    config["ml_cfg"]["bias"] = float(config["ml_cfg"].get("bias", 0.0)) + float(bias)


def run_b3_tests(per_run_metrics: List[Dict]) -> List[Dict]:
    metrics = [
        {"name": "regret_brier", "better_if_negative": True},
        {"name": "regret_log_bad", "better_if_negative": True},
        {"name": "regret_abs_error", "better_if_negative": True},
    ]
    rows = []
    comparisons = ["capopm", "structural_only", "ml_only"]
    for metric in metrics:
        metric_name = metric["name"]
        better_if_negative = metric["better_if_negative"]
        for model in comparisons:
            cap_vals, base_vals = paired_finite_values(per_run_metrics, metric_name, model)
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
                    "model": model,
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
                    "model": model,
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
                    "model": model,
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


def paired_finite_values(per_run_metrics: List[Dict], metric_name: str, model: str):
    model_vals = []
    base_vals = []
    for run in per_run_metrics:
        cap = run["metrics"][model].get(metric_name)
        base = run["metrics"]["uncorrected"].get(metric_name)
        if cap is None or base is None:
            continue
        if isinstance(cap, float) and np.isnan(cap):
            continue
        if isinstance(base, float) and np.isnan(base):
            continue
        model_vals.append(float(cap))
        base_vals.append(float(base))
    return model_vals, base_vals


def build_summary(aggregated: Dict) -> Dict:
    cap = aggregated.get("capopm", {})
    worst_regret_brier = cap.get("regret_brier", float("nan"))
    worst_regret_log_bad = cap.get("regret_log_bad", float("nan"))
    pass_regret = (worst_regret_brier is not None and worst_regret_brier <= 0.0) and (
        worst_regret_log_bad is not None and worst_regret_log_bad <= 0.0
    )
    return {
        "status": {
            "pass": bool(pass_regret),
            "criteria": {
                "mean_regret_brier_leq_0": True,
                "mean_regret_log_bad_leq_0": True,
            },
            "metrics": {
                "mean_regret_brier": worst_regret_brier,
                "mean_regret_log_bad": worst_regret_log_bad,
            },
        }
    }


def build_metadata(config: Dict) -> Dict:
    meta = {
        "scenario_name": config.get("scenario_name"),
        "experiment_id": config.get("experiment_id"),
        "tier": config.get("tier"),
        "seed": int(config.get("seed", 0)),
    }
    meta.update(config.get("sweep_params", {}))
    return meta


def scenario_id(struct_shift: float, ml_bias: float, seed: int) -> str:
    return f"B3_misspec_regret__struct{int(round(struct_shift*100))}__ml{int(round(ml_bias*100))}__seed{seed}"


def raw_parimutuel_prob(trade_tape, pool_path) -> float:
    if pool_path:
        return float(pool_path[-1][3])
    if trade_tape:
        return float(trade_tape[-1].implied_yes_after)
    return 0.5


def build_base_config() -> Dict:
    cfg = {
        "p_true_dist": {"type": "fixed", "value": 0.55},
        "models": ["capopm", "structural_only", "ml_only", "uncorrected", "raw_parimutuel"],
        "traders": {
            "n_traders": 60,
            "proportions": {"informed": 0.5, "adversarial": 0.0, "noise": 0.5},
            "params": {
                "informed": {"signal_quality": 0.7, "noise_yes_prob": 0.5, "herding_intensity": 0.0},
                "adversarial": {"signal_quality": 0.7, "noise_yes_prob": 0.5, "herding_intensity": 0.0},
                "noise": {"signal_quality": 0.5, "noise_yes_prob": 0.5, "herding_intensity": 0.0},
            },
        },
        "market": {
            "n_steps": 20,
            "arrivals_per_step": 4,
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
        "ml_cfg": {"base_prob": 0.55, "bias": -0.02, "noise_std": 0.01, "calibration": 1.0, "r_ml": 0.8},
        "prior_cfg": {"n_str": 10.0, "n_ml_eff": 4.0, "n_ml_scale": 1.0},
        "stage1_cfg": {"enabled": True, "w_min": 0.1, "w_max": 1.25, "longshot_ref_p": 0.5, "longshot_gamma": 0.8, "herding_lambda": 0.0, "herding_window": 50},
        "stage2_cfg": {"enabled": False},
        "ece_bins": 10,
        "calibration_binning": "equal_width",
        "ece_min_nonempty_bins": 5,
        "coverage_include_outcome": False,
    }
    return copy.deepcopy(cfg)
