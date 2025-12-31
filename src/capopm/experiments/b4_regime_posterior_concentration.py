"""
Tier B Experiment B4: REGIME_POSTERIOR_CONCENTRATION.

Evaluates mixture concentration under varying evidence strength settings.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..likelihood import beta_binomial_update, counts_from_trade_tape, posterior_moments
from ..market_simulator import MarketConfig, simulate_market
from ..metrics.calibration import calibration_ece, interval_coverage_outcome, interval_coverage_ptrue
from ..metrics.distributional import mad_posterior_median, posterior_mean_bias, posterior_variance_ratio, wasserstein_distance_beta
from ..metrics.scoring import brier, log_score, mae_prob
from ..posterior import capopm_pipeline
from ..pricing import posterior_prices
from ..trader_model import build_traders
from ..stats.tests import bootstrap_ci, bonferroni_correction, holm_correction, paired_t_test, wilcoxon_signed_rank
from .regime_metrics import regime_entropy, regime_max_weight
from .runner import (
    REPORTING_VERSION,
    aggregate_metrics,
    build_trader_params,
    draw_p_true,
    format_p_value,
    write_scenario_outputs,
)

EXPERIMENT_ID = "B4.REGIME_POSTERIOR_CONCENTRATION"
TIER = "B"

DEFAULT_EVIDENCE_STRENGTH = [0.5, 1.0, 2.0]
DEFAULT_BASE_SEED = 202720
DEFAULT_N_RUNS = 60


def run_b4_regime_posterior_concentration(
    evidence_strength_grid: Iterable[float] = DEFAULT_EVIDENCE_STRENGTH,
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
) -> List[Dict]:
    results = []
    for idx, strength in enumerate(evidence_strength_grid):
        seed = base_seed + idx
        scenario_name = scenario_id(strength, seed)
        cfg = build_base_config()
        cfg["seed"] = seed
        cfg["n_runs"] = n_runs
        cfg["scenario_name"] = scenario_name
        cfg["experiment_id"] = EXPERIMENT_ID
        cfg["tier"] = TIER
        cfg["sweep_params"] = {
            "evidence_strength": float(strength),
        }
        apply_evidence_strength(cfg, strength)
        run_b4_scenario(cfg)
        results.append({"scenario_name": scenario_name, "seed": seed, "sweep_params": cfg["sweep_params"]})
    return results


def run_b4_scenario(config: Dict) -> Dict:
    cfg = copy.deepcopy(config)
    seed = int(cfg.get("seed", 123))
    n_runs = int(cfg.get("n_runs", 5))
    calibration_binning = cfg.get("calibration_binning", cfg.get("ece_binning", "equal_width"))
    min_nonempty_bins = int(cfg.get("ece_min_nonempty_bins", 5))
    n_bins = int(cfg.get("ece_bins", 10))
    rng_master = np.random.default_rng(seed)

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
                "regime_entropy": out.get("regime_entropy", float("nan")),
                "regime_max_weight": out.get("regime_max_weight", float("nan")),
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

    tests = run_b4_tests(per_run_metrics)

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
        "status": summary.get("status"),
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
            p_hat = out["mixture_mean"] if out.get("mixture_enabled") else out["pi_yes"]
            entropy = float("nan")
            max_weight = float("nan")
            if out.get("mixture_weights"):
                entropy = regime_entropy(out["mixture_weights"])
                max_weight = regime_max_weight(out["mixture_weights"])
            model_outputs[model] = {
                "p_hat": p_hat,
                "alpha": out["alpha_post"],
                "beta": out["beta_post"],
                "regime_entropy": entropy,
                "regime_max_weight": max_weight,
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
                "regime_entropy": float("nan"),
                "regime_max_weight": float("nan"),
            }
        elif model == "uncorrected":
            model_outputs[model] = {
                "p_hat": base_out["pi_yes"],
                "alpha": base_out["alpha_post"],
                "beta": base_out["beta_post"],
                "regime_entropy": float("nan"),
                "regime_max_weight": float("nan"),
            }
        elif model == "raw_parimutuel":
            model_outputs[model] = {
                "p_hat": raw_parimutuel_prob(trade_tape, pool_path),
                "alpha": None,
                "beta": None,
                "regime_entropy": float("nan"),
                "regime_max_weight": float("nan"),
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
                "regime_entropy": float("nan"),
                "regime_max_weight": float("nan"),
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
                "regime_entropy": float("nan"),
                "regime_max_weight": float("nan"),
            }
        elif model == "beta_1_1":
            alpha0, beta0 = 1.0, 1.0
            alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
            p_hat, _ = posterior_prices(alpha, beta)
            model_outputs[model] = {"p_hat": p_hat, "alpha": alpha, "beta": beta, "regime_entropy": float("nan"), "regime_max_weight": float("nan")}
        elif model == "beta_0_5_0_5":
            alpha0, beta0 = 0.5, 0.5
            alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
            p_hat, _ = posterior_prices(alpha, beta)
            model_outputs[model] = {"p_hat": p_hat, "alpha": alpha, "beta": beta, "regime_entropy": float("nan"), "regime_max_weight": float("nan")}
        else:
            raise ValueError(f"Unknown model: {model}")

    return model_outputs


def run_b4_tests(per_run_metrics: List[Dict]) -> List[Dict]:
    metrics = [
        {"name": "brier", "better_if_negative": True},
        {"name": "log_score", "better_if_negative": True},
        {"name": "mae_prob", "better_if_negative": True},
        {"name": "regret_brier", "better_if_negative": True},
        {"name": "regret_log_bad", "better_if_negative": True},
        {"name": "regret_abs_error", "better_if_negative": True},
    ]
    rows = []
    comparisons = ["capopm", "capopm_stage1_only"]
    for metric in metrics:
        metric_name = metric["name"]
        better_if_negative = metric["better_if_negative"]
        for model in comparisons:
            model_vals, base_vals = paired_finite_values(per_run_metrics, metric_name, model)
            if len(model_vals) < 2:
                continue
            t_stat, p_t = paired_t_test(model_vals, base_vals)
            w_stat, p_w = wilcoxon_signed_rank(model_vals, base_vals)
            diff = np.asarray(model_vals) - np.asarray(base_vals)
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


def paired_finite_values(per_run_metrics: List[Dict], metric_name: str, model: str) -> Tuple[List[float], List[float]]:
    model_vals = []
    base_vals = []
    for run in per_run_metrics:
        cap = run["metrics"][model].get(metric_name)
        base = run["metrics"]["uncorrected"].get(metric_name)
        if cap is None or base is None:
            continue
        if isinstance(cap, float) and math.isnan(cap):
            continue
        if isinstance(base, float) and math.isnan(base):
            continue
        model_vals.append(float(cap))
        base_vals.append(float(base))
    return model_vals, base_vals


def build_summary(aggregated: Dict) -> Dict:
    cap = aggregated.get("capopm", {})
    entropy = cap.get("regime_entropy", float("nan"))
    max_w = cap.get("regime_max_weight", float("nan"))
    return {
        "status": {
            "pass": None,
            "criteria": {
                "regime_entropy_should_decrease_with_strength": True,
                "regime_max_weight_should_increase_with_strength": True,
            },
            "metrics": {
                "regime_entropy": entropy,
                "regime_max_weight": max_w,
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


def scenario_id(evidence_strength: float, seed: int) -> str:
    return f"B4_regime_concentration__evidence{int(round(evidence_strength*100))}__seed{seed}"


def raw_parimutuel_prob(trade_tape, pool_path) -> float:
    if pool_path:
        return float(pool_path[-1][3])
    if trade_tape:
        return float(trade_tape[-1].implied_yes_after)
    return 0.5


def apply_evidence_strength(config: Dict, strength: float) -> None:
    config["stage2_cfg"] = dict(config.get("stage2_cfg", {}))
    regimes = []
    for regime in config["stage2_cfg"].get("regimes", []):
        regime_copy = dict(regime)
        regime_copy["g_plus_scale"] = float(regime_copy.get("g_plus_scale", 0.0)) * float(strength)
        regime_copy["g_minus_scale"] = float(regime_copy.get("g_minus_scale", 0.0)) * float(strength)
        regimes.append(regime_copy)
    config["stage2_cfg"]["regimes"] = regimes


def build_base_config() -> Dict:
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
            "proportions": {"informed": 0.5, "adversarial": 0.1, "noise": 0.4},
            "params": {
                "informed": {"signal_quality": 0.7, "noise_yes_prob": 0.5, "herding_intensity": 0.0},
                "adversarial": {"signal_quality": 0.7, "noise_yes_prob": 0.5, "herding_intensity": 0.0},
                "noise": {"signal_quality": 0.5, "noise_yes_prob": 0.5, "herding_intensity": 0.0},
            },
        },
        "market": {
            "n_steps": 25,
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
        "stage2_cfg": {
            "enabled": True,
            "mode": "offsets_mixture",
            "delta_plus": 0.0,
            "delta_minus": 0.0,
            "regimes": [
                {"pi": 0.5, "g_plus_scale": 0.08, "g_minus_scale": 0.08},
                {"pi": 0.5, "g_plus_scale": -0.08, "g_minus_scale": -0.08},
            ],
        },
        "ece_bins": 10,
        "calibration_binning": "equal_width",
        "ece_min_nonempty_bins": 5,
        "coverage_include_outcome": False,
    }
    return copy.deepcopy(cfg)
