"""
Tier A Experiment A2: TIME_TO_CONVERGE.

Quantifies convergence speed vs liquidity and pool seeding using time-path
metrics derived from transaction-level simulation outputs.
"""

from __future__ import annotations

import copy
import itertools
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..corrections.stage1_behavioral import apply_behavioral_weights
from ..corrections.stage2_structural import (
    apply_linear_offsets,
    summarize_stage1_stats,
    mixture_posterior_params,
)
from ..likelihood import beta_binomial_update, counts_from_trade_tape, posterior_moments
from ..market_simulator import MarketConfig, simulate_market
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
    aggregate_metrics,
    build_trader_params,
    draw_p_true,
    format_p_value,
    write_scenario_outputs,
    REPORTING_VERSION,
)
from .timepath_metrics import compute_time_to_eps, compute_var_decay_slope, group_trades_by_step

EXPERIMENT_ID = "A2.TIME_TO_CONVERGE"
TIER = "A"

DEFAULT_ARRIVALS = [1, 2, 3, 5]
DEFAULT_POOLS = [0.5, 1.0, 5.0]
DEFAULT_STEPS = [25, 50]
DEFAULT_BASE_SEED = 202620
DEFAULT_N_RUNS = 100


def run_a2_time_to_converge(
    arrivals_grid: Iterable[int] = DEFAULT_ARRIVALS,
    pool_grid: Iterable[float] = DEFAULT_POOLS,
    steps_grid: Iterable[int] = DEFAULT_STEPS,
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
    stage1_cfg: Dict | None = None,
    stage2_cfg: Dict | None = None,
) -> List[Dict]:
    """Run A2 sweeps deterministically across liquidity and seeding settings."""

    sweep = list(itertools.product(list(arrivals_grid), list(pool_grid), list(steps_grid)))
    results = []
    for idx, (arrivals, pool, steps) in enumerate(sweep):
        seed = base_seed + idx
        scenario_name = scenario_id(arrivals, pool, steps, seed)
        cfg = build_base_config()
        cfg["seed"] = seed
        cfg["n_runs"] = n_runs
        cfg["scenario_name"] = scenario_name
        cfg["experiment_id"] = EXPERIMENT_ID
        cfg["tier"] = TIER
        cfg["sweep_params"] = {
            "arrivals_per_step": arrivals,
            "initial_pool": pool,
            "n_steps": steps,
        }
        cfg["market"]["arrivals_per_step"] = int(arrivals)
        cfg["market"]["initial_yes_pool"] = float(pool)
        cfg["market"]["initial_no_pool"] = float(pool)
        cfg["market"]["n_steps"] = int(steps)

        if stage1_cfg is not None:
            cfg["stage1_cfg"] = copy.deepcopy(stage1_cfg)
        if stage2_cfg is not None:
            cfg["stage2_cfg"] = copy.deepcopy(stage2_cfg)

        run_a2_scenario(cfg)
        results.append(
            {
                "scenario_name": scenario_name,
                "seed": seed,
                "sweep_params": cfg["sweep_params"],
            }
        )
    return results


def run_a2_scenario(config: Dict) -> Dict:
    """Run a single A2 scenario and write Phase 7 artifacts."""

    seed = int(config.get("seed", 123))
    n_runs = int(config.get("n_runs", 5))
    calibration_binning = config.get("calibration_binning", config.get("ece_binning", "equal_width"))
    min_nonempty_bins = int(config.get("ece_min_nonempty_bins", 5))
    n_bins = int(config.get("ece_bins", 10))
    rng_master = np.random.default_rng(seed)

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

    per_run_metrics = []
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

        # Uncorrected posterior for variance normalization.
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

        model_outputs = build_model_outputs(rng, trade_tape, pool_path, config, base_out, y_raw, n_raw)

        trades_by_step = group_trades_by_step(trade_tape, market_cfg.n_steps)
        counts_by_step = cumulative_counts_by_step(trades_by_step)
        parimutuel_path = [float(p[3]) for p in pool_path]

        metrics = {}
        coverage_include_outcome = bool(
            config.get("coverage_include_outcome", config.get("include_outcome_coverage", False))
        )
        for model, out in model_outputs.items():
            p_hat = out["p_hat"]
            alpha = out["alpha"]
            beta = out["beta"]
            alpha0 = out.get("alpha0")
            beta0 = out.get("beta0")

            p_hat_path, var_path = build_time_paths(
                model=model,
                parimutuel_path=parimutuel_path,
                trades_by_step=trades_by_step,
                counts_by_step=counts_by_step,
                alpha0=alpha0,
                beta0=beta0,
                stage1_cfg=out.get("stage1_cfg"),
                stage2_cfg=out.get("stage2_cfg"),
            )

            metrics[model] = {
                "brier": brier(p_true, p_hat),
                "log_score": log_score(p_hat, outcome),
                "mae_prob": mae_prob(p_true, p_hat),
                "abs_error_outcome": abs(p_hat - outcome),
                "calibration_ece": None,
                "calibration_diagnostics": {},
                "posterior_mean_bias": posterior_mean_bias(p_hat, p_true),
                "p_hat": p_hat,
                "time_to_eps_0.05": compute_time_to_eps(p_hat_path, p_true, 0.05),
                "time_to_eps_0.10": compute_time_to_eps(p_hat_path, p_true, 0.10),
                "var_decay_slope": compute_var_decay_slope(var_path),
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

            p_hat_lists[model].append(p_hat)
            outcome_lists[model].append(int(outcome))

        per_run_metrics.append(
            {"run": run_idx, "p_true": p_true, "outcome": outcome, "metrics": metrics}
        )

    aggregated = aggregate_metrics(per_run_metrics, model_names)
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

    tests = run_timepath_tests(per_run_metrics, model_names)

    warnings = []
    for m in model_names:
        diag = aggregated[m].get("calibration_diagnostics", {})
        if diag and diag.get("degenerate_binning"):
            warnings.append(
                f"Calibration diagnostics: degenerate binning detected for model '{m}'."
            )

    scenario_name = config.get("scenario_name")
    results = {
        "per_run_metrics": per_run_metrics,
        "aggregated_metrics": aggregated,
        "tests": tests,
        "warnings": warnings,
        "note": "Partial validation under controlled synthetic assumptions.",
        "metadata": build_metadata(config),
        "reporting_version": REPORTING_VERSION,
        "status": {"pass": None, "notes": "", "criteria": {}},
        "sweep_params": config.get("sweep_params", {}),
    }

    write_scenario_outputs(
        scenario_name=scenario_name,
        results=results,
        p_hat_lists=p_hat_lists,
        outcome_lists=outcome_lists,
        n_bins=n_bins,
        binning=calibration_binning,
        min_nonempty_bins=min_nonempty_bins,
        experiment_id=config.get("experiment_id"),
        tier=config.get("tier"),
        seed=seed,
        extra_metadata=config.get("sweep_params", {}),
        config_snapshot=config,
    )
    return results


def build_time_paths(
    model: str,
    parimutuel_path: List[float],
    trades_by_step: List[List],
    counts_by_step: List[Tuple[float, float]],
    alpha0: float | None,
    beta0: float | None,
    stage1_cfg: Dict | None,
    stage2_cfg: Dict | None,
) -> Tuple[List[float], List[float]]:
    """Compute p_hat and variance paths for a model."""

    if model == "raw_parimutuel":
        return list(parimutuel_path), [float("nan")] * len(parimutuel_path)
    if alpha0 is None or beta0 is None:
        return [float("nan")] * len(parimutuel_path), [float("nan")] * len(parimutuel_path)

    stage1_enabled = stage1_cfg is not None and bool(stage1_cfg.get("enabled", False))
    stage2_enabled = stage2_cfg is not None and bool(stage2_cfg.get("enabled", False))
    stage2_mode = (stage2_cfg or {}).get("mode", "offsets")

    # Fast path: no Stage 1/2 corrections.
    if not stage1_enabled and not stage2_enabled:
        return posterior_paths_from_counts(alpha0, beta0, counts_by_step)

    # Slow path: apply corrections on prefix trade tape per step.
    p_hat_path = []
    var_path = []
    prefix: List = []
    for step_trades in trades_by_step:
        prefix.extend(step_trades)
        y_used, n_used = counts_from_trade_tape(prefix)
        if stage1_enabled:
            y_used, n_used, _ = apply_behavioral_weights(prefix, stage1_cfg)

        if stage2_enabled and ("offsets" in stage2_mode):
            y_used, n_used = apply_linear_offsets(y_used, n_used, stage2_cfg)

        alpha_post, beta_post = beta_binomial_update(alpha0, beta0, y_used, n_used)
        mean, var = posterior_moments(alpha_post, beta_post)

        if stage2_enabled and ("mixture" in stage2_mode):
            s = summarize_stage1_stats(prefix, y_used, n_used, stage2_cfg)
            mix = mixture_posterior_params(alpha0, beta0, s, stage2_cfg.get("regimes", []), y_used, n_used)
            mean = mix["mixture_mean"]
            var = mix["mixture_var"]

        p_hat_path.append(mean)
        var_path.append(var)

    return p_hat_path, var_path


def posterior_paths_from_counts(
    alpha0: float,
    beta0: float,
    counts_by_step: List[Tuple[float, float]],
) -> Tuple[List[float], List[float]]:
    """Posterior mean/variance paths using cumulative counts."""

    p_hat_path = []
    var_path = []
    for y, n in counts_by_step:
        alpha_post, beta_post = beta_binomial_update(alpha0, beta0, y, n)
        mean, var = posterior_moments(alpha_post, beta_post)
        p_hat_path.append(mean)
        var_path.append(var)
    return p_hat_path, var_path


def cumulative_counts_by_step(trades_by_step: List[List]) -> List[Tuple[float, float]]:
    """Cumulative YES/total counts by step."""

    counts = []
    y = 0.0
    n = 0.0
    for step_trades in trades_by_step:
        for trade in step_trades:
            size = float(getattr(trade, "size"))
            n += size
            if getattr(trade, "side") == "YES":
                y += size
        counts.append((y, n))
    return counts


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
            alpha0 = out["alpha0"]
            beta0 = out["beta0"]
            model_outputs[model] = {
                "p_hat": p_hat,
                "alpha": alpha,
                "beta": beta,
                "alpha0": alpha0,
                "beta0": beta0,
                "stage1_cfg": config.get("stage1_cfg", {"enabled": False}),
                "stage2_cfg": config.get("stage2_cfg", {"enabled": False}),
            }
        elif model == "raw_parimutuel":
            model_outputs[model] = {
                "p_hat": raw_parimutuel_prob(trade_tape, pool_path),
                "alpha": None,
                "beta": None,
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
                "alpha0": out["alpha0"],
                "beta0": out["beta0"],
                "stage1_cfg": {"enabled": False},
                "stage2_cfg": {"enabled": False},
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
                "alpha0": out["alpha0"],
                "beta0": out["beta0"],
                "stage1_cfg": {"enabled": False},
                "stage2_cfg": {"enabled": False},
            }
        elif model == "uncorrected":
            model_outputs[model] = {
                "p_hat": base_out["pi_yes"],
                "alpha": base_out["alpha_post"],
                "beta": base_out["beta_post"],
                "alpha0": base_out["alpha0"],
                "beta0": base_out["beta0"],
                "stage1_cfg": {"enabled": False},
                "stage2_cfg": {"enabled": False},
            }
        elif model == "beta_1_1":
            alpha0, beta0 = 1.0, 1.0
            alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
            p_hat, _ = posterior_prices(alpha, beta)
            model_outputs[model] = {
                "p_hat": p_hat,
                "alpha": alpha,
                "beta": beta,
                "alpha0": alpha0,
                "beta0": beta0,
                "stage1_cfg": {"enabled": False},
                "stage2_cfg": {"enabled": False},
            }
        elif model == "beta_0_5_0_5":
            alpha0, beta0 = 0.5, 0.5
            alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
            p_hat, _ = posterior_prices(alpha, beta)
            model_outputs[model] = {
                "p_hat": p_hat,
                "alpha": alpha,
                "beta": beta,
                "alpha0": alpha0,
                "beta0": beta0,
                "stage1_cfg": {"enabled": False},
                "stage2_cfg": {"enabled": False},
            }
        else:
            raise ValueError(f"Unknown model: {model}")

    return model_outputs


def run_timepath_tests(per_run_metrics: List[Dict], model_names: List[str]) -> List[Dict]:
    """Paired tests for time-to-eps, variance decay, and final scores."""

    metrics = [
        {"name": "time_to_eps_0.05", "better_if_negative": True},
        {"name": "time_to_eps_0.10", "better_if_negative": True},
        {"name": "var_decay_slope", "better_if_negative": True},
        {"name": "brier", "better_if_negative": True},
        {"name": "log_score", "better_if_negative": False},
    ]
    rows = []
    for metric in metrics:
        metric_name = metric["name"]
        better_if_negative = metric["better_if_negative"]
        p_values = []
        model_order = []
        metric_results = {}
        for model in model_names:
            if model == "capopm":
                continue
            cap_vals, base_vals = paired_finite_values(per_run_metrics, metric_name, model)
            t_stat, p_t = paired_t_test(cap_vals, base_vals)
            w_stat, p_w = wilcoxon_signed_rank(cap_vals, base_vals)
            diff = np.asarray(cap_vals) - np.asarray(base_vals)
            ci_lo, ci_hi = bootstrap_ci(diff)
            diff_mean = float(np.nanmean(diff)) if diff.size > 0 else np.nan
            metric_results[model] = {
                "paired_t": {"stat": t_stat, "p_value": p_t, "p_value_str": format_p_value(p_t)},
                "wilcoxon": {"stat": w_stat, "p_value": p_w, "p_value_str": format_p_value(p_w)},
                "bootstrap_ci": {"low": ci_lo, "high": ci_hi},
                "metric": metric_name,
                "diff_mean": diff_mean,
                "better_if_negative": better_if_negative,
            }
            p_values.append(p_t)
            model_order.append(model)

        holm = holm_correction(p_values)
        bonf = bonferroni_correction(p_values)
        for model, p_h, p_b in zip(model_order, holm, bonf):
            metric_results[model]["paired_t"]["p_holm"] = p_h
            metric_results[model]["paired_t"]["p_bonferroni"] = p_b
            metric_results[model]["paired_t"]["p_holm_str"] = format_p_value(p_h)
            metric_results[model]["paired_t"]["p_bonferroni_str"] = format_p_value(p_b)

        for model in model_order:
            res = metric_results[model]
            # paired t-test row
            rows.append(
                {
                    "model": model,
                    "metric": metric_name,
                    "test": "paired_t",
                    "stat": res["paired_t"]["stat"],
                    "p_value": max(res["paired_t"]["p_value"], 1e-300),
                    "p_value_str": res["paired_t"]["p_value_str"],
                    "p_holm": max(res["paired_t"].get("p_holm", 1.0), 1e-300),
                    "p_holm_str": res["paired_t"].get("p_holm_str"),
                    "p_bonferroni": max(res["paired_t"].get("p_bonferroni", 1.0), 1e-300),
                    "p_bonferroni_str": res["paired_t"].get("p_bonferroni_str"),
                    "diff_mean": res["diff_mean"],
                    "better_if_negative": better_if_negative,
                    "ci_low": None,
                    "ci_high": None,
                }
            )
            # wilcoxon row
            rows.append(
                {
                    "model": model,
                    "metric": metric_name,
                    "test": "wilcoxon",
                    "stat": res["wilcoxon"]["stat"],
                    "p_value": max(res["wilcoxon"]["p_value"], 1e-300),
                    "p_value_str": res["wilcoxon"]["p_value_str"],
                    "p_holm": None,
                    "p_holm_str": None,
                    "p_bonferroni": None,
                    "p_bonferroni_str": None,
                    "diff_mean": res["diff_mean"],
                    "better_if_negative": better_if_negative,
                    "ci_low": None,
                    "ci_high": None,
                }
            )
            # bootstrap CI row
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
                    "diff_mean": res["diff_mean"],
                    "better_if_negative": better_if_negative,
                    "ci_low": res["bootstrap_ci"]["low"],
                    "ci_high": res["bootstrap_ci"]["high"],
                }
            )

    return rows


def paired_finite_values(
    per_run_metrics: List[Dict],
    metric_name: str,
    baseline: str,
) -> Tuple[List[float], List[float]]:
    """Collect paired finite values for CAPOPM and baseline metrics."""

    cap_vals = []
    base_vals = []
    for run in per_run_metrics:
        cap = run["metrics"]["capopm"].get(metric_name)
        base = run["metrics"][baseline].get(metric_name)
        if cap is None or base is None:
            continue
        if isinstance(cap, float) and np.isnan(cap):
            continue
        if isinstance(base, float) and np.isnan(base):
            continue
        cap_vals.append(float(cap))
        base_vals.append(float(base))
    return cap_vals, base_vals


def build_metadata(config: Dict) -> Dict:
    """Build summary metadata with sweep params."""

    meta = {
        "scenario_name": config.get("scenario_name"),
        "experiment_id": config.get("experiment_id"),
        "tier": config.get("tier"),
        "seed": int(config.get("seed", 0)),
    }
    meta.update(config.get("sweep_params", {}))
    return meta


def raw_parimutuel_prob(trade_tape, pool_path) -> float:
    """Raw parimutuel implied YES probability from the latest pool state."""

    if pool_path:
        return float(pool_path[-1][3])
    if trade_tape:
        return float(trade_tape[-1].implied_yes_after)
    return 0.5


def scenario_id(arrivals: int, pool: float, steps: int, seed: int) -> str:
    """Stable scenario name encoding sweep settings."""

    return f"A2_time_to_converge__arr{arrivals}__seed{seed}__pool{pool}__steps{steps}"


def build_base_config() -> Dict:
    """Base configuration for A2 with Stage 1/2 disabled."""

    cfg = {
        "p_true_dist": {"type": "beta", "a": 2.0, "b": 2.0},
        "models": [
            "capopm",
            "raw_parimutuel",
            "structural_only",
            "ml_only",
            "uncorrected",
            "beta_1_1",
            "beta_0_5_0_5",
        ],
        "traders": {
            "n_traders": 60,
            "proportions": {
                "informed": 0.5,
                "adversarial": 0.1,
                "noise": 0.4,
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
            "n_steps": 25,
            "arrivals_per_step": 2,
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
        "stage1_cfg": {"enabled": False},
        "stage2_cfg": {"enabled": False},
        "ece_bins": 10,
        "calibration_binning": "equal_width",
        "ece_min_nonempty_bins": 5,
        "coverage_include_outcome": False,
    }

    return copy.deepcopy(cfg)
