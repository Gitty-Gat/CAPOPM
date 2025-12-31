
"""
Tier B Experiment B1: CORRECTION_NO_REGRET.

Evaluates whether corrections avoid regret relative to uncorrected baselines
under a stress-suite of behavioral and market conditions.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..likelihood import beta_binomial_update, counts_from_trade_tape, posterior_moments
from ..market_simulator import MarketConfig, Trade, implied_probs, parimutuel_odds, simulate_market
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
from .regime_metrics import regime_entropy, regime_max_weight
from .runner import (
    REPORTING_VERSION,
    aggregate_metrics,
    build_trader_params,
    draw_p_true,
    format_p_value,
    write_scenario_outputs,
)

EXPERIMENT_ID = "B1.CORRECTION_NO_REGRET"
TIER = "B"

DEFAULT_LONGSHOT_BIAS = [0, 1]
DEFAULT_HERDING = [0, 1]
DEFAULT_TIMING_ATTACK = [0.0, 1.0]
DEFAULT_LIQUIDITY_LEVELS = ["low", "high"]
DEFAULT_BASE_SEED = 202640
DEFAULT_N_RUNS = 80

LONGSHOT_THRESHOLD = 0.35
LONGSHOT_SIZE_SCALE = 2.0
Herding_intensity = 0.5
ATTACK_WINDOW = 0.2

LIQUIDITY_LEVELS = {
    "low": {"arrivals_per_step": 1, "initial_pool": 0.5},
    "high": {"arrivals_per_step": 5, "initial_pool": 5.0},
}


def run_b1_correction_no_regret(
    longshot_bias_grid: Iterable[int] = DEFAULT_LONGSHOT_BIAS,
    herding_grid: Iterable[int] = DEFAULT_HERDING,
    timing_attack_grid: Iterable[float] = DEFAULT_TIMING_ATTACK,
    liquidity_level_grid: Iterable[str] = DEFAULT_LIQUIDITY_LEVELS,
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
) -> List[Dict]:
    """Run B1 sweeps deterministically across the stress suite."""

    sweep = []
    for longshot_bias in longshot_bias_grid:
        for herding_on in herding_grid:
            for timing_attack in timing_attack_grid:
                for liquidity_level in liquidity_level_grid:
                    sweep.append(
                        (int(longshot_bias), int(herding_on), float(timing_attack), str(liquidity_level))
                    )

    results = []
    for idx, (longshot_bias, herding_on, timing_attack, liquidity_level) in enumerate(sweep):
        seed = base_seed + idx
        scenario_name = scenario_id(longshot_bias, herding_on, timing_attack, liquidity_level, seed)
        cfg = build_base_config()
        cfg["seed"] = seed
        cfg["n_runs"] = n_runs
        cfg["scenario_name"] = scenario_name
        cfg["experiment_id"] = EXPERIMENT_ID
        cfg["tier"] = TIER
        cfg["sweep_params"] = {
            "longshot_bias": int(longshot_bias),
            "herding": int(herding_on),
            "timing_attack_strength": float(timing_attack),
            "liquidity_level": str(liquidity_level),
        }

        run_b1_scenario(cfg)
        results.append(
            {
                "scenario_name": scenario_name,
                "seed": seed,
                "sweep_params": cfg["sweep_params"],
            }
        )
    return results


def run_b1_scenario(config: Dict) -> Dict:
    """Run a single B1 scenario and write Phase 7 artifacts."""

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
            "structural_only",
            "ml_only",
        ],
    )

    per_run_metrics = []
    p_hat_lists = {m: [] for m in model_names}
    outcome_lists = {m: [] for m in model_names}

    apply_liquidity_level(cfg)
    apply_herding(cfg)

    for run_idx in range(n_runs):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

        p_true = draw_p_true(rng, cfg.get("p_true_dist", {"type": "fixed", "value": 0.5}))
        outcome = 1 if float(rng.random()) < p_true else 0

        traders = build_traders(
            n_traders=int(cfg["traders"]["n_traders"]),
            proportions=cfg["traders"]["proportions"],
            params_by_type=build_trader_params(cfg["traders"].get("params", {})),
        )

        market_cfg = MarketConfig(**cfg["market"])
        trade_tape, pool_path = simulate_market(rng, market_cfg, traders, p_true)

        longshot_bias = int(cfg.get("sweep_params", {}).get("longshot_bias", 0))
        timing_attack = float(cfg.get("sweep_params", {}).get("timing_attack_strength", 0.0))

        if longshot_bias:
            trade_tape, pool_path = apply_longshot_bias(trade_tape, market_cfg)
        if timing_attack > 0.0:
            trade_tape, pool_path = apply_strategic_timing_attack(
                trade_tape=trade_tape,
                market_cfg=market_cfg,
                attack_strength=timing_attack,
                attack_window=ATTACK_WINDOW,
                rng=rng,
            )

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

    tests = run_b1_tests(per_run_metrics)

    warnings = []
    for m in model_names:
        diag = aggregated[m].get("calibration_diagnostics", {})
        if diag and diag.get("degenerate_binning"):
            warnings.append(
                f"Calibration diagnostics: degenerate binning detected for model '{m}'."
            )

    summary = build_summary(aggregated, per_run_metrics)
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

def apply_longshot_bias(
    trade_tape: List[Trade],
    market_cfg: MarketConfig,
) -> Tuple[List[Trade], List[Tuple[int, float, float, float, float]]]:
    """Increase sizes on longshot-side trades and rebuild pool path."""

    trades = []
    for idx, trade in enumerate(trade_tape):
        size = float(trade.size)
        is_longshot = False
        if trade.side == "YES" and trade.implied_yes_before < LONGSHOT_THRESHOLD:
            is_longshot = True
        if trade.side == "NO" and trade.implied_no_before < LONGSHOT_THRESHOLD:
            is_longshot = True
        if is_longshot:
            size *= LONGSHOT_SIZE_SCALE
        trades.append(
            {
                "t": int(trade.t),
                "trader_id": int(trade.trader_id),
                "trader_type": trade.trader_type,
                "side": trade.side,
                "size": size,
                "orig_idx": idx,
            }
        )

    trades.sort(key=lambda tr: (tr["t"], tr["orig_idx"]))
    return rebuild_trade_tape(trades, market_cfg)


def apply_strategic_timing_attack(
    trade_tape: List[Trade],
    market_cfg: MarketConfig,
    attack_strength: float,
    attack_window: float,
    rng: np.random.Generator,
) -> Tuple[List[Trade], List[Tuple[int, float, float, float, float]]]:
    """Shift a fraction of adversarial volume into the late window."""

    if attack_strength < 0.0 or attack_strength > 1.0:
        raise ValueError("attack_strength must be in [0,1]")
    if attack_window <= 0.0 or attack_window > 1.0:
        raise ValueError("attack_window must be in (0,1]")

    trades = []
    for idx, trade in enumerate(trade_tape):
        trades.append(
            {
                "t": int(trade.t),
                "trader_id": int(trade.trader_id),
                "trader_type": trade.trader_type,
                "side": trade.side,
                "size": float(trade.size),
                "orig_idx": idx,
            }
        )

    adv_indices = [i for i, tr in enumerate(trades) if tr["trader_type"] == "adversarial"]
    total_adv = sum(trades[i]["size"] for i in adv_indices)
    target = attack_strength * total_adv

    if target > 0.0 and adv_indices:
        rng.shuffle(adv_indices)
        selected = []
        acc = 0.0
        for idx in adv_indices:
            selected.append(idx)
            acc += trades[idx]["size"]
            if acc >= target:
                break
        final_step = max(0, market_cfg.n_steps - 1)
        for idx in selected:
            trades[idx]["t"] = final_step

    trades.sort(key=lambda tr: (tr["t"], tr["orig_idx"]))
    return rebuild_trade_tape(trades, market_cfg)


def rebuild_trade_tape(
    trades: List[Dict],
    market_cfg: MarketConfig,
) -> Tuple[List[Trade], List[Tuple[int, float, float, float, float]]]:
    """Recompute trade tape and pool path after modifications."""

    yes_pool = market_cfg.initial_yes_pool
    no_pool = market_cfg.initial_no_pool
    trade_tape: List[Trade] = []
    pool_path: List[Tuple[int, float, float, float, float]] = []

    grouped = [[] for _ in range(market_cfg.n_steps)]
    for tr in trades:
        grouped[int(tr["t"])].append(tr)

    for t in range(market_cfg.n_steps):
        for tr in grouped[t]:
            yes_pool_before = yes_pool
            no_pool_before = no_pool
            implied_yes_before, implied_no_before = implied_probs(
                yes_pool_before, no_pool_before, market_cfg.fee_rate
            )
            odds_yes_before = parimutuel_odds(yes_pool_before, no_pool_before)

            if tr["side"] == "YES":
                yes_pool += tr["size"]
            else:
                no_pool += tr["size"]
            implied_yes_after, implied_no_after = implied_probs(
                yes_pool, no_pool, market_cfg.fee_rate
            )
            odds_yes_after = parimutuel_odds(yes_pool, no_pool)

            trade_tape.append(
                Trade(
                    t=t,
                    trader_id=tr["trader_id"],
                    trader_type=tr["trader_type"],
                    side=tr["side"],
                    size=tr["size"],
                    yes_pool_before=yes_pool_before,
                    no_pool_before=no_pool_before,
                    yes_pool_after=yes_pool,
                    no_pool_after=no_pool,
                    implied_yes_before=implied_yes_before,
                    implied_yes_after=implied_yes_after,
                    implied_no_before=implied_no_before,
                    implied_no_after=implied_no_after,
                    odds_yes_before=odds_yes_before,
                    odds_yes_after=odds_yes_after,
                )
            )

        implied_yes, implied_no = implied_probs(yes_pool, no_pool, market_cfg.fee_rate)
        pool_path.append((t, yes_pool, no_pool, implied_yes, implied_no))

    return trade_tape, pool_path


def build_model_outputs(
    rng: np.random.Generator,
    trade_tape: List[Trade],
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
            model_outputs[model] = {
                "p_hat": p_hat,
                "alpha": alpha,
                "beta": beta,
                "regime_entropy": float("nan"),
                "regime_max_weight": float("nan"),
            }
        elif model == "beta_0_5_0_5":
            alpha0, beta0 = 0.5, 0.5
            alpha, beta = beta_binomial_update(alpha0, beta0, y_raw, n_raw)
            p_hat, _ = posterior_prices(alpha, beta)
            model_outputs[model] = {
                "p_hat": p_hat,
                "alpha": alpha,
                "beta": beta,
                "regime_entropy": float("nan"),
                "regime_max_weight": float("nan"),
            }
        else:
            raise ValueError(f"Unknown model: {model}")

    return model_outputs

def run_b1_tests(per_run_metrics: List[Dict]) -> List[Dict]:
    """Paired tests for regret metrics against uncorrected baseline."""

    metrics = [
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


def paired_finite_values(
    per_run_metrics: List[Dict],
    metric_name: str,
    model: str,
) -> Tuple[List[float], List[float]]:
    """Collect paired values for model vs uncorrected (zero-regret baseline)."""

    model_vals = []
    base_vals = []
    for run in per_run_metrics:
        model_val = run["metrics"][model].get(metric_name)
        base_val = run["metrics"]["uncorrected"].get(metric_name)
        if model_val is None or base_val is None:
            continue
        if isinstance(model_val, float) and np.isnan(model_val):
            continue
        if isinstance(base_val, float) and np.isnan(base_val):
            continue
        model_vals.append(float(model_val))
        base_vals.append(float(base_val))
    return model_vals, base_vals


def build_summary(aggregated: Dict, per_run_metrics: List[Dict]) -> Dict:
    """Build B1 summary with pass/fail and regret diagnostics."""

    cap = aggregated.get("capopm", {})
    mean_regret_brier = cap.get("regret_brier", float("nan"))
    mean_regret_log_bad = cap.get("regret_log_bad", float("nan"))
    pass_regret = (mean_regret_brier is not None and mean_regret_brier <= 0.0) and (
        mean_regret_log_bad is not None and mean_regret_log_bad <= 0.0
    )

    brier_pos = 0
    log_pos = 0
    total = 0
    for run in per_run_metrics:
        total += 1
        if run["metrics"]["capopm"].get("regret_brier", 0.0) > 0.0:
            brier_pos += 1
        if run["metrics"]["capopm"].get("regret_log_bad", 0.0) > 0.0:
            log_pos += 1
    frac_brier_pos = brier_pos / total if total > 0 else float("nan")
    frac_log_pos = log_pos / total if total > 0 else float("nan")

    return {
        "status": {
            "pass": bool(pass_regret),
            "criteria": {
                "mean_regret_brier_leq_0": True,
                "mean_regret_log_bad_leq_0": True,
            },
            "metrics": {
                "mean_regret_brier": mean_regret_brier,
                "mean_regret_log_bad": mean_regret_log_bad,
                "frac_regret_brier_positive": frac_brier_pos,
                "frac_regret_log_bad_positive": frac_log_pos,
            },
        }
    }


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


def apply_liquidity_level(config: Dict) -> None:
    """Apply liquidity stressor to the market configuration."""

    level = str(config.get("sweep_params", {}).get("liquidity_level", "low"))
    if level not in LIQUIDITY_LEVELS:
        raise ValueError(f"Unknown liquidity_level: {level}")
    settings = LIQUIDITY_LEVELS[level]
    config["market"]["arrivals_per_step"] = settings["arrivals_per_step"]
    config["market"]["initial_yes_pool"] = settings["initial_pool"]
    config["market"]["initial_no_pool"] = settings["initial_pool"]


def apply_herding(config: Dict) -> None:
    """Apply herding stressor to trader params and market config."""

    herding_on = int(config.get("sweep_params", {}).get("herding", 0))
    config["market"]["herding_enabled"] = bool(herding_on)
    if herding_on:
        for params in config["traders"]["params"].values():
            params["herding_intensity"] = Herding_intensity
    else:
        for params in config["traders"]["params"].values():
            params["herding_intensity"] = 0.0


def scenario_id(
    longshot_bias: int,
    herding_on: int,
    timing_attack: float,
    liquidity_level: str,
    seed: int,
) -> str:
    """Stable scenario name encoding stress settings."""

    attack_tag = int(round(100 * timing_attack))
    return (
        f"B1_correction_no_regret__longshot{longshot_bias}__herd{herding_on}"
        f"__attack{attack_tag}__liq{liquidity_level}__seed{seed}"
    )


def build_base_config() -> Dict:
    """Base configuration for B1 with Stage 1 + Stage 2 mixture enabled."""

    cfg = {
        "p_true_dist": {"type": "beta", "a": 2.0, "b": 2.0},
        "models": [
            "capopm",
            "capopm_stage1_only",
            "uncorrected",
            "raw_parimutuel",
            "structural_only",
            "ml_only",
        ],
        "traders": {
            "n_traders": 60,
            "proportions": {
                "informed": 0.5,
                "adversarial": 0.2,
                "noise": 0.3,
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
            "n_steps": 30,
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
        "stage1_cfg": {
            "enabled": True,
            "w_min": 0.1,
            "w_max": 1.25,
            "longshot_ref_p": 0.5,
            "longshot_gamma": 0.8,
            "herding_lambda": 0.8,
            "herding_window": 50,
        },
        "stage2_cfg": {
            "enabled": True,
            "mode": "offsets_mixture",
            "delta_plus": 0.0,
            "delta_minus": 0.0,
            "regimes": [
                {"pi": 0.5, "g_plus_scale": 0.05, "g_minus_scale": 0.05},
                {"pi": 0.5, "g_plus_scale": -0.05, "g_minus_scale": -0.05},
            ],
        },
        "ece_bins": 10,
        "calibration_binning": "equal_width",
        "ece_min_nonempty_bins": 5,
        "coverage_include_outcome": False,
    }

    return copy.deepcopy(cfg)
