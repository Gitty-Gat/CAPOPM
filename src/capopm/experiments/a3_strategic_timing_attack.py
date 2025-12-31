"""
Tier A Experiment A3: STRATEGIC_TIMING_ATTACK.

Evaluates robustness to late adversarial timing and size scaling.
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

EXPERIMENT_ID = "A3.STRATEGIC_TIMING_ATTACK"
TIER = "A"

DEFAULT_ATTACK_STRENGTH = [0.0, 0.5, 1.0]
DEFAULT_ATTACK_WINDOW = [0.10, 0.20]
DEFAULT_SIZE_SCALE = [1, 3, 5]
DEFAULT_BASE_SEED = 202630
DEFAULT_N_RUNS = 80


def run_a3_strategic_timing_attack(
    attack_strength_grid: Iterable[float] = DEFAULT_ATTACK_STRENGTH,
    attack_window_grid: Iterable[float] = DEFAULT_ATTACK_WINDOW,
    adversarial_size_scale_grid: Iterable[int] = DEFAULT_SIZE_SCALE,
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
) -> List[Dict]:
    """Run A3 sweeps deterministically across attack settings."""

    sweep = []
    for strength in attack_strength_grid:
        for window in attack_window_grid:
            for scale in adversarial_size_scale_grid:
                sweep.append((float(strength), float(window), int(scale)))

    results = []
    for idx, (strength, window, scale) in enumerate(sweep):
        seed = base_seed + idx
        scenario_name = scenario_id(strength, window, scale, seed)
        cfg = build_base_config()
        cfg["seed"] = seed
        cfg["n_runs"] = n_runs
        cfg["scenario_name"] = scenario_name
        cfg["experiment_id"] = EXPERIMENT_ID
        cfg["tier"] = TIER
        cfg["sweep_params"] = {
            "attack_strength": strength,
            "attack_window": window,
            "adversarial_size_scale": scale,
        }

        run_a3_scenario(cfg)
        results.append(
            {
                "scenario_name": scenario_name,
                "seed": seed,
                "sweep_params": cfg["sweep_params"],
            }
        )
    return results


def run_a3_scenario(config: Dict) -> Dict:
    """Run a single A3 scenario and write Phase 7 artifacts."""

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

        attack_strength = float(config.get("sweep_params", {}).get("attack_strength", 0.0))
        attack_window = float(config.get("sweep_params", {}).get("attack_window", 0.1))
        adv_scale = float(config.get("sweep_params", {}).get("adversarial_size_scale", 1.0))

        trade_tape, pool_path = apply_strategic_timing_attack(
            trade_tape=trade_tape,
            market_cfg=market_cfg,
            attack_strength=attack_strength,
            attack_window=attack_window,
            adversarial_size_scale=adv_scale,
            rng=rng,
        )

        y_raw, n_raw = counts_from_trade_tape(trade_tape)

        # Uncorrected posterior for variance normalization and regret reference.
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
                "regret_log": None,
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

        # Regret vs uncorrected baseline.
        uncorrected_log = metrics["uncorrected"]["log_score"]
        for model in metrics:
            metrics[model]["regret_log"] = metrics[model]["log_score"] - uncorrected_log

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

    tests = run_a3_tests(per_run_metrics, model_names)

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
        "note": "Regret defined as log_score(model) - log_score(uncorrected).",
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


def apply_strategic_timing_attack(
    trade_tape: List[Trade],
    market_cfg: MarketConfig,
    attack_strength: float,
    attack_window: float,
    adversarial_size_scale: float,
    rng: np.random.Generator,
) -> Tuple[List[Trade], List[Tuple[int, float, float, float, float]]]:
    """Shift a fraction of adversarial volume into the late window."""

    if attack_strength < 0.0 or attack_strength > 1.0:
        raise ValueError("attack_strength must be in [0,1]")
    if attack_window <= 0.0 or attack_window > 1.0:
        raise ValueError("attack_window must be in (0,1]")
    if adversarial_size_scale <= 0.0:
        raise ValueError("adversarial_size_scale must be positive")

    trades = []
    for idx, trade in enumerate(trade_tape):
        size = float(trade.size)
        if trade.trader_type == "adversarial":
            size *= adversarial_size_scale
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
        # Concentrate shifted trades at the final timestep to maximize late-window impact.
        final_step = max(0, market_cfg.n_steps - 1)
        for idx in selected:
            trades[idx]["t"] = final_step

    # Rebuild pool path using updated timing and sizes.
    trades.sort(key=lambda tr: (tr["t"], tr["orig_idx"]))
    return rebuild_trade_tape(trades, market_cfg)


def rebuild_trade_tape(
    trades: List[Dict],
    market_cfg: MarketConfig,
) -> Tuple[List[Trade], List[Tuple[int, float, float, float, float]]]:
    """Recompute trade tape and pool path after timing adjustments."""

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


def run_a3_tests(per_run_metrics: List[Dict], model_names: List[str]) -> List[Dict]:
    """Paired tests comparing CAPOPM vs baselines for A3 metrics."""

    metrics = [
        {"name": "brier", "better_if_negative": True},
        {"name": "log_score", "better_if_negative": False},
        {"name": "mae_prob", "better_if_negative": True},
        {"name": "regret_log", "better_if_negative": False},
        {"name": "regime_entropy", "better_if_negative": True},
        {"name": "regime_max_weight", "better_if_negative": False},
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
            if len(cap_vals) < 2:
                continue
            t_stat, p_t = paired_t_test(cap_vals, base_vals)
            w_stat, p_w = wilcoxon_signed_rank(cap_vals, base_vals)
            diff = np.asarray(cap_vals) - np.asarray(base_vals)
            ci_lo, ci_hi = bootstrap_ci(diff)
            diff_mean = float(np.nanmean(diff)) if diff.size > 0 else float("nan")
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

        if not model_order:
            continue
        holm = holm_correction(p_values)
        bonf = bonferroni_correction(p_values)
        for model, p_h, p_b in zip(model_order, holm, bonf):
            metric_results[model]["paired_t"]["p_holm"] = p_h
            metric_results[model]["paired_t"]["p_bonferroni"] = p_b
            metric_results[model]["paired_t"]["p_holm_str"] = format_p_value(p_h)
            metric_results[model]["paired_t"]["p_bonferroni_str"] = format_p_value(p_b)

        for model in model_order:
            res = metric_results[model]
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


def scenario_id(strength: float, window: float, scale: int, seed: int) -> str:
    """Stable scenario name encoding attack settings."""

    strength_tag = int(round(100 * strength))
    window_tag = int(round(100 * window))
    return f"A3_strategic_timing__attack{strength_tag}__seed{seed}__window{window_tag}__scale{scale}"


def build_base_config() -> Dict:
    """Base configuration for A3 with Stage 1 + Stage 2 mixture enabled."""

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
