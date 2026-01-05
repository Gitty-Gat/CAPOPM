"""
Tier C3: NONSTATIONARY_PTRUE_DRIFT

Descriptive stress run with linearly drifting p_true over the horizon.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List

import numpy as np

from ..market_simulator import MarketConfig, Trade
from ..trader_model import build_traders
from ..likelihood import counts_from_trade_tape, beta_binomial_update, posterior_moments
from ..posterior import capopm_pipeline
from ..pricing import posterior_prices
from ..metrics.scoring import brier, log_score, mae_prob
from ..metrics.calibration import interval_coverage_ptrue, interval_coverage_outcome, calibration_ece
from ..metrics.distributional import posterior_variance_ratio, mad_posterior_median, wasserstein_distance_beta
from .runner import (
    REPORTING_VERSION,
    aggregate_metrics,
    build_trader_params,
    write_scenario_outputs,
)
from ..invariant_runtime import (
    InvariantContext,
    current_context,
    reset_invariant_context,
    set_invariant_context,
    stable_config_hash,
)

EXPERIMENT_ID = "C3.NONSTATIONARY_PTRUE_DRIFT"
TIER = "C"
DEFAULT_BASE_SEED = 304000
DEFAULT_N_RUNS = 30


def build_base_config() -> Dict:
    cfg = {
        "models": [
            "capopm",
            "raw_parimutuel",
            "structural_only",
            "ml_only",
            "uncorrected",
            "beta_1_1",
            "beta_0_5_0_5",
        ],
        "p_true_start": 0.2,
        "p_true_end": 0.8,
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
            "n_steps": 40,
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


def simulate_market_dynamic(
    rng: np.random.Generator, cfg: MarketConfig, traders: List, p_true_path: List[float]
) -> (List[Trade], List[tuple]):
    """Simulate with time-varying p_true path."""

    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if cfg.arrivals_per_step <= 0:
        raise ValueError("arrivals_per_step must be positive")
    if cfg.initial_yes_pool <= 0 or cfg.initial_no_pool <= 0:
        raise ValueError("initial pools must be positive")
    if not traders:
        raise ValueError("traders list must be non-empty")

    yes_pool = cfg.initial_yes_pool
    no_pool = cfg.initial_no_pool
    trade_tape: List[Trade] = []
    pool_path: List[tuple] = []
    history: List[str] = []

    for t in range(cfg.n_steps):
        p_t = p_true_path[min(t, len(p_true_path) - 1)]
        realized_state = 1 if float(rng.random()) < p_t else 0
        for _ in range(cfg.arrivals_per_step):
            trader = traders[int(rng.integers(0, len(traders)))]
            p_hist = float(history.count("YES") / len(history)) if history else 0.5
            side = trader.decide(
                rng=rng,
                p_true=p_t,
                p_hist=p_hist,
                signal_model=cfg.signal_model,
                realized_state=realized_state,
                herding_enabled=cfg.herding_enabled,
            )

            trade_size = cfg.size_dist_params.get("size", 1.0)
            yes_pool_before = yes_pool
            no_pool_before = no_pool
            implied_yes_before = yes_pool_before / (yes_pool_before + no_pool_before)
            implied_no_before = no_pool_before / (yes_pool_before + no_pool_before)
            odds_yes_before = (yes_pool_before / no_pool_before) if no_pool_before > 0 else 1.0

            if side == "YES":
                yes_pool += trade_size
            else:
                no_pool += trade_size

            implied_yes_after = yes_pool / (yes_pool + no_pool)
            implied_no_after = no_pool / (yes_pool + no_pool)
            odds_yes_after = (yes_pool / no_pool) if no_pool > 0 else 1.0

            trade_tape.append(
                Trade(
                    t=t,
                    trader_id=trader.trader_id,
                    trader_type=trader.trader_type,
                    side=side,
                    size=trade_size,
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
            history.append(side)

        implied_yes = yes_pool / (yes_pool + no_pool)
        implied_no = no_pool / (yes_pool + no_pool)
        pool_path.append((t, yes_pool, no_pool, implied_yes, implied_no))

    return trade_tape, pool_path


def run_c3_nonstationary_ptrue_drift(
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
    p_start: float = 0.2,
    p_end: float = 0.8,
) -> List[Dict]:
    scenario_name = f"C3_nonstationary_ptrue_drift__seed{base_seed}"
    cfg = build_base_config()
    cfg["seed"] = int(base_seed)
    cfg["n_runs"] = int(n_runs)
    cfg["scenario_name"] = scenario_name
    cfg["experiment_id"] = EXPERIMENT_ID
    cfg["tier"] = TIER
    cfg["p_true_start"] = float(p_start)
    cfg["p_true_end"] = float(p_end)
    cfg["sweep_params"] = {"p_true_start": cfg["p_true_start"], "p_true_end": cfg["p_true_end"]}

    rng_master = np.random.default_rng(base_seed)
    model_names = cfg["models"]
    per_run_metrics = []
    p_hat_lists = {m: [] for m in model_names}
    outcome_lists = {m: [] for m in model_names}
    config_hash = stable_config_hash(cfg)

    for run_idx in range(n_runs):
        run_seed = int(rng_master.integers(0, 2**32 - 1))
        rng = np.random.default_rng(run_seed)

        p_path = list(np.linspace(cfg["p_true_start"], cfg["p_true_end"], cfg["market"]["n_steps"]))
        traders = build_traders(
            n_traders=int(cfg["traders"]["n_traders"]),
            proportions=cfg["traders"]["proportions"],
            params_by_type=build_trader_params(cfg["traders"].get("params", {})),
        )
        market_cfg = MarketConfig(**cfg["market"])
        ctx_token = set_invariant_context(
            InvariantContext(
                experiment_id=cfg.get("experiment_id"),
                scenario_name=cfg.get("scenario_name"),
                run_seed=run_seed,
                config_hash=config_hash,
            )
        )
        try:
            trade_tape, pool_path = simulate_market_dynamic(rng, market_cfg, traders, p_path)
            # use final p_true for downstream realized outcome
            p_true_final = p_path[-1]
            outcome = 1 if float(rng.random()) < p_true_final else 0

            y_raw, n_raw = counts_from_trade_tape(trade_tape)
            base_out = capopm_pipeline(
                rng=rng,
                trade_tape=trade_tape,
                structural_cfg=cfg["structural_cfg"],
                ml_cfg=cfg["ml_cfg"],
                prior_cfg=cfg["prior_cfg"],
                stage1_cfg=cfg.get("stage1_cfg", {"enabled": False}),
                stage2_cfg=cfg.get("stage2_cfg", {"enabled": False}),
            )
            var_independent = posterior_moments(base_out["alpha_post"], base_out["beta_post"])[1]

            model_outputs = {}
            for model in model_names:
                if model == "capopm":
                    out = capopm_pipeline(
                        rng=rng,
                        trade_tape=trade_tape,
                        structural_cfg=cfg["structural_cfg"],
                        ml_cfg=cfg["ml_cfg"],
                        prior_cfg=cfg["prior_cfg"],
                        stage1_cfg=cfg.get("stage1_cfg", {"enabled": False}),
                        stage2_cfg=cfg.get("stage2_cfg", {"enabled": False}),
                    )
                    p_hat = out["mixture_mean"] if out.get("mixture_enabled") else out["pi_yes"]
                    alpha = out["alpha_post"]
                    beta = out["beta_post"]
                elif model == "raw_parimutuel":
                    p_hat = pool_path[-1][3] if pool_path else 0.5
                    alpha = None
                    beta = None
                elif model == "structural_only":
                    out = capopm_pipeline(
                        rng=rng,
                        trade_tape=trade_tape,
                        structural_cfg=cfg["structural_cfg"],
                        ml_cfg={**cfg["ml_cfg"], "r_ml": 0.0, "noise_std": 0.0},
                        prior_cfg={**cfg["prior_cfg"], "n_ml_eff": 0.0},
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
                        structural_cfg=cfg["structural_cfg"],
                        ml_cfg=cfg["ml_cfg"],
                        prior_cfg={**cfg["prior_cfg"], "n_str": 0.0},
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
                    continue

                model_outputs[model] = {"p_hat": p_hat, "alpha": alpha, "beta": beta}
                p_hat_lists[model].append(p_hat)
                outcome_lists[model].append(int(outcome))

            metrics = {}
            coverage_include_outcome = bool(
                cfg.get("coverage_include_outcome", cfg.get("include_outcome_coverage", False))
            )
            for model, out in model_outputs.items():
                p_hat = out["p_hat"]
                alpha = out["alpha"]
                beta = out["beta"]
                metrics[model] = {
                    "brier": brier(p_true_final, p_hat),
                    "log_score": log_score(p_hat, outcome),
                    "mae_prob": mae_prob(p_true_final, p_hat),
                    "abs_error_outcome": abs(p_hat - outcome),
                    "calibration_ece": None,
                    "calibration_diagnostics": {},
                    "posterior_mean_bias": p_hat - p_true_final,
                    "p_hat": p_hat,
                }
                if alpha is not None and beta is not None:
                    cov90 = interval_coverage_ptrue(alpha, beta, p_true_final, 0.90)
                    cov95 = interval_coverage_ptrue(alpha, beta, p_true_final, 0.95)
                    _, var_adj = posterior_moments(alpha, beta)
                    metrics[model].update(
                        {
                            "posterior_variance_ratio": posterior_variance_ratio(var_adj, var_independent),
                            "mad_posterior_median": mad_posterior_median(alpha, beta, p_true_final),
                            "wasserstein_distance_beta": wasserstein_distance_beta(alpha, beta, p_true_final),
                            "coverage_90": cov90,
                            "coverage_95": cov95,
                            "coverage_90_outcome": np.nan,
                            "coverage_95_outcome": np.nan,
                            "coverage_outcome_90": np.nan,
                            "coverage_outcome_95": np.nan,
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

            per_run_metrics.append(
                {
                    "run": run_idx,
                    "p_true": p_true_final,
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
        # calibration across full run set
        for m in model_names:
            if len(p_hat_lists[m]) != len(outcome_lists[m]) or len(p_hat_lists[m]) < 2:
                continue
            unique_outcomes = set(outcome_lists[m])
            if not unique_outcomes.issubset({0, 1}):
                continue
            ece, diag = calibration_ece(
                p_hat_lists[m],
                outcome_lists[m],
                n_bins=cfg.get("ece_bins", 10),
                binning=cfg.get("calibration_binning", "equal_width"),
                min_nonempty_bins=cfg.get("ece_min_nonempty_bins", 5),
                allow_fallback=True,
            )
            aggregated[m]["calibration_ece"] = ece
            aggregated[m]["calibration_diagnostics"] = diag
        warnings = []
        scenario_invariant_log = [vars(rec) for rec in scenario_ctx.invariant_log]
        scenario_fallback_log = [vars(rec) for rec in scenario_ctx.fallback_log]
    finally:
        reset_invariant_context(scenario_token)

    meta = {
        "scenario_name": scenario_name,
        "experiment_id": EXPERIMENT_ID,
        "tier": TIER,
        "seed": base_seed,
        "config_hash": config_hash,
    }
    results = {
        "per_run_metrics": per_run_metrics,
        "aggregated_metrics": aggregated,
        "tests": {},
        "warnings": warnings,
        "note": "Descriptive drift stress run.",
        "metadata": meta,
        "reporting_version": REPORTING_VERSION,
        "scenario_invariants": scenario_invariant_log,
        "scenario_fallbacks": scenario_fallback_log,
        "sweep_params": cfg.get("sweep_params", {}),
        "grid_status": {"grid_axes": [], "grid_point": {}},
    }

    write_scenario_outputs(
        scenario_name=scenario_name,
        results=results,
        p_hat_lists=p_hat_lists,
        outcome_lists=outcome_lists,
        n_bins=cfg.get("ece_bins", 10),
        binning=cfg.get("calibration_binning", "equal_width"),
        min_nonempty_bins=cfg.get("ece_min_nonempty_bins", 5),
        experiment_id=EXPERIMENT_ID,
        tier=TIER,
        seed=base_seed,
        extra_metadata={},
        config_snapshot=cfg,
    )

    return [
        {
            "scenario_name": scenario_name,
            "seed": base_seed,
            "base_seed": base_seed,
            "run_index": 0,
            "sweep_params": cfg["sweep_params"],
        }
    ]
