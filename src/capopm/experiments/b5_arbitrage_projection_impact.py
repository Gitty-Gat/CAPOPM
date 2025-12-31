"""
Tier B Experiment B5: ARBITRAGE_PROJECTION_IMPACT.

Evaluates projection effects under coherence violations vs coherent regime.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..market_simulator import MarketConfig, simulate_market
from ..metrics.calibration import calibration_ece
from ..metrics.scoring import brier, log_score, mae_prob
from ..posterior import capopm_pipeline
from ..trader_model import build_traders
from ..stats.tests import bootstrap_ci, bonferroni_correction, holm_correction, paired_t_test, wilcoxon_signed_rank
from .projection_utils import detect_violation, project_probs, projection_distance
from .runner import (
    REPORTING_VERSION,
    aggregate_metrics,
    build_trader_params,
    draw_p_true,
    format_p_value,
    write_scenario_outputs,
)

EXPERIMENT_ID = "B5.ARBITRAGE_PROJECTION_IMPACT"
TIER = "B"

DEFAULT_VIOLATION_STRENGTHS = [0.0, 0.3, 0.6]
DEFAULT_PROJECTION_METHOD = "euclidean"
DEFAULT_BASE_SEED = 202780
DEFAULT_N_RUNS = 60
EPS = 1e-9
VIOLATION_PUSH = 0.35


def run_b5_arbitrage_projection_impact(
    violation_strength_grid: Iterable[float] = DEFAULT_VIOLATION_STRENGTHS,
    projection_method: str = DEFAULT_PROJECTION_METHOD,
    base_seed: int = DEFAULT_BASE_SEED,
    n_runs: int = DEFAULT_N_RUNS,
) -> List[Dict]:
    results = []
    for idx, strength in enumerate(violation_strength_grid):
        seed = base_seed + idx
        scenario_name = scenario_id(projection_method, strength, seed)
        cfg = build_base_config()
        cfg["seed"] = seed
        cfg["n_runs"] = n_runs
        cfg["scenario_name"] = scenario_name
        cfg["experiment_id"] = EXPERIMENT_ID
        cfg["tier"] = TIER
        cfg["projection_method"] = projection_method
        cfg["sweep_params"] = {
            "violation_strength": float(strength),
            "projection_method": projection_method,
            "coherent_flag": bool(abs(float(strength)) < 1e-12),
        }
        run_b5_scenario(cfg)
        results.append({"scenario_name": scenario_name, "seed": seed, "sweep_params": cfg["sweep_params"]})
    return results


def run_b5_scenario(config: Dict) -> Dict:
    cfg = copy.deepcopy(config)
    seed = int(cfg.get("seed", 123))
    n_runs = int(cfg.get("n_runs", 5))
    calibration_binning = cfg.get("calibration_binning", cfg.get("ece_binning", "equal_width"))
    min_nonempty_bins = int(cfg.get("ece_min_nonempty_bins", 5))
    n_bins = int(cfg.get("ece_bins", 10))
    rng_master = np.random.default_rng(seed)

    model_names = cfg.get("models", ["capopm", "before_projection", "after_projection"])

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

        base_out = capopm_pipeline(
            rng=rng,
            trade_tape=trade_tape,
            structural_cfg=cfg["structural_cfg"],
            ml_cfg=cfg["ml_cfg"],
            prior_cfg=cfg["prior_cfg"],
            stage1_cfg={"enabled": False},
            stage2_cfg={"enabled": False},
        )
        p_raw_yes = float(base_out["pi_yes"])
        p_raw_no = float(base_out["pi_no"])

        violation_strength = float(cfg.get("sweep_params", {}).get("violation_strength", 0.0))
        proj_method = str(cfg.get("projection_method", DEFAULT_PROJECTION_METHOD))
        before_vec = inject_violation([p_raw_yes, p_raw_no], violation_strength, eps=EPS)
        detection = detect_violation(before_vec, tol=1e-9, require_sum=True)
        after_vec = project_probs(before_vec, method=proj_method, eps=EPS)
        distances = projection_distance(before_vec, after_vec, metrics=["l1", "l2", "kl"], eps=EPS)

        p_before = before_vec[0]
        p_after = after_vec[0]

        before_metrics = {
            "brier": brier(p_true, p_before),
            "log_score": log_score(p_before, outcome),
            "mae_prob": mae_prob(p_true, p_before),
            "abs_error_outcome": abs(p_before - outcome),
            "calibration_ece": None,
            "calibration_diagnostics": {},
            "posterior_mean_bias": p_before - p_true,
            "p_hat": p_before,
            "proj_l1": 0.0,
            "proj_l2": 0.0,
            "proj_kl": 0.0,
            "delta_brier": 0.0,
            "delta_log_score": 0.0,
            "delta_abs_error_outcome": 0.0,
            "frac_violations_detected": 1.0 if detection["violated"] else 0.0,
            "coherent_flag": cfg["sweep_params"].get("coherent_flag", False),
        }

        after_metrics = {
            "brier": brier(p_true, p_after),
            "log_score": log_score(p_after, outcome),
            "mae_prob": mae_prob(p_true, p_after),
            "abs_error_outcome": abs(p_after - outcome),
            "calibration_ece": None,
            "calibration_diagnostics": {},
            "posterior_mean_bias": p_after - p_true,
            "p_hat": p_after,
            "proj_l1": distances.get("proj_l1", 0.0),
            "proj_l2": distances.get("proj_l2", 0.0),
            "proj_kl": distances.get("proj_kl", 0.0),
            "delta_brier": None,
            "delta_log_score": None,
            "delta_abs_error_outcome": None,
            "frac_violations_detected": 1.0 if detection["violated"] else 0.0,
            "coherent_flag": cfg["sweep_params"].get("coherent_flag", False),
        }

        after_metrics["delta_brier"] = after_metrics["brier"] - before_metrics["brier"]
        after_metrics["delta_log_score"] = after_metrics["log_score"] - before_metrics["log_score"]
        after_metrics["delta_abs_error_outcome"] = after_metrics["abs_error_outcome"] - before_metrics["abs_error_outcome"]

        capopm_metrics = {
            "brier": brier(p_true, p_raw_yes),
            "log_score": log_score(p_raw_yes, outcome),
            "mae_prob": mae_prob(p_true, p_raw_yes),
            "abs_error_outcome": abs(p_raw_yes - outcome),
            "calibration_ece": None,
            "calibration_diagnostics": {},
            "posterior_mean_bias": p_raw_yes - p_true,
            "p_hat": p_raw_yes,
            "proj_l1": 0.0,
            "proj_l2": 0.0,
            "proj_kl": 0.0,
            "delta_brier": 0.0,
            "delta_log_score": 0.0,
            "delta_abs_error_outcome": 0.0,
            "frac_violations_detected": 0.0,
            "coherent_flag": cfg["sweep_params"].get("coherent_flag", False),
        }

        metrics = {
            "capopm": capopm_metrics,
            "before_projection": before_metrics,
            "after_projection": after_metrics,
        }

        per_run_metrics.append({"run": run_idx, "p_true": p_true, "outcome": outcome, "metrics": metrics})
        for model in model_names:
            p_hat_lists[model].append(metrics[model]["p_hat"])
            outcome_lists[model].append(int(outcome))

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

    tests = run_b5_tests(per_run_metrics)

    warnings = []
    for m in model_names:
        diag = aggregated[m].get("calibration_diagnostics", {})
        if diag and diag.get("degenerate_binning"):
            warnings.append(f"Calibration diagnostics: degenerate binning detected for model '{m}'.")

    summary = build_summary(aggregated, cfg)
    scenario_name = cfg.get("scenario_name")
    results = {
        "per_run_metrics": per_run_metrics,
        "aggregated_metrics": aggregated,
        "tests": tests,
        "warnings": warnings,
        "note": "Projection distances and score deltas relative to unprojected probabilities.",
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


def run_b5_tests(per_run_metrics: List[Dict]) -> List[Dict]:
    metrics = [
        {"name": "brier", "better_if_negative": True},
        {"name": "log_score", "better_if_negative": False},
        {"name": "abs_error_outcome", "better_if_negative": True},
        {"name": "proj_l1", "better_if_negative": True},
        {"name": "proj_l2", "better_if_negative": True},
    ]
    rows = []
    for metric in metrics:
        metric_name = metric["name"]
        better_if_negative = metric["better_if_negative"]
        after_vals, before_vals = paired_values(per_run_metrics, metric_name)
        if len(after_vals) < 2:
            continue
        t_stat, p_t = paired_t_test(after_vals, before_vals)
        w_stat, p_w = wilcoxon_signed_rank(after_vals, before_vals)
        diff = np.asarray(after_vals) - np.asarray(before_vals)
        ci_lo, ci_hi = bootstrap_ci(diff)
        diff_mean = float(np.nanmean(diff)) if diff.size > 0 else float("nan")
        p_h = holm_correction([p_t])[0]
        p_b = bonferroni_correction([p_t])[0]

        rows.append(
            {
                "model": "after_projection",
                "metric": f"delta_{metric_name}",
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
                "model": "after_projection",
                "metric": f"delta_{metric_name}",
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
                "model": "after_projection",
                "metric": f"delta_{metric_name}",
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


def paired_values(per_run_metrics: List[Dict], metric_name: str) -> Tuple[List[float], List[float]]:
    after_vals = []
    before_vals = []
    for run in per_run_metrics:
        after = run["metrics"]["after_projection"].get(metric_name)
        before = run["metrics"]["before_projection"].get(metric_name)
        if after is None or before is None:
            continue
        if isinstance(after, float) and math.isnan(after):
            continue
        if isinstance(before, float) and math.isnan(before):
            continue
        after_vals.append(float(after))
        before_vals.append(float(before))
    return after_vals, before_vals


def build_summary(aggregated: Dict, config: Dict) -> Dict:
    after = aggregated.get("after_projection", {})
    coherent = bool(config.get("sweep_params", {}).get("coherent_flag", False))
    proj_l1 = after.get("proj_l1", float("nan"))
    delta_brier = after.get("delta_brier", float("nan"))
    delta_log = after.get("delta_log_score", float("nan"))
    distances_small = math.isfinite(proj_l1) and abs(proj_l1) <= 1e-6
    deltas_small = all(
        math.isfinite(x) and abs(x) <= 1e-6 for x in [delta_brier, delta_log if math.isfinite(delta_log) else 0.0]
    )
    coherent_pass = coherent and distances_small and deltas_small
    violated_pass = (not coherent) and (proj_l1 > 1e-6) and (math.isfinite(delta_brier) and delta_brier < 0.0) and (
        math.isfinite(delta_log) and delta_log > 0.0
    )
    return {
        "status": {
            "pass": bool(coherent_pass or violated_pass),
            "criteria": {
                "coherent_min_adjustment": True,
                "violated_improves_scores": True,
            },
            "metrics": {
                "proj_l1": proj_l1,
                "delta_brier": delta_brier,
                "delta_log_score": delta_log,
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


def scenario_id(projection_method: str, violation_strength: float, seed: int) -> str:
    return f"B5_arbitrage_proj__method{projection_method}__viol{int(round(violation_strength*100))}__seed{seed}"


def inject_violation(prob_vec: List[float], strength: float, eps: float = 1e-12) -> List[float]:
    """Inject coherence violation by adding mass to YES side only."""

    if len(prob_vec) != 2:
        raise ValueError("Binary probability vector expected for violation injection")
    bump = max(strength, 0.0) * VIOLATION_PUSH
    p_yes = max(min(prob_vec[0] + bump, 1.0 - eps), eps)
    p_no = max(min(prob_vec[1], 1.0 - eps), eps)
    return [p_yes, p_no]


def build_base_config() -> Dict:
    cfg = {
        "p_true_dist": {"type": "fixed", "value": 0.55},
        "models": ["capopm", "before_projection", "after_projection"],
        "traders": {
            "n_traders": 40,
            "proportions": {"informed": 0.5, "adversarial": 0.1, "noise": 0.4},
            "params": {
                "informed": {"signal_quality": 0.7, "noise_yes_prob": 0.5, "herding_intensity": 0.0},
                "adversarial": {"signal_quality": 0.7, "noise_yes_prob": 0.5, "herding_intensity": 0.0},
                "noise": {"signal_quality": 0.5, "noise_yes_prob": 0.5, "herding_intensity": 0.0},
            },
        },
        "market": {
            "n_steps": 20,
            "arrivals_per_step": 3,
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
        "stage1_cfg": {"enabled": False},
        "stage2_cfg": {"enabled": False},
        "ece_bins": 10,
        "calibration_binning": "equal_width",
        "ece_min_nonempty_bins": 5,
        "coverage_include_outcome": False,
    }
    return copy.deepcopy(cfg)
