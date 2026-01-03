"""
CAPOPM Phase 1–5 pipeline (no Phase 6 corrections).

Runs Phase 1 surrogate structural prior, Phase 2 ML prior + hybrid fusion,
Phase 4 Beta–Binomial update, and Phase 5 pricing outputs.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .structural_prior import compute_q_str, enforce_structural_invariants
from .ml_prior import compute_n_ml, compute_r_ml, simulate_p_ml
from .hybrid_prior import build_ml_beta, build_structural_beta, fuse_priors
from .likelihood import beta_binomial_update, counts_from_trade_tape
from .pricing import credible_intervals, posterior_prices
from .corrections.stage1_behavioral import apply_behavioral_weights
from .corrections.stage2_structural import (
    apply_linear_offsets,
    offset_summary,
    summarize_stage1_stats,
    mixture_posterior_params,
)
from .invariant_runtime import require_invariant


def capopm_pipeline(
    rng: Optional[np.random.Generator],
    trade_tape,
    structural_cfg: Dict,
    ml_cfg: Dict,
    prior_cfg: Dict,
    stage1_cfg: Optional[Dict] = None,
    stage2_cfg: Optional[Dict] = None,
) -> Dict:
    """Phase 1-5 pipeline: structural prior -> ML prior -> hybrid -> update -> prices."""

    structural_mode = structural_cfg.get("mode", "surrogate_heston")
    require_invariant(
        structural_mode == "surrogate_heston",
        invariant_id="S-0",
        message="Only surrogate_heston structural prior mode is permitted in Stage B.1",
        data={"mode": structural_mode},
    )
    # Phase 1: Structural prior (SURROGATE).
    q_str = compute_q_str(structural_cfg, rng=rng if structural_cfg.get("use_rng") else None)
    enforce_structural_invariants(structural_cfg, q_str)

    # Phase 2: ML prior.
    if rng is None:
        raise ValueError("rng must be provided for ML prior generation")
    p_ml = simulate_p_ml(rng, ml_cfg)
    r_ml = compute_r_ml(ml_cfg)
    n_ml = compute_n_ml(r_ml, float(prior_cfg.get("n_ml_eff", 0.0)), float(prior_cfg.get("n_ml_scale", 1.0)))
    require_invariant(
        0.0 <= p_ml <= 1.0,
        invariant_id="M-1",
        message="ML prior probability in [0,1]",
        tolerance=1e-8,
        data={"p_ml": float(p_ml)},
    )
    require_invariant(
        n_ml >= 0.0,
        invariant_id="B-1",
        message="n_ml nonnegative",
        tolerance=0.0,
        data={"n_ml": float(n_ml)},
    )

    # Phase 2: Hybrid prior fusion.
    n_str = float(prior_cfg.get("n_str", 0.0))
    alpha_str, beta_str = build_structural_beta(q_str, n_str)
    alpha_ml, beta_ml = build_ml_beta(p_ml, n_ml)
    alpha0, beta0 = fuse_priors(alpha_str, beta_str, alpha_ml, beta_ml)
    require_invariant(
        alpha0 > 0.0 and beta0 > 0.0,
        invariant_id="B-1",
        message="Prior alpha0/beta0 positive (likelihood normalization)",
        tolerance=0.0,
        data={"alpha0": float(alpha0), "beta0": float(beta0)},
    )

    # Phase 4: Counts and Beta-Binomial update (optional Phase 6.2 Stage 1).
    y_raw, n_raw = counts_from_trade_tape(trade_tape)
    y_used, n_used = y_raw, n_raw
    y_stage1 = None
    n_stage1 = None
    weights_summary = None
    if stage1_cfg is not None and bool(stage1_cfg.get("enabled", False)):
        y_used, n_used, weights_summary = apply_behavioral_weights(trade_tape, stage1_cfg)
        y_stage1, n_stage1 = y_used, n_used

    stage2_summary = None
    y_stage2 = None
    n_stage2 = None
    stage2_enabled = stage2_cfg is not None and bool(stage2_cfg.get("enabled", False))
    stage2_mode = (stage2_cfg or {}).get("mode", "offsets")
    if stage2_enabled and ("offsets" in stage2_mode):
        y_stage2, n_stage2 = apply_linear_offsets(y_used, n_used, stage2_cfg)
        stage2_summary = offset_summary(
            y1=y_used,
            n1=n_used,
            y_star=y_stage2,
            n_star=n_stage2,
            cfg=stage2_cfg,
        )
        y_used, n_used = y_stage2, n_stage2

    alpha_post, beta_post = beta_binomial_update(alpha0, beta0, y_used, n_used)
    require_invariant(
        alpha_post > 0.0 and beta_post > 0.0,
        invariant_id="B-2",
        message="Posterior alpha/beta positive",
        tolerance=0.0,
        data={"alpha_post": float(alpha_post), "beta_post": float(beta_post)},
    )

    # Phase 5: Posterior prices and credible intervals.
    pi_yes, pi_no = posterior_prices(alpha_post, beta_post)
    require_invariant(
        0.0 <= pi_yes <= 1.0 and 0.0 <= pi_no <= 1.0,
        invariant_id="M-1",
        message="Posterior prices within [0,1]",
        tolerance=1e-8,
        data={"pi_yes": float(pi_yes), "pi_no": float(pi_no)},
    )
    require_invariant(
        abs((pi_yes + pi_no) - 1.0) <= 1e-8,
        invariant_id="B-2",
        message="Posterior normalization pi_yes+pi_no=1",
        tolerance=1e-8,
        data={"pi_yes": float(pi_yes), "pi_no": float(pi_no)},
    )
    ci90 = credible_intervals(alpha_post, beta_post, level=0.90)
    ci95 = credible_intervals(alpha_post, beta_post, level=0.95)

    mixture_enabled = False
    mixture_mean = None
    mixture_var = None
    mixture_weights = None
    regime_params = None
    mixture_diagnostics = None
    if stage2_enabled and ("mixture" in stage2_mode):
        s = summarize_stage1_stats(trade_tape, y_used, n_used, stage2_cfg)
        mix = mixture_posterior_params(alpha0, beta0, s, stage2_cfg.get("regimes", []), y_used, n_used)
        mixture_enabled = True
        mixture_mean = mix["mixture_mean"]
        mixture_var = mix["mixture_var"]
        mixture_weights = mix["regime_weights"]
        regime_params = mix["regime_params"]
        mixture_diagnostics = mix["diagnostics"]
        require_invariant(
            all(w >= 0.0 for w in mixture_weights) and abs(sum(mixture_weights) - 1.0) <= 1e-8,
            invariant_id="M-2",
            message="Mixture weights normalized",
            tolerance=1e-8,
            data={"sum_weights": float(sum(mixture_weights)), "min_weight": float(min(mixture_weights or [0.0]))},
        )
        require_invariant(
            0.0 <= mixture_mean <= 1.0,
            invariant_id="M-1",
            message="Mixture mean within [0,1]",
            tolerance=1e-8,
            data={"mixture_mean": float(mixture_mean)},
        )

    return {
        "q_str": q_str,
        "p_ML": p_ml,
        "r_ML": r_ml,
        "n_ML": n_ml,
        "alpha0": alpha0,
        "beta0": beta0,
        "y": y_raw,
        "n": n_raw,
        "y_stage1": y_stage1,
        "n_stage1": n_stage1,
        "y_stage2": y_stage2,
        "n_stage2": n_stage2,
        "weights_summary": weights_summary,
        "stage2_summary": stage2_summary,
        "alpha_post": alpha_post,
        "beta_post": beta_post,
        "pi_yes": pi_yes,
        "pi_no": pi_no,
        "mixture_enabled": mixture_enabled,
        "mixture_mean": mixture_mean,
        "mixture_var": mixture_var,
        "mixture_weights": mixture_weights,
        "regime_params": regime_params,
        "mixture_diagnostics": mixture_diagnostics,
        "credible_intervals_90": ci90,
        "credible_intervals_95": ci95,
    }
