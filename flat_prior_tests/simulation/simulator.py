"""End-to-end driver for flat-prior synthetic simulation and v2 MP4 output."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from flat_prior_tests.data.imbalance_synth import generate_synth_imbalance
from flat_prior_tests.data.mbo_synth import generate_synth_mbo
from flat_prior_tests.priors.hybrid_mixture_prior import HybridMixturePrior, MixturePriorState
from flat_prior_tests.priors.ml_prior_principled import MLPriorConfig, PrincipledMLPrior
from flat_prior_tests.simulation.visualization import make_capopm_v2_animation
from src.capopm.corrections.stage1_behavioral import apply_behavioral_weights
from src.capopm.corrections.stage2_structural import (
    apply_linear_offsets,
    mixture_posterior_params,
    summarize_stage1_stats,
)

EPS = 1e-12


@dataclass
class SimulationConfig:
    cfg_path: str
    regimes_path: str
    out_dir: str
    symbol: str = "A"
    historical_years: int = 2
    rolling_days: int = 7
    rolling_samples: int = 50
    n_chains: int = 10
    live_plot: bool = False
    mode: str = "synthetic"


@dataclass
class SimpleTrade:
    implied_yes_before: float
    side: str
    size: float


class FlatPriorSimulationRunner:
    def __init__(self, sim_cfg: SimulationConfig):
        self.sim_cfg = sim_cfg
        with open(sim_cfg.cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.log = logging.getLogger("flat_prior_sim")
        self.log.setLevel(getattr(logging, self.cfg.get("logging", {}).get("level", "INFO")))
        if not self.log.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.log.addHandler(ch)

        Path(sim_cfg.out_dir).mkdir(parents=True, exist_ok=True)
        Path(sim_cfg.out_dir, "plots").mkdir(parents=True, exist_ok=True)
        Path(sim_cfg.out_dir, "data").mkdir(parents=True, exist_ok=True)
        Path(sim_cfg.out_dir, "mcmc_diagnostics").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _moment_match_beta(mean: float, var: float, default_conc: float = 25.0) -> Tuple[float, float]:
        mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
        var = float(var)
        if not np.isfinite(var) or var <= 0.0:
            return max(1e-3, mean * default_conc), max(1e-3, (1.0 - mean) * default_conc)
        conc = mean * (1.0 - mean) / max(var, 1e-12) - 1.0
        if conc <= 1e-6:
            conc = default_conc
        return max(1e-3, mean * conc), max(1e-3, (1.0 - mean) * conc)

    @staticmethod
    def _beta_mean_var(alpha: float, beta: float) -> Tuple[float, float]:
        alpha = max(alpha, 1e-9)
        beta = max(beta, 1e-9)
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        return float(mean), float(max(var, 0.0))

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _build_ml_prior(self) -> PrincipledMLPrior:
        ml_cfg = self.cfg.get("ml_prior", {})
        return PrincipledMLPrior(
            MLPriorConfig(
                model_type=ml_cfg.get("model_type", "ensemble_logistic"),
                n_models=int(ml_cfg.get("n_models", 16)),
                N_eff_min=float(ml_cfg.get("N_eff_min", 2.0)),
                N_eff_max=float(ml_cfg.get("N_eff_max", 120.0)),
                l2=float(ml_cfg.get("regularization", {}).get("l2", 1.0)),
                lookback_events=int(ml_cfg.get("features", {}).get("lookback_events", 2000)),
            ),
            logger=self.log,
            seed=int(self.cfg.get("simulation", {}).get("seed", 12345) + 99),
        )

    def _mixture_mean_var(self, state: MixturePriorState) -> Tuple[float, float]:
        mean_ml = state.alpha_ml / (state.alpha_ml + state.beta_ml)
        var_ml = (state.alpha_ml * state.beta_ml) / ((state.alpha_ml + state.beta_ml) ** 2 * (state.alpha_ml + state.beta_ml + 1.0))
        mean_flat = state.alpha_flat / (state.alpha_flat + state.beta_flat)
        var_flat = (state.alpha_flat * state.beta_flat) / ((state.alpha_flat + state.beta_flat) ** 2 * (state.alpha_flat + state.beta_flat + 1.0))
        mean_mix = state.weight * mean_ml + (1.0 - state.weight) * mean_flat
        second = state.weight * (var_ml + mean_ml**2) + (1.0 - state.weight) * (var_flat + mean_flat**2)
        var_mix = max(second - mean_mix**2, 1e-12)
        return float(mean_mix), float(var_mix)

    def _build_trade_tape(
        self,
        y: int,
        n: int,
        p_true: float,
        total_imbalance_qty: float,
        paired_qty: float,
        rng: np.random.Generator,
    ) -> List[SimpleTrade]:
        if n <= 0:
            return []
        yes_count = int(max(0, min(y, n)))
        no_count = int(max(0, n - yes_count))
        ratio = float(total_imbalance_qty / (abs(paired_qty) + 1.0))
        tape: List[SimpleTrade] = []
        base_size = max(1.0, 0.5 * np.sqrt(abs(paired_qty)) / max(1, n))
        for _ in range(yes_count):
            implied = np.clip(p_true + 0.06 * np.tanh(ratio) + rng.normal(0.0, 0.035), 1e-4, 1.0 - 1e-4)
            size = float(max(0.25, rng.lognormal(np.log(base_size + 1.0), 0.35)))
            tape.append(SimpleTrade(implied_yes_before=float(implied), side="YES", size=size))
        for _ in range(no_count):
            implied = np.clip(p_true - 0.06 * np.tanh(ratio) + rng.normal(0.0, 0.035), 1e-4, 1.0 - 1e-4)
            size = float(max(0.25, rng.lognormal(np.log(base_size + 1.0), 0.35)))
            tape.append(SimpleTrade(implied_yes_before=float(implied), side="NO", size=size))
        if tape:
            rng.shuffle(tape)
        return tape

    def _apply_stage_corrections(
        self,
        alpha_base: float,
        beta_base: float,
        tape: List[SimpleTrade],
        stage1_cfg: Dict,
        stage2_cfg: Dict,
    ) -> Tuple[float, float, float, float, float, float, Dict]:
        if not tape:
            mean, var = self._beta_mean_var(alpha_base, beta_base)
            return alpha_base, beta_base, mean, var, 0.0, 0.0, {}

        y1, n1, summary1 = apply_behavioral_weights(tape, stage1_cfg)
        y2, n2 = apply_linear_offsets(y1, n1, stage2_cfg)
        summary_stats = summarize_stage1_stats(tape, y2, n2, stage1_cfg)
        regimes = stage2_cfg.get("regimes")
        if not regimes:
            regimes = [
                {"pi": 0.50, "g_plus_scale": 0.08, "g_minus_scale": 0.05},
                {"pi": 0.30, "g_plus_scale": 0.16, "g_minus_scale": 0.11},
                {"pi": 0.20, "g_plus_scale": 0.04, "g_minus_scale": 0.17},
            ]
        mix = mixture_posterior_params(alpha_base, beta_base, summary_stats, regimes, y2, n2)
        mean_corr = float(np.clip(mix.get("mixture_mean", 0.5), 1e-4, 1.0 - 1e-4))
        var_corr = float(max(mix.get("mixture_var", 0.0), 1e-9))
        alpha_corr, beta_corr = self._moment_match_beta(mean_corr, var_corr, default_conc=18.0)

        stage1_strength = float(np.clip(1.0 - summary1.get("mean_weight", 1.0), 0.0, 1.0))
        stage2_shift = abs(float(y2 - y1)) + abs(float(n2 - n1))
        stage2_strength = float(np.clip(stage2_shift / max(1.0, n1 + 1.0), 0.0, 1.0))
        stage2_strength = float(
            np.clip(
                max(stage2_strength, np.std(np.asarray(mix.get("regime_weights", [0.0])))),
                0.0,
                1.0,
            )
        )
        return alpha_corr, beta_corr, mean_corr, var_corr, stage1_strength, stage2_strength, mix

    def run(self):
        run_start = time.time()
        self.log.info(
            "Simulation run start (mode=%s, out_dir=%s, live_plot=%s)",
            self.sim_cfg.mode,
            self.sim_cfg.out_dir,
            self.sim_cfg.live_plot,
        )
        if self.sim_cfg.mode != "synthetic":
            raise ValueError("Only synthetic mode is supported.")

        sim_cfg = self.cfg.get("simulation", {})
        training_cfg = self.cfg.get("training", {})
        synth_cfg = self.cfg.get("synthetic", {})
        vis_cfg = self.cfg.get("visualization", {})
        corr_cfg = self.cfg.get("corrections", {})
        stage1_cfg = corr_cfg.get("stage1", {})
        stage2_cfg = corr_cfg.get("stage2", {})
        posterior_cfg = self.cfg.get("posterior_dynamics", {})

        seed = int(sim_cfg.get("seed", 12345))
        rng_synth = np.random.default_rng(seed)
        rng_post = np.random.default_rng(seed + 7)

        total_days = int(sim_cfg.get("total_days", self.sim_cfg.historical_years * 365))
        training_days = int(training_cfg.get("training_days", int(0.7 * total_days)))
        incoming_days = int(training_cfg.get("incoming_days", max(1, total_days - training_days)))
        training_days = max(1, min(training_days, total_days - 1))
        incoming_days = max(1, min(incoming_days, total_days - training_days))
        total_days = training_days + incoming_days

        price_scale = float(sim_cfg.get("price_scale", 1e-9))
        tick_size = float(synth_cfg.get("tick_size", 0.01))
        max_events = int(synth_cfg.get("max_events", max(20000, total_days * int(synth_cfg.get("avg_events_per_day", 180)))))

        gen_t0 = time.time()
        events_int, daily = generate_synth_mbo(
            instrument_id=int(synth_cfg.get("instrument_id", 1)),
            start_ts_event_ns=int(synth_cfg.get("start_ts_event_ns", 0)),
            avg_events_per_day=float(synth_cfg.get("avg_events_per_day", 180.0)),
            initial_mid=float(synth_cfg.get("initial_mid", 100.0)),
            tick_size=tick_size,
            order_id_start=int(synth_cfg.get("order_id_start", 1000)),
            price_scale=price_scale,
            days=total_days,
            rng=rng_synth,
            logger=self.log,
            max_events=max_events,
            process_cfg=synth_cfg.get("price_process", {}),
            return_daily=True,
        )
        self.log.info("Synthetic MBO generated in %.2fs", time.time() - gen_t0)

        events = events_int.copy()
        events["price_raw"] = events["price"].astype(np.int64)
        events["price"] = events["price_raw"].astype(np.float64) * price_scale

        imbalance_t0 = time.time()
        imbalance_df = generate_synth_imbalance(
            daily=daily,
            instrument_id=int(synth_cfg.get("instrument_id", 1)),
            start_ts_event_ns=int(synth_cfg.get("start_ts_event_ns", 0)),
            tick_size=tick_size,
            price_scale=price_scale,
            rng=np.random.default_rng(seed + 1),
            logger=self.log,
            cfg=synth_cfg.get("imbalance_synth", {}),
        )
        self.log.info("Synthetic imbalance generated in %.2fs", time.time() - imbalance_t0)

        imbalance_out = Path(self.sim_cfg.out_dir, "data", "synthetic_imbalance.csv")
        imbalance_df.to_csv(imbalance_out, index=False)

        fill_daily = (
            events_int.loc[events_int["action"] == "F", ["sim_day", "size"]]
            .groupby("sim_day", as_index=False)
            .agg(fill_count=("size", "count"), fill_volume=("size", "sum"))
        )
        imb_daily = (
            imbalance_df.groupby("sim_day", as_index=False)
            .agg(
                paired_qty=("paired_qty", "sum"),
                total_imbalance_qty=("total_imbalance_qty", "sum"),
                imbalance_events=("ts_event", "count"),
                auction_events=("auction_status", lambda x: int(np.sum(np.asarray(x) != 0))),
                opening_auctions=("auction_status", lambda x: int(np.sum(np.asarray(x) == 1))),
                closing_auctions=("auction_status", lambda x: int(np.sum(np.asarray(x) == 2))),
                halt_events=("auction_status", lambda x: int(np.sum(np.asarray(x) == 3))),
            )
            .reset_index(drop=True)
        )

        daily_panel = daily.merge(imb_daily, on="sim_day", how="left").merge(fill_daily, on="sim_day", how="left")
        for col in [
            "paired_qty",
            "total_imbalance_qty",
            "imbalance_events",
            "auction_events",
            "opening_auctions",
            "closing_auctions",
            "halt_events",
            "fill_count",
            "fill_volume",
        ]:
            daily_panel[col] = daily_panel[col].fillna(0.0)
        daily_panel = daily_panel.sort_values("sim_day", kind="mergesort").reset_index(drop=True)

        # Phase-1 diagnostic logs requested by user.
        vol_stats = daily_panel["realized_vol_10d"].describe(percentiles=[0.1, 0.5, 0.9]).to_dict()
        self.log.info(
            "Per-day realized vol stats: p10=%.6f p50=%.6f p90=%.6f max=%.6f",
            float(vol_stats.get("10%", 0.0)),
            float(vol_stats.get("50%", 0.0)),
            float(vol_stats.get("90%", 0.0)),
            float(vol_stats.get("max", 0.0)),
        )

        ml_prior = self._build_ml_prior()
        train_events = events.loc[events["sim_day"] < training_days].copy()
        if train_events.empty:
            train_events = events.copy()
        alpha_ml, beta_ml, ml_diag = ml_prior.predict_beta(train_events)

        mixture = HybridMixturePrior(
            weight=float(self.cfg.get("hybrid_prior", {}).get("mixture_weight_w", 0.55)),
            logger=self.log,
        )
        mix_state = mixture.initialize(alpha_ml, beta_ml)
        mean0, var0 = self._mixture_mean_var(mix_state)
        alpha_state, beta_state = self._moment_match_beta(mean0, var0, default_conc=24.0)

        days = daily_panel["sim_day"].to_numpy(dtype=np.float64)
        mid_prices = daily_panel["mid_close"].to_numpy(dtype=np.float64)
        if len(days) != total_days:
            total_days = len(days)
            training_days = max(1, min(training_days, total_days - 1))
            incoming_days = max(1, min(incoming_days, total_days - training_days))

        alpha_full = np.full(total_days, alpha_state, dtype=np.float64)
        beta_full = np.full(total_days, beta_state, dtype=np.float64)
        posterior_mean = np.full(total_days, mean0, dtype=np.float64)
        posterior_var = np.full(total_days, var0, dtype=np.float64)
        p_true_full = np.full(total_days, np.nan, dtype=np.float64)
        y_full = np.zeros(total_days, dtype=np.int32)
        n_full = np.zeros(total_days, dtype=np.int32)
        stage1_strength = np.zeros(total_days, dtype=np.float64)
        stage2_strength = np.zeros(total_days, dtype=np.float64)
        corr_strength = np.zeros(total_days, dtype=np.float64)
        corr_active = np.zeros(total_days, dtype=bool)

        # Signal prep for dynamic p_true(t).
        imbalance_ratio = daily_panel["total_imbalance_qty"].to_numpy(dtype=np.float64) / (
            np.abs(daily_panel["paired_qty"].to_numpy(dtype=np.float64)) + 1.0
        )
        imb_std = float(np.std(imbalance_ratio)) + 1e-9
        imb_z = (imbalance_ratio - float(np.mean(imbalance_ratio))) / imb_std
        trend_signal = (
            pd.Series(np.log(mid_prices + 1e-9))
            .diff()
            .rolling(window=int(posterior_cfg.get("trend_window", 8)), min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        regime_effect = np.asarray(posterior_cfg.get("regime_effect", [0.0, 0.85, 0.40, -0.35]), dtype=np.float64)
        if regime_effect.size <= int(daily_panel["regime_id"].max()):
            pad = int(daily_panel["regime_id"].max()) + 1 - regime_effect.size
            regime_effect = np.pad(regime_effect, (0, pad), mode="edge")

        logistic_a0 = float(posterior_cfg.get("a0", -0.02))
        logistic_a_imb = float(posterior_cfg.get("a_imbalance", 0.85))
        logistic_a_trend = float(posterior_cfg.get("a_trend", 6.5))
        logistic_a_auc = float(posterior_cfg.get("a_auction", 0.40))
        p_noise = float(posterior_cfg.get("p_noise_sigma", 0.18))
        p_clip = posterior_cfg.get("p_clip", [0.02, 0.98])
        p_clip_lo = float(p_clip[0])
        p_clip_hi = float(p_clip[1])

        n_base = float(posterior_cfg.get("n_base", 2.0))
        n_fill_scale = float(posterior_cfg.get("n_fill_scale", 0.35))
        n_pair_scale = float(posterior_cfg.get("n_pair_scale", 0.30))
        n_auction_scale = float(posterior_cfg.get("n_auction_scale", 1.20))
        n_vol_scale = float(posterior_cfg.get("n_vol_scale", 4.0))
        burst_prob = float(posterior_cfg.get("burst_prob", 0.28))
        burst_low = int(posterior_cfg.get("burst_low", 4))
        burst_high = int(posterior_cfg.get("burst_high", 11))
        correction_carry = float(posterior_cfg.get("correction_carry", 0.62))
        active_threshold = float(corr_cfg.get("active_threshold", 0.12))

        # Training is explicitly frozen.
        incoming_start = training_days
        incoming_end = min(total_days, training_days + incoming_days)
        incoming_rows: List[Dict] = []
        self.log.info(
            "Posterior incoming loop start: training_days=%d incoming_days=%d",
            training_days,
            incoming_end - incoming_start,
        )
        for day_idx in range(incoming_start, incoming_end):
            row = daily_panel.iloc[day_idx]
            reg_id = int(row["regime_id"])
            auc_intensity = float(row["auction_events"] / max(1.0, row["imbalance_events"]))
            reg_term = float(regime_effect[reg_id]) if reg_id < len(regime_effect) else float(regime_effect[-1])
            latent = (
                logistic_a0
                + reg_term
                + logistic_a_imb * float(imb_z[day_idx])
                + logistic_a_trend * float(trend_signal[day_idx])
                + logistic_a_auc * auc_intensity
                + float(rng_post.normal(0.0, p_noise))
            )
            p_true = float(np.clip(self._sigmoid(latent), p_clip_lo, p_clip_hi))

            n_signal = (
                n_base
                + n_fill_scale * np.log1p(float(row["fill_volume"]))
                + n_pair_scale * np.log1p(abs(float(row["paired_qty"])))
                + n_auction_scale * auc_intensity
                + n_vol_scale * float(row["realized_vol_10d"])
            )
            if rng_post.random() < burst_prob:
                n_signal += int(rng_post.integers(burst_low, burst_high + 1))
            n_obs = int(np.clip(np.round(rng_post.normal(n_signal, 2.2)), 0, 20))
            y_obs = int(rng_post.binomial(n_obs, p_true)) if n_obs > 0 else 0

            alpha_state = max(1e-3, alpha_state + y_obs)
            beta_state = max(1e-3, beta_state + (n_obs - y_obs))
            base_mean, base_var = self._beta_mean_var(alpha_state, beta_state)

            tape = self._build_trade_tape(
                y=y_obs,
                n=n_obs,
                p_true=p_true,
                total_imbalance_qty=float(row["total_imbalance_qty"]),
                paired_qty=float(row["paired_qty"]),
                rng=rng_post,
            )
            alpha_corr, beta_corr, mean_corr, var_corr, s1, s2, corr_diag = self._apply_stage_corrections(
                alpha_base=alpha_state,
                beta_base=beta_state,
                tape=tape,
                stage1_cfg=stage1_cfg,
                stage2_cfg=stage2_cfg,
            )

            alpha_state = max(1e-3, (1.0 - correction_carry) * alpha_state + correction_carry * alpha_corr)
            beta_state = max(1e-3, (1.0 - correction_carry) * beta_state + correction_carry * beta_corr)
            post_mean, post_var = self._beta_mean_var(alpha_state, beta_state)

            local_strength = float(np.clip(0.55 * s1 + 0.45 * s2 + 0.25 * abs(mean_corr - base_mean), 0.0, 1.0))
            active = bool(local_strength > active_threshold)

            alpha_full[day_idx] = alpha_state
            beta_full[day_idx] = beta_state
            posterior_mean[day_idx] = post_mean
            posterior_var[day_idx] = post_var
            p_true_full[day_idx] = p_true
            y_full[day_idx] = y_obs
            n_full[day_idx] = n_obs
            stage1_strength[day_idx] = s1
            stage2_strength[day_idx] = s2
            corr_strength[day_idx] = local_strength
            corr_active[day_idx] = active

            incoming_rows.append(
                {
                    "sim_day": int(day_idx),
                    "regime_id": reg_id,
                    "p_true": p_true,
                    "y": y_obs,
                    "n": n_obs,
                    "alpha": alpha_state,
                    "beta": beta_state,
                    "posterior_mean": post_mean,
                    "posterior_var": post_var,
                    "stage1_strength": s1,
                    "stage2_strength": s2,
                    "correction_strength": local_strength,
                    "correction_active": int(active),
                    "imbalance_ratio": float(imbalance_ratio[day_idx]),
                    "auction_intensity": auc_intensity,
                    "base_posterior_mean": base_mean,
                    "base_posterior_var": base_var,
                    "regime_weights": corr_diag.get("regime_weights"),
                }
            )

        incoming_df = pd.DataFrame.from_records(incoming_rows)
        if not incoming_df.empty:
            self.log.info(
                "Incoming y,n summary: n[min=%d max=%d mean=%.2f] y[min=%d max=%d mean=%.2f]",
                int(incoming_df["n"].min()),
                int(incoming_df["n"].max()),
                float(incoming_df["n"].mean()),
                int(incoming_df["y"].min()),
                int(incoming_df["y"].max()),
                float(incoming_df["y"].mean()),
            )
            self.log.info(
                "Incoming posterior mean/var summary: mean[min=%.4f max=%.4f] var[min=%.6f max=%.6f]",
                float(incoming_df["posterior_mean"].min()),
                float(incoming_df["posterior_mean"].max()),
                float(incoming_df["posterior_var"].min()),
                float(incoming_df["posterior_var"].max()),
            )
            self.log.info(
                "Incoming alpha/beta ranges: alpha[min=%.4f max=%.4f] beta[min=%.4f max=%.4f]",
                float(incoming_df["alpha"].min()),
                float(incoming_df["alpha"].max()),
                float(incoming_df["beta"].min()),
                float(incoming_df["beta"].max()),
            )
        else:
            self.log.warning("Incoming loop produced no rows; posterior remains at training prior.")

        # Persist outputs.
        events_int.sort_values("ts_event", kind="mergesort").to_csv(
            Path(self.sim_cfg.out_dir, "synthetic_mbo.csv"), index=False
        )
        incoming_df.to_csv(Path(self.sim_cfg.out_dir, "posterior_windows.csv"), index=False)
        incoming_df.to_csv(Path(self.sim_cfg.out_dir, "trade_update_log.csv"), index=False)
        daily_out = daily_panel.copy()
        daily_out["posterior_mean"] = posterior_mean
        daily_out["posterior_var"] = posterior_var
        daily_out["alpha"] = alpha_full
        daily_out["beta"] = beta_full
        daily_out["p_true"] = p_true_full
        daily_out["y"] = y_full
        daily_out["n"] = n_full
        daily_out["stage1_strength"] = stage1_strength
        daily_out["stage2_strength"] = stage2_strength
        daily_out["correction_strength"] = corr_strength
        daily_out["correction_active"] = corr_active.astype(np.int8)
        daily_out.to_csv(Path(self.sim_cfg.out_dir, "posterior_daily_diagnostics.csv"), index=False)

        np.savez(
            Path(self.sim_cfg.out_dir, "posterior_draws.npz"),
            posterior_mean=posterior_mean,
            posterior_var=posterior_var,
            alpha=alpha_full,
            beta=beta_full,
            p_true=p_true_full,
        )

        manifest = {
            "config": self.sim_cfg.cfg_path,
            "regimes": self.sim_cfg.regimes_path,
            "out_dir": self.sim_cfg.out_dir,
            "mode": self.sim_cfg.mode,
            "seed": seed,
            "total_days": total_days,
            "training_days": training_days,
            "incoming_days": incoming_end - incoming_start,
            "ml_prior": ml_diag,
            "mp4_name": vis_cfg.get("out_mp4_name", "capopm_v2.mp4"),
        }
        with open(Path(self.sim_cfg.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        incoming_days_arr = days[incoming_start:incoming_end]
        alpha_in_arr = alpha_full[incoming_start:incoming_end]
        beta_in_arr = beta_full[incoming_start:incoming_end]
        mp4_path = Path(self.sim_cfg.out_dir, "plots", vis_cfg.get("out_mp4_name", "capopm_v2.mp4"))
        make_capopm_v2_animation(
            full_days=days,
            full_prices=mid_prices,
            incoming_days=incoming_days_arr,
            alpha_incoming=alpha_in_arr,
            beta_incoming=beta_in_arr,
            training_cutoff_day=float(training_days),
            correction_active=corr_active,
            correction_strength=corr_strength,
            stage1_strength=stage1_strength,
            stage2_strength=stage2_strength,
            save_path=str(mp4_path),
            fps=int(vis_cfg.get("fps", 30)),
            max_frames=int(vis_cfg.get("max_frames", 900)),
            posterior_grid_points=int(vis_cfg.get("posterior_grid_points", 140)),
            posterior_rolling_window=int(vis_cfg.get("posterior_rolling_window", 120)),
            camera_elev=float(vis_cfg.get("camera_elev", 28)),
            camera_azim=float(vis_cfg.get("camera_azim", -55)),
            logger=self.log,
        )

        self.log.info(
            "Simulation complete. Outputs stored under %s (total_elapsed=%.2fs)",
            self.sim_cfg.out_dir,
            time.time() - run_start,
        )
