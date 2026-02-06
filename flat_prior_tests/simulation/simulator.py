"""
End-to-end driver for the flat prior simulation path.

Pipeline:
1) Generate synthetic Databento-form L3 data (event-time).
2) Build principled ML prior -> Beta(alpha_ml, beta_ml).
3) Form hybrid mixture with flat structural Beta(1,1).
4) For each window (historical horizon + random rolling):
   - map L3 events to (y, n)
   - update mixture posterior
   - record posterior draws and diagnostics
   - run event-time MCMC extrapolation seeded by current state
5) Emit artifacts and a live animation.
"""

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

from flat_prior_tests.data.mbo_synth import generate_synth_mbo
from flat_prior_tests.diagnostics.mcmc_diagnostics import summarize_metric
from flat_prior_tests.mcmc.event_time_mcmc import EventTimeMCMCSampler, load_regime_config
from flat_prior_tests.priors.hybrid_mixture_prior import HybridMixturePrior
from flat_prior_tests.priors.ml_prior_principled import MLPriorConfig, PrincipledMLPrior
from flat_prior_tests.simulation.map_l3_to_counts import map_l3_to_counts
from flat_prior_tests.simulation.visualization import make_dual_panel_animation, make_live_animation
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
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        if not self.log.handlers:
            self.log.addHandler(ch)

        Path(sim_cfg.out_dir, "plots").mkdir(parents=True, exist_ok=True)
        Path(sim_cfg.out_dir, "mcmc_diagnostics").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _moment_match_beta(mean: float, var: float, default_conc: float = 20.0) -> Tuple[float, float]:
        """Moment-match a Beta distribution from mean/variance with a safe fallback."""

        mean = float(mean)
        var = float(var)
        if not np.isfinite(mean) or mean <= 0.0 or mean >= 1.0:
            mean = 0.5
        if not np.isfinite(var) or var <= 0.0:
            alpha = mean * default_conc
            beta = (1.0 - mean) * default_conc
            return max(alpha, 1e-3), max(beta, 1e-3)

        conc = mean * (1.0 - mean) / var - 1.0
        if conc <= 0.0:
            alpha = mean * default_conc
            beta = (1.0 - mean) * default_conc
        else:
            alpha = mean * conc
            beta = (1.0 - mean) * conc
        return max(alpha, 1e-3), max(beta, 1e-3)

    def _build_trade_tape(self, fills_diag: List[Dict]) -> List[SimpleTrade]:
        """Convert fill diagnostics into the minimal trade tape required by corrections."""

        tape: List[SimpleTrade] = []
        for fill in fills_diag:
            mid_before = fill.get("mid_before", np.nan)
            mid_after = fill.get("mid_after", np.nan)
            delta = 0.0
            if np.isfinite(mid_before) and np.isfinite(mid_after) and mid_before != 0:
                delta = (mid_after - mid_before) / abs(mid_before)
            implied_yes = 0.5 + 0.1 * np.tanh(delta)
            implied_yes = min(max(implied_yes, 1e-4), 1.0 - 1e-4)
            side = "YES" if fill.get("yes") else "NO"
            size = float(fill.get("size", 1.0))
            if size <= 0.0:
                size = 1.0
            tape.append(SimpleTrade(implied_yes_before=implied_yes, side=side, size=size))
        return tape

    def _apply_stage_corrections(
        self,
        fills_diag: List[Dict],
        alpha_base: float,
        beta_base: float,
        stage1_cfg: Dict,
        stage2_cfg: Dict,
    ) -> Tuple[float, float, float]:
        """
        Apply Stage1 + Stage2 corrections to derive corrected Beta parameters and mean.
        Returns (alpha_corr, beta_corr, mean_corr).
        """

        tape = self._build_trade_tape(fills_diag)
        if not tape:
            mean = alpha_base / (alpha_base + beta_base)
            return alpha_base, beta_base, mean

        y1, n1, _ = apply_behavioral_weights(tape, stage1_cfg)
        y2, n2 = apply_linear_offsets(y1, n1, stage2_cfg)

        regimes = stage2_cfg.get("regimes") or [{"pi": 1.0, "g_plus_scale": 0.0, "g_minus_scale": 0.0}]
        summary = summarize_stage1_stats(tape, y2, n2, stage1_cfg)
        mix = mixture_posterior_params(alpha_base, beta_base, summary, regimes, y2, n2)
        mean_corr = float(mix.get("mixture_mean", 0.5))
        var_corr = float(mix.get("mixture_var", 0.0))
        alpha_corr, beta_corr = self._moment_match_beta(mean_corr, var_corr)
        return alpha_corr, beta_corr, mean_corr

    def _build_ml_prior(self) -> PrincipledMLPrior:
        ml_cfg = self.cfg.get("ml_prior", {})
        return PrincipledMLPrior(
            MLPriorConfig(
                model_type=ml_cfg.get("model_type", "ensemble_logistic"),
                n_models=int(ml_cfg.get("n_models", 10)),
                N_eff_min=float(ml_cfg.get("N_eff_min", 2)),
                N_eff_max=float(ml_cfg.get("N_eff_max", 500)),
                l2=float(ml_cfg.get("regularization", {}).get("l2", 1.0)),
                lookback_events=int(ml_cfg.get("features", {}).get("lookback_events", 500)),
            ),
            logger=self.log,
        )

    def _init_mixture(self, alpha_ml: float, beta_ml: float) -> HybridMixturePrior:
        w = float(self.cfg.get("hybrid_prior", {}).get("mixture_weight_w", 0.6))
        return HybridMixturePrior(weight=w, logger=self.log).initialize(alpha_ml, beta_ml)

    def _window_slices(self, events: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        windows_cfg = self.cfg.get("windows", {})
        hist_years = int(windows_cfg.get("historical_years", self.sim_cfg.historical_years))
        rolling_days = int(windows_cfg.get("rolling_window_days", self.sim_cfg.rolling_days))
        rolling_samples = int(windows_cfg.get("rolling_sample_count", self.sim_cfg.rolling_samples))

        ts_max = int(events["ts_event"].max())
        ns_year = int(365 * 24 * 3600 * 1e9)
        ns_day = int(24 * 3600 * 1e9)

        windows: List[Tuple[str, pd.DataFrame]] = []

        hist_start = ts_max - hist_years * ns_year
        hist_df = events[events["ts_event"] >= hist_start]
        windows.append(("historical", hist_df))

        rng = np.random.default_rng(0)
        for i in range(rolling_samples):
            start_candidates = events["ts_event"].iloc[:-1].values
            if len(start_candidates) == 0:
                break
            start_ts = int(rng.choice(start_candidates))
            end_ts = start_ts + rolling_days * ns_day
            roll_df = events[(events["ts_event"] >= start_ts) & (events["ts_event"] < end_ts)]
            if roll_df.empty:
                continue
            windows.append((f"rolling_{i}", roll_df))

        return windows

    def run(self):
        run_start = time.time()
        self.log.info(
            "Simulation run start (mode=%s, out_dir=%s, live_plot=%s)",
            self.sim_cfg.mode,
            self.sim_cfg.out_dir,
            self.sim_cfg.live_plot,
        )
        # 1) Generate synthetic MBO data
        price_scale = float(self.cfg.get("simulation", {}).get("price_scale", 1.0))
        synth_cfg = self.cfg.get("synthetic", {})
        days_total = float(self.sim_cfg.historical_years * 365)
        stage_cfg = self.cfg.get("corrections", {})
        stage1_cfg = stage_cfg.get("stage1", {})
        stage2_cfg = stage_cfg.get("stage2", {})
        vis_cfg = self.cfg.get("visualization", {})
        training_fraction = float(self.cfg.get("training", {}).get("historical_fraction", 0.7))
        kappa = float(vis_cfg.get("kappa", 0.10))
        gen_t0 = time.time()
        self.log.info(
            "Synthetic MBO generation start (days=%.2f, avg_events_per_day=%.1f, max_events=%s)",
            days_total,
            float(synth_cfg.get("avg_events_per_day", 200.0)),
            int(synth_cfg.get("max_events", 5e5)) if "max_events" in synth_cfg else 500000,
        )
        events_int = generate_synth_mbo(
            instrument_id=int(synth_cfg.get("instrument_id", 1)),
            start_ts_event_ns=int(synth_cfg.get("start_ts_event_ns", 0)),
            avg_events_per_day=float(synth_cfg.get("avg_events_per_day", 200.0)),
            initial_mid=float(synth_cfg.get("initial_mid", 100.0)),
            tick_size=float(synth_cfg.get("tick_size", 0.01)),
            order_id_start=int(synth_cfg.get("order_id_start", 1000)),
            price_scale=price_scale,
            days=days_total,
            rng=np.random.default_rng(123),
            logger=self.log,
            max_events=int(5e5),
        )
        self.log.info(
            "Synthetic MBO generation end: events=%d, elapsed=%.2fs",
            len(events_int),
            time.time() - gen_t0,
        )

        # Convert for downstream processing (float prices) while preserving int price.
        events = events_int.copy()
        events["price_raw"] = events["price"]
        events["price"] = events["price"].astype(np.float64) * price_scale

        # 2) ML prior
        ml_t0 = time.time()
        self.log.info("ML prior construction start")
        ml_prior = self._build_ml_prior()
        alpha_ml, beta_ml, ml_diag = ml_prior.predict_beta(events)
        self.log.info(
            "ML prior construction end (alpha_ml=%.4f, beta_ml=%.4f, elapsed=%.2fs)",
            alpha_ml,
            beta_ml,
            time.time() - ml_t0,
        )

        # 3) Hybrid mixture
        mixture_state = self._init_mixture(alpha_ml, beta_ml)
        mixture = HybridMixturePrior(
            weight=float(self.cfg.get("hybrid_prior", {}).get("mixture_weight_w", 0.6)),
            logger=self.log,
        )

        def _mixture_mean_var(state):
            mean_ml = state.alpha_ml / (state.alpha_ml + state.beta_ml)
            var_ml = (state.alpha_ml * state.beta_ml) / ((state.alpha_ml + state.beta_ml) ** 2 * (state.alpha_ml + state.beta_ml + 1.0))
            mean_flat = state.alpha_flat / (state.alpha_flat + state.beta_flat)
            var_flat = (state.alpha_flat * state.beta_flat) / ((state.alpha_flat + state.beta_flat) ** 2 * (state.alpha_flat + state.beta_flat + 1.0))
            mean_mix = state.weight * mean_ml + (1.0 - state.weight) * mean_flat
            second = state.weight * (var_ml + mean_ml**2) + (1.0 - state.weight) * (var_flat + mean_flat**2)
            var_mix = second - mean_mix**2
            return mean_mix, var_mix

        mixture_mean0, mixture_var0 = _mixture_mean_var(mixture_state)
        alpha_ref, beta_ref = self._moment_match_beta(mixture_mean0, mixture_var0)
        p_ref = mixture_mean0

        alpha_seq: List[float] = []
        beta_seq: List[float] = []
        corrected_mean_seq: List[float] = []
        capopm_price_seq: List[float] = []

        posterior_records: List[Dict] = []
        draw_records: List[np.ndarray] = []
        weight_series: List[float] = []
        posterior_mean_series: List[float] = []
        mid_series: List[float] = []
        regime_series: List[int] = []
        trade_log_rows: List[Dict] = []
        mcmc_events: List[pd.DataFrame] = []
        mcmc_state = {
            "ts_event": int(events_int["ts_event"].max()) if not events_int.empty else 0,
            "order_id": int(events_int["order_id"].max()) if not events_int.empty else 0,
            "mid": float(events["price"].iloc[-1]),
            "regime": 0,
        }

        # 4) Windows
        windows = self._window_slices(events)
        rng = np.random.default_rng(42)
        mcmc_cfg = self.cfg.get("mcmc", {})
        regimes, transition = load_regime_config(self.sim_cfg.regimes_path)
        sampler = EventTimeMCMCSampler(
            regimes=regimes,
            transition_matrix=transition,
            n_chains=int(mcmc_cfg.get("n_chains", self.sim_cfg.n_chains)),
            warmup=int(mcmc_cfg.get("warmup", 500)),
            draws=int(mcmc_cfg.get("draws", 1000)),
            max_events_per_chain=int(mcmc_cfg.get("max_events_per_chain", 20000)),
            logger=self.log,
        )

        current_state = mixture_state
        all_chains: List[pd.DataFrame] = []
        self.log.info("Posterior window loop start: %d windows", len(windows))
        for win_id, win_df in windows:
            win_t0 = time.time()
            self.log.info("Window %s start (len=%d)", win_id, len(win_df))
            y, n, diag = map_l3_to_counts(win_df)
            prev_state = current_state
            current_state = mixture.update(current_state, y=y, n=n)
            posterior_mean = current_state.posterior_mean()
            mid_seq = diag.get("mid_series", [])
            segment_start = len(mid_series)
            mid_series.extend(mid_seq)
            regime_series.extend([0] * len(mid_seq))  # placeholder until MCMC append
            posterior_mean_series.extend([posterior_mean] * max(1, len(mid_seq)))
            weight_series.extend([current_state.weight] * max(1, len(mid_seq)))
            trade_log_rows.extend(
                [
                    {
                        "window": win_id,
                        **fill,
                    }
                    for fill in diag.get("fills", [])
                ]
            )

            draws = current_state.sample(draws=200, rng=rng)
            draw_records.append(draws)
            posterior_records.append(
                {
                    "window": win_id,
                    "start_ts": int(win_df["ts_event"].min()),
                    "end_ts": int(win_df["ts_event"].max()),
                    "mode": "historical" if win_id == "historical" else "rolling",
                    "y": y,
                    "n": n,
                    "posterior_mean": posterior_mean,
                    "weight": current_state.weight,
                    "alpha_ml": current_state.alpha_ml,
                    "beta_ml": current_state.beta_ml,
                    "alpha_flat": current_state.alpha_flat,
                    "beta_flat": current_state.beta_flat,
                }
            )

            # MCMC extrapolation seeded by latest mid price
            init_mid = mid_series[-1] if mid_series else float(win_df["price"].iloc[-1])
            init_mid = float(init_mid) if np.isfinite(init_mid) else mcmc_state["mid"]
            start_ts = mcmc_state["ts_event"] + 1
            start_oid = mcmc_state["order_id"] + 1
            mcmc_t0 = time.time()
            self.log.info(
                "Window %s: starting MCMC (init_mid=%.5f, chains=%d)",
                win_id,
                init_mid,
                int(mcmc_cfg.get("n_chains", self.sim_cfg.n_chains)),
            )
            chains, combined_df, end_state = sampler.run(
                init_regime=mcmc_state["regime"],
                init_mid=init_mid,
                start_ts_event_ns=start_ts,
                start_order_id=start_oid,
                instrument_id=int(synth_cfg.get("instrument_id", 1)),
                price_scale=price_scale,
                tick_size=float(synth_cfg.get("tick_size", 0.01)),
            )
            self.log.info(
                "Window %s: finished MCMC (chains=%d, total_mid=%d, elapsed=%.2fs)",
                win_id,
                len(chains),
                sum(len(cdf) for cdf in chains),
                time.time() - mcmc_t0,
            )
            for cdf in chains:
                mid_series.extend(cdf["mid"].tolist())
                regime_series.extend(cdf["regime"].tolist())
                posterior_mean_series.extend([posterior_mean] * len(cdf))
                weight_series.extend([current_state.weight] * len(cdf))
            all_chains.extend(chains)
            if not combined_df.empty:
                mcmc_events.append(combined_df)
            mcmc_state = {
                "ts_event": int(end_state["ts_event"]),
                "order_id": int(end_state["order_id"]),
                "mid": float(end_state["mid"]),
                "regime": int(end_state["regime"]),
            }
            segment_len = len(mid_series) - segment_start

            alpha_corr, beta_corr, mean_corr = self._apply_stage_corrections(
                diag.get("fills", []),
                alpha_base=prev_state.alpha_ml,
                beta_base=prev_state.beta_ml,
                stage1_cfg=stage1_cfg,
                stage2_cfg=stage2_cfg,
            )
            if segment_len > 0:
                alpha_seq.extend([alpha_corr] * segment_len)
                beta_seq.extend([beta_corr] * segment_len)
                corrected_mean_seq.extend([mean_corr] * segment_len)
            self.log.info("Window %s end (elapsed=%.2fs)", win_id, time.time() - win_t0)

        # Align series lengths for visualization safety.
        min_len = min(len(mid_series), len(posterior_mean_series), len(weight_series))
        if min_len == 0:
            # Fallback to raw prices if no mid prices were captured.
            mid_series = events["price"].tolist()
            posterior_mean_series = [posterior_mean_series[-1] if posterior_mean_series else 0.5] * len(mid_series)
            weight_series = [weight_series[-1] if weight_series else 0.5] * len(mid_series)
            regime_series = [0] * len(mid_series)
        else:
            mid_series = mid_series[:min_len]
            posterior_mean_series = posterior_mean_series[:min_len]
            weight_series = weight_series[:min_len]
            regime_series = regime_series[:min_len] if regime_series else [0] * min_len

        total_len = len(mid_series)
        if total_len == 0:
            self.log.warning("No mid-price series to visualize; aborting animation steps.")
            return
        if len(alpha_seq) < total_len:
            alpha_seq.extend([alpha_seq[-1] if alpha_seq else alpha_ref] * (total_len - len(alpha_seq)))
        if len(beta_seq) < total_len:
            beta_seq.extend([beta_seq[-1] if beta_seq else beta_ref] * (total_len - len(beta_seq)))
        if len(corrected_mean_seq) < total_len:
            corrected_mean_seq.extend([p_ref] * (total_len - len(corrected_mean_seq)))

        cutoff_idx = int(training_fraction * total_len)
        cutoff_idx = max(0, min(total_len, cutoff_idx))
        if cutoff_idx > 0:
            alpha_seq[:cutoff_idx] = [alpha_ref] * cutoff_idx
            beta_seq[:cutoff_idx] = [beta_ref] * cutoff_idx
            corrected_mean_seq[:cutoff_idx] = [p_ref] * cutoff_idx

        capopm_price_seq = []
        for mid_val, p_t in zip(mid_series, corrected_mean_seq):
            price = float(mid_val) * (1.0 + kappa * (p_t - p_ref))
            if price <= 0.0:
                price = max(price, 1e-6)
            capopm_price_seq.append(price)

        ts_series = np.linspace(float(events_int["ts_event"].min()), float(events_int["ts_event"].max()), total_len)
        ts_days = (ts_series - ts_series[0]) / (24 * 3600 * 1e9)

        # 5) Diagnostics
        diag_t0 = time.time()
        self.log.info("Diagnostics start (chains=%d)", len(all_chains))
        diag_tables = []
        # Use mid from chains grouped by chain for rhat/ess
        chain_mids = [cdf["mid"].to_numpy() for cdf in all_chains] if all_chains else []
        diag_tables.append(summarize_metric("mid", chain_mids))
        diag_tables.append(summarize_metric("posterior_mean", [np.array(posterior_mean_series)]))

        diag_df = pd.concat(diag_tables, ignore_index=True)
        diag_path = Path(self.sim_cfg.out_dir, "mcmc_diagnostics", "diagnostics.csv")
        diag_df.to_csv(diag_path, index=False)
        self._save_chain_plots(all_chains, Path(self.sim_cfg.out_dir, "mcmc_diagnostics"))
        self.log.info("Diagnostics end (elapsed=%.2fs)", time.time() - diag_t0)

        # 6) Persist outputs
        combined_stream = events_int.copy()
        combined_stream["source"] = "synthetic_base"
        if mcmc_events:
            mcmc_concat = pd.concat(mcmc_events, ignore_index=True)
            mcmc_concat["source"] = "mcmc_extrapolation"
            combined_stream = pd.concat([combined_stream, mcmc_concat], ignore_index=True)
        combined_stream = combined_stream.sort_values("ts_event", kind="mergesort").reset_index(drop=True)
        csv_path = Path(self.sim_cfg.out_dir, "synthetic_mbo.csv")
        csv_t0 = time.time()
        self.log.info("Starting CSV write to %s (rows=%d)", csv_path, len(combined_stream))
        combined_stream.to_csv(csv_path, index=False)
        self.log.info("Finished CSV write to %s (elapsed=%.2fs)", csv_path, time.time() - csv_t0)

        write_parquet = bool(self.cfg.get("logging", {}).get("write_parquet", False))
        if write_parquet:
            pq_path = Path(self.sim_cfg.out_dir, "synthetic_mbo.parquet")
            pq_t0 = time.time()
            self.log.info("Starting parquet write to %s", pq_path)
            try:
                combined_stream.to_parquet(pq_path, index=False)
                self.log.info("Finished parquet write to %s (elapsed=%.2fs)", pq_path, time.time() - pq_t0)
            except Exception as exc:  # pragma: no cover - parquet optional
                self.log.warning("Parquet export failed: %s", exc)
        else:
            self.log.info("Parquet export skipped (write_parquet=False)")

        posterior_df = pd.DataFrame.from_records(posterior_records)
        posterior_df.to_csv(Path(self.sim_cfg.out_dir, "posterior_windows.csv"), index=False)

        np.savez(
            Path(self.sim_cfg.out_dir, "posterior_draws.npz"),
            draws=np.array(draw_records, dtype=object),
            weights=np.array(weight_series),
            posterior_means=np.array(posterior_mean_series),
        )

        pd.DataFrame.from_records(trade_log_rows).to_csv(
            Path(self.sim_cfg.out_dir, "trade_update_log.csv"), index=False
        )

        manifest = {
            "config": self.sim_cfg.cfg_path,
            "regimes": self.sim_cfg.regimes_path,
            "synthetic": synth_cfg,
            "symbol": self.sim_cfg.symbol,
            "mode": self.sim_cfg.mode,
            "mixture_weight": float(self.cfg.get("hybrid_prior", {}).get("mixture_weight_w", 0.6)),
            "ml_prior": ml_diag,
        }
        with open(Path(self.sim_cfg.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        dual_anim_path = Path(self.sim_cfg.out_dir, "plots", vis_cfg.get("out_mp4_name", "capopm_dual_viz.mp4"))
        self.log.info(
            "Dual-panel animation start (len=%d, cutoff_idx=%d, save_path=%s)",
            total_len,
            cutoff_idx,
            dual_anim_path,
        )
        make_dual_panel_animation(
            ts=ts_days,
            mid_prices=mid_series,
            alpha_seq=alpha_seq,
            beta_seq=beta_seq,
            capopm_price=capopm_price_seq,
            cutoff_index=cutoff_idx,
            save_path=str(dual_anim_path),
            fps=int(vis_cfg.get("fps", 30)),
            max_frames=int(vis_cfg.get("max_frames", 900)),
            grid_points=int(vis_cfg.get("posterior_grid_points", 120)),
            rolling_window=int(vis_cfg.get("posterior_rolling_window", 80)),
            camera_elev=float(vis_cfg.get("camera_elev", 25)),
            camera_azim=float(vis_cfg.get("camera_azim", -60)),
            logger=self.log,
        )

        anim_path = Path(self.sim_cfg.out_dir, "plots", "live_simulation.mp4")
        live_flag = bool(self.sim_cfg.live_plot)
        anim_t0 = time.time()
        self.log.info(
            "Animation generation start (events=%d, live=%s, save_path=%s)",
            len(mid_series),
            live_flag,
            anim_path,
        )
        make_live_animation(
            mid_prices=mid_series,
            regimes=regime_series,
            posterior_means=posterior_mean_series,
            weights=weight_series,
            save_path=str(anim_path),
            fps=int(self.cfg.get("visualization", {}).get("fps", 30)),
            live=live_flag,
            logger=self.log,
        )
        self.log.info("Animation generation finished (elapsed=%.2fs)", time.time() - anim_t0)

        # Static summary plots
        self._save_static_plots(mid_series, posterior_mean_series, weight_series, Path(self.sim_cfg.out_dir, "plots"))

        self.log.info(
            "Simulation complete. Outputs stored under %s (total_elapsed=%.2fs)",
            self.sim_cfg.out_dir,
            time.time() - run_start,
        )

    def _save_static_plots(
        self,
        mids: List[float],
        post_means: List[float],
        weights: List[float],
        plots_dir: Path,
    ):
        import matplotlib.pyplot as plt

        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(mids, color="steelblue")
        axes[0].set_ylabel("Mid price")
        axes[1].plot(post_means, color="darkorange")
        axes[1].set_ylabel("Posterior mean")
        axes[1].set_ylim(0, 1)
        axes[2].plot(weights, color="seagreen")
        axes[2].set_ylabel("Mixture weight")
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel("Event index")
        fig.tight_layout()
        path = plots_dir / "static_diagnostics.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        self.log.info("Saved static diagnostics to %s", path)

    def _save_chain_plots(self, chains: List[pd.DataFrame], diag_dir: Path):
        import matplotlib.pyplot as plt
        import numpy as np

        if not chains:
            return

        diag_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        for cdf in chains:
            ax.plot(cdf["mid"].values, alpha=0.7, lw=1)
        ax.set_title("MCMC mid traces")
        ax.set_xlabel("Event index")
        ax.set_ylabel("Mid")
        fig.tight_layout()
        fig.savefig(diag_dir / "trace_mid.png", dpi=120)
        plt.close(fig)

        # Autocorrelation for first chain
        chain0 = chains[0]["mid"].values
        if len(chain0) > 1:
            lags = min(50, len(chain0) - 1)
            ac = [np.corrcoef(chain0[:-k], chain0[k:])[0, 1] for k in range(1, lags)]
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(range(1, lags), ac, width=0.8, color="steelblue")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorr")
            ax.set_title("Autocorrelation (chain 0)")
            fig.tight_layout()
            fig.savefig(diag_dir / "autocorr_mid.png", dpi=120)
            plt.close(fig)

            # Rank histogram across chains
            combined = np.concatenate([cdf["mid"].values for cdf in chains])
            ranks = np.searchsorted(np.sort(combined), chain0)
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(ranks, bins=20, color="darkorange", alpha=0.8)
            ax.set_title("Rank histogram (chain 0 vs combined)")
            ax.set_xlabel("Rank")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            fig.savefig(diag_dir / "rank_hist_mid.png", dpi=120)
            plt.close(fig)
