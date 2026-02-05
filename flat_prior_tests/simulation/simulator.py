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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from flat_prior_tests.data.mbo_synth import generate_synth_mbo
from flat_prior_tests.diagnostics.mcmc_diagnostics import summarize_metric
from flat_prior_tests.mcmc.event_time_mcmc import EventTimeMCMCSampler, load_regime_config
from flat_prior_tests.priors.hybrid_mixture_prior import HybridMixturePrior
from flat_prior_tests.priors.ml_prior_principled import MLPriorConfig, PrincipledMLPrior
from flat_prior_tests.simulation.map_l3_to_counts import map_l3_to_counts
from flat_prior_tests.simulation.visualization import make_live_animation

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
        # 1) Generate synthetic MBO data
        price_scale = float(self.cfg.get("simulation", {}).get("price_scale", 1.0))
        synth_cfg = self.cfg.get("synthetic", {})
        days_total = float(self.sim_cfg.historical_years * 365)
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

        # Convert for downstream processing (float prices) while preserving int price.
        events = events_int.copy()
        events["price_raw"] = events["price"]
        events["price"] = events["price"].astype(np.float64) * price_scale

        # 2) ML prior
        ml_prior = self._build_ml_prior()
        alpha_ml, beta_ml, ml_diag = ml_prior.predict_beta(events)

        # 3) Hybrid mixture
        mixture_state = self._init_mixture(alpha_ml, beta_ml)
        mixture = HybridMixturePrior(
            weight=float(self.cfg.get("hybrid_prior", {}).get("mixture_weight_w", 0.6)),
            logger=self.log,
        )

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
        for win_id, win_df in windows:
            y, n, diag = map_l3_to_counts(win_df)
            current_state = mixture.update(current_state, y=y, n=n)
            posterior_mean = current_state.posterior_mean()
            mid_seq = diag.get("mid_series", [])
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
            chains, combined_df, end_state = sampler.run(
                init_regime=mcmc_state["regime"],
                init_mid=init_mid,
                start_ts_event_ns=start_ts,
                start_order_id=start_oid,
                instrument_id=int(synth_cfg.get("instrument_id", 1)),
                price_scale=price_scale,
                tick_size=float(synth_cfg.get("tick_size", 0.01)),
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

        # 5) Diagnostics
        diag_tables = []
        # Use mid from chains grouped by chain for rhat/ess
        chain_mids = [cdf["mid"].to_numpy() for cdf in all_chains] if all_chains else []
        diag_tables.append(summarize_metric("mid", chain_mids))
        diag_tables.append(summarize_metric("posterior_mean", [np.array(posterior_mean_series)]))

        diag_df = pd.concat(diag_tables, ignore_index=True)
        diag_path = Path(self.sim_cfg.out_dir, "mcmc_diagnostics", "diagnostics.csv")
        diag_df.to_csv(diag_path, index=False)
        self._save_chain_plots(all_chains, Path(self.sim_cfg.out_dir, "mcmc_diagnostics"))

        # 6) Persist outputs
        combined_stream = events_int.copy()
        combined_stream["source"] = "synthetic_base"
        if mcmc_events:
            mcmc_concat = pd.concat(mcmc_events, ignore_index=True)
            mcmc_concat["source"] = "mcmc_extrapolation"
            combined_stream = pd.concat([combined_stream, mcmc_concat], ignore_index=True)
        combined_stream = combined_stream.sort_values("ts_event", kind="mergesort").reset_index(drop=True)
        combined_stream.to_csv(Path(self.sim_cfg.out_dir, "synthetic_mbo.csv"), index=False)
        try:
            combined_stream.to_parquet(Path(self.sim_cfg.out_dir, "synthetic_mbo.parquet"), index=False)
        except Exception as exc:  # pragma: no cover - parquet optional
            self.log.warning("Parquet export failed: %s", exc)

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

        anim_path = Path(self.sim_cfg.out_dir, "plots", "live_simulation.mp4")
        make_live_animation(
            mid_prices=mid_series,
            regimes=regime_series,
            posterior_means=posterior_mean_series,
            weights=weight_series,
            save_path=str(anim_path),
            fps=int(self.cfg.get("visualization", {}).get("fps", 30)),
            live=bool(self.cfg.get("visualization", {}).get("live", False) or self.sim_cfg.live_plot),
            logger=self.log,
        )

        # Static summary plots
        self._save_static_plots(mid_series, posterior_mean_series, weight_series, Path(self.sim_cfg.out_dir, "plots"))

        self.log.info("Simulation complete. Outputs stored under %s", self.sim_cfg.out_dir)

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
