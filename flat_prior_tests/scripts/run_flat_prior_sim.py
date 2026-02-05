#!/usr/bin/env python
"""
CLI entrypoint for flat prior simulation.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from flat_prior_tests.simulation.simulator import FlatPriorSimulationRunner, SimulationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flat prior simulation (synthetic MBO event-time).")
    parser.add_argument("--symbol", default="A", help="Symbol (default: A).")
    parser.add_argument("--years", type=int, default=2, help="Historical horizon in years.")
    parser.add_argument("--window_days", type=int, default=7, help="Rolling window size in days.")
    parser.add_argument("--n_chains", type=int, default=10, help="Number of MCMC chains.")
    parser.add_argument(
        "--mode",
        default="synthetic",
        choices=["synthetic"],
        help="Simulation mode. Only 'synthetic' is supported in this path.",
    )
    parser.add_argument(
        "--config",
        default="flat_prior_tests/config/default_sim.yaml",
        help="Simulation config YAML.",
    )
    parser.add_argument(
        "--regimes",
        default="flat_prior_tests/config/regimes.yaml",
        help="Regime configuration YAML.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: flat_prior_tests/results/<timestamp>).",
    )
    parser.add_argument(
        "--live_plot",
        default="false",
        help="Whether to show live plot (true/false).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out
    if out_dir is None:
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = f"flat_prior_tests/results/run_{stamp}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sim_cfg = SimulationConfig(
        cfg_path=args.config,
        regimes_path=args.regimes,
        out_dir=out_dir,
        symbol=args.symbol,
        historical_years=args.years,
        rolling_days=args.window_days,
        n_chains=args.n_chains,
        live_plot=str(args.live_plot).lower() == "true",
        mode=args.mode,
    )

    runner = FlatPriorSimulationRunner(sim_cfg)
    runner.run()


if __name__ == "__main__":
    main()
