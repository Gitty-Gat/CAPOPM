import json
import yaml

from flat_prior_tests.simulation.simulator import FlatPriorSimulationRunner, SimulationConfig


def test_simulator_small_run(tmp_path):
    # Synthetic-only config with short horizon for speed.
    cfg = {
        "simulation": {"price_scale": 0.01, "event_time": True},
        "windows": {"historical_years": 0.01, "rolling_window_days": 1, "rolling_sample_count": 1},
        "ml_prior": {
            "model_type": "ensemble_logistic",
            "n_models": 3,
            "N_eff_min": 1,
            "N_eff_max": 5,
            "features": {"lookback_events": 50},
        },
        "hybrid_prior": {"mixture_weight_w": 0.5},
        "mcmc": {"n_chains": 1, "warmup": 1, "draws": 3, "max_events_per_chain": 5},
        "visualization": {"live": False, "save_animation": False, "fps": 5},
        "logging": {"level": "ERROR"},
        "synthetic": {
            "instrument_id": 1,
            "start_ts_event_ns": 0,
            "avg_events_per_day": 10,
            "initial_mid": 10.0,
            "tick_size": 0.01,
            "order_id_start": 1,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Regime config
    regimes = {
        "regimes": [
            {
                "name": "r0",
                "lambda": 0.5,
                "p_buy": 0.5,
                "size_lognormal": {"mean": 0.0, "sigma": 0.1},
                "price_impact": {"mean": 0.0, "sigma": 0.01},
            }
        ],
        "transitions": {"matrix": [[1.0]]},
    }
    regime_path = tmp_path / "regimes.yaml"
    with open(regime_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(regimes, f)

    out_dir = tmp_path / "out"
    sim_cfg = SimulationConfig(
        cfg_path=str(cfg_path),
        regimes_path=str(regime_path),
        out_dir=str(out_dir),
        symbol="A",
        historical_years=1,
        rolling_days=1,
        n_chains=1,
        live_plot=False,
        mode="synthetic",
    )

    runner = FlatPriorSimulationRunner(sim_cfg)
    runner.run()

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "synthetic_mbo.csv").exists()
    with open(out_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "synthetic" in manifest
    cols = (out_dir / "synthetic_mbo.csv").read_text().splitlines()[0].split(",")
    for required in ["ts_event", "ts_recv", "instrument_id", "action", "side", "price", "size", "order_id"]:
        assert required in cols
