"""Synthetic Databento-style MBO generator with regime-switching price dynamics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_FIELDS = [
    "ts_event",
    "ts_recv",
    "instrument_id",
    "action",
    "side",
    "price",
    "size",
    "order_id",
]

NS_PER_DAY = int(24 * 3600 * 1e9)


@dataclass(frozen=True)
class PriceRegime:
    name: str
    mu: float
    sigma: float
    jump_prob: float
    jump_sigma: float
    mean_revert: float
    event_mult: float
    buy_bias: float


def _default_regimes() -> List[PriceRegime]:
    return [
        PriceRegime("low_vol", mu=0.0000, sigma=0.007, jump_prob=0.004, jump_sigma=0.018, mean_revert=0.08, event_mult=0.8, buy_bias=0.00),
        PriceRegime("high_vol", mu=0.0000, sigma=0.020, jump_prob=0.018, jump_sigma=0.050, mean_revert=0.03, event_mult=1.6, buy_bias=0.00),
        PriceRegime("trend_up", mu=0.0015, sigma=0.012, jump_prob=0.010, jump_sigma=0.030, mean_revert=0.01, event_mult=1.2, buy_bias=0.08),
        PriceRegime("mean_revert", mu=0.0000, sigma=0.010, jump_prob=0.007, jump_sigma=0.020, mean_revert=0.22, event_mult=1.0, buy_bias=-0.03),
    ]


def _default_transition() -> np.ndarray:
    return np.array(
        [
            [0.87, 0.06, 0.04, 0.03],
            [0.16, 0.70, 0.08, 0.06],
            [0.09, 0.08, 0.74, 0.09],
            [0.15, 0.06, 0.08, 0.71],
        ],
        dtype=np.float64,
    )


def _load_process_regimes(process_cfg: Dict | None) -> Tuple[List[PriceRegime], np.ndarray]:
    if not process_cfg:
        return _default_regimes(), _default_transition()

    raw_regimes = process_cfg.get("regimes", [])
    if not raw_regimes:
        return _default_regimes(), _default_transition()

    regimes: List[PriceRegime] = []
    for item in raw_regimes:
        regimes.append(
            PriceRegime(
                name=str(item["name"]),
                mu=float(item.get("mu", 0.0)),
                sigma=float(item.get("sigma", 0.01)),
                jump_prob=float(item.get("jump_prob", 0.0)),
                jump_sigma=float(item.get("jump_sigma", 0.0)),
                mean_revert=float(item.get("mean_revert", 0.0)),
                event_mult=float(item.get("event_mult", 1.0)),
                buy_bias=float(item.get("buy_bias", 0.0)),
            )
        )

    matrix = np.asarray(process_cfg.get("transition_matrix", _default_transition()), dtype=np.float64)
    if matrix.shape != (len(regimes), len(regimes)):
        matrix = _default_transition()
        if matrix.shape[0] != len(regimes):
            matrix = np.eye(len(regimes), dtype=np.float64)
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
    matrix = np.maximum(matrix, 1e-12)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return regimes, matrix


def _quantize_price(price_float: float, tick_size: float, price_scale: float) -> Tuple[int, float]:
    snapped = max(tick_size, tick_size * round(price_float / tick_size))
    price_int = int(round(snapped / price_scale))
    return price_int, snapped


def _simulate_daily_path(
    total_days: int,
    initial_mid: float,
    rng: np.random.Generator,
    regimes: Sequence[PriceRegime],
    transition: np.ndarray,
) -> pd.DataFrame:
    total_days = max(2, int(total_days))
    k_reg = len(regimes)
    regime_idx = np.zeros(total_days, dtype=np.int32)
    mid = np.zeros(total_days, dtype=np.float64)
    open_mid = np.zeros(total_days, dtype=np.float64)
    drift_signal = np.zeros(total_days, dtype=np.float64)
    sigma_eff = np.zeros(total_days, dtype=np.float64)

    log_s = float(np.log(max(initial_mid, 1e-6)))
    log_anchor = log_s
    vol_state = 0.0
    trend_state = 0.0
    regime = 0

    for d in range(total_days):
        spec = regimes[regime]
        eps = float(rng.normal())
        vol_state = 0.90 * vol_state + 0.25 * abs(eps)
        local_sigma = max(1e-6, spec.sigma * (0.65 + 0.65 * vol_state))
        jump = float(rng.normal(0.0, spec.jump_sigma)) if rng.random() < spec.jump_prob else 0.0
        trend_state = 0.88 * trend_state + 0.12 * eps
        log_anchor = 0.98 * log_anchor + 0.02 * log_s
        revert_term = -spec.mean_revert * (log_s - log_anchor)
        dlog = spec.mu + 0.003 * trend_state + local_sigma * eps + jump + revert_term
        log_s = float(log_s + dlog)

        open_mid[d] = np.exp(log_s - dlog)
        mid[d] = np.exp(log_s)
        drift_signal[d] = dlog
        sigma_eff[d] = local_sigma
        regime_idx[d] = regime
        probs = transition[regime]
        regime = int(rng.choice(k_reg, p=probs))

    ret = np.diff(np.log(mid), prepend=np.log(mid[0]))
    rv = pd.Series(ret).rolling(window=10, min_periods=2).std().fillna(0.0).to_numpy()
    return pd.DataFrame(
        {
            "sim_day": np.arange(total_days, dtype=np.int32),
            "regime_id": regime_idx,
            "mid_open": open_mid,
            "mid_close": mid,
            "dlog_mid": drift_signal,
            "sigma_eff": sigma_eff,
            "realized_vol_10d": rv,
            "daily_return": ret,
        }
    )


def generate_synth_mbo(
    instrument_id: int,
    start_ts_event_ns: int,
    avg_events_per_day: float,
    initial_mid: float,
    tick_size: float,
    order_id_start: int,
    price_scale: float,
    days: float,
    rng: np.random.Generator,
    logger: logging.Logger,
    max_events: int | None = None,
    process_cfg: Dict | None = None,
    return_daily: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic MBO and an internal regime-switching daily path."""

    log = logger
    regimes, transition = _load_process_regimes(process_cfg)
    total_days = max(2, int(round(days)))
    daily = _simulate_daily_path(total_days=total_days, initial_mid=initial_mid, rng=rng, regimes=regimes, transition=transition)

    max_events = int(max_events) if max_events is not None else int(max(avg_events_per_day * total_days * 2, 20000))
    records: List[Dict] = []
    active_order_ids: List[int] = []
    order_id = int(order_id_start)
    mid_live = float(initial_mid)
    action_choices = np.array(["A", "M", "C", "F"], dtype=object)

    for row in daily.itertuples(index=False):
        day = int(row.sim_day)
        regime_id = int(row.regime_id)
        day_mid = float(row.mid_close)
        spec = regimes[regime_id]
        event_rate = max(5.0, avg_events_per_day * spec.event_mult)
        n_events = max(5, int(rng.poisson(event_rate)))
        if len(records) + n_events > max_events:
            n_events = max_events - len(records)
            if n_events <= 0:
                break

        day_start = int(start_ts_event_ns + day * NS_PER_DAY)
        intra_ns = np.sort(rng.integers(0, NS_PER_DAY, size=n_events, endpoint=False))
        fill_intensity = min(0.60, 0.14 + 6.5 * spec.sigma + 0.08 * abs(spec.buy_bias))
        add_intensity = max(0.20, 0.48 - 2.2 * spec.sigma)
        cancel_intensity = max(0.12, 0.23 - 0.6 * spec.sigma)
        modify_intensity = max(0.08, 1.0 - (fill_intensity + add_intensity + cancel_intensity))
        probs = np.array([add_intensity, modify_intensity, cancel_intensity, fill_intensity], dtype=np.float64)
        probs = probs / probs.sum()

        for ns_offset in intra_ns:
            if len(records) >= max_events:
                break
            ts_event = day_start + int(ns_offset)
            ts_recv = ts_event + int(rng.integers(0, 2_000_000))
            mid_live += 0.16 * (day_mid - mid_live) + float(rng.normal(0.0, tick_size * (1.0 + 4.0 * spec.sigma)))
            buy_prob = float(np.clip(0.5 + spec.buy_bias + 0.20 * np.tanh(row.dlog_mid * 200.0), 0.05, 0.95))
            side_buy = rng.random() < buy_prob
            side = "B" if side_buy else "S"
            action = str(rng.choice(action_choices, p=probs))

            if action in {"M", "C", "F"} and not active_order_ids:
                action = "A"

            if action == "A":
                order_id += 1
                oid = order_id
                active_order_ids.append(oid)
                level_ticks = int(rng.integers(1, 5))
                level_dir = -1 if side_buy else 1
                px = mid_live + level_dir * level_ticks * tick_size
            else:
                idx = int(rng.integers(0, len(active_order_ids)))
                oid = int(active_order_ids[idx])
                if action == "C":
                    active_order_ids.pop(idx)
                if action == "F":
                    impact = float(rng.normal(0.0, tick_size * (2.5 + 10.0 * spec.sigma)))
                    mid_live += impact if side_buy else -impact
                    px = mid_live + (0.5 if side_buy else -0.5) * tick_size
                    if rng.random() < 0.35 and active_order_ids:
                        active_order_ids.pop(idx)
                else:
                    px = mid_live + float(rng.normal(0.0, tick_size * 2.0))

            size = int(max(1, round(rng.lognormal(mean=1.5 + 0.7 * spec.sigma, sigma=0.65))))
            price_int, price_float = _quantize_price(px, tick_size=tick_size, price_scale=price_scale)

            records.append(
                {
                    "ts_event": int(ts_event),
                    "ts_recv": int(ts_recv),
                    "instrument_id": int(instrument_id),
                    "action": action,
                    "side": side,
                    "price": int(price_int),
                    "size": int(size),
                    "order_id": int(oid),
                    "price_float": float(price_float),
                    "sim_day": day,
                    "regime_id": regime_id,
                    "mid_target": float(day_mid),
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        df = pd.DataFrame(columns=REQUIRED_FIELDS + ["price_float", "sim_day", "regime_id", "mid_target"])
    df = df.sort_values("ts_event", kind="mergesort").reset_index(drop=True)
    daily_events = df.groupby("sim_day", as_index=False).size().rename(columns={"size": "events"})
    daily = daily.merge(daily_events, on="sim_day", how="left")
    daily["events"] = daily["events"].fillna(0).astype(int)

    log.info(
        "Synthetic MBO generated: events=%d days=%d regimes=%d mid_range=[%.2f, %.2f]",
        len(df),
        int(daily["sim_day"].max()) + 1 if not daily.empty else 0,
        len(regimes),
        float(daily["mid_close"].min()) if not daily.empty else float("nan"),
        float(daily["mid_close"].max()) if not daily.empty else float("nan"),
    )
    out_df = df[REQUIRED_FIELDS + ["price_float", "sim_day", "regime_id", "mid_target"]]
    if return_daily:
        return out_df, daily
    return out_df

