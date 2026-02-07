"""Synthetic Databento-like imbalance stream generator."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

NS_PER_DAY = int(24 * 3600 * 1e9)

IMBALANCE_FIELDS = [
    "ts_event",
    "instrument_id",
    "ref_price",
    "paired_qty",
    "total_imbalance_qty",
    "auction_status",
    "ind_match_price",
    "cont_book_clr_price",
    "auct_interest_clr_price",
    "sim_day",
    "regime_id",
]


def _to_price_int(px: float, tick_size: float, price_scale: float) -> int:
    snapped = max(tick_size, tick_size * round(px / tick_size))
    return int(round(snapped / price_scale))


def _auction_status_for_time(ns_offset: int, day_ns: int, halt_prob: float, rng: np.random.Generator) -> int:
    open_cut = int(0.03 * day_ns)
    close_cut = int(0.97 * day_ns)
    if rng.random() < halt_prob:
        return 3
    if ns_offset <= open_cut:
        return 1
    if ns_offset >= close_cut:
        return 2
    return 0


def generate_synth_imbalance(
    daily: pd.DataFrame,
    instrument_id: int,
    start_ts_event_ns: int,
    tick_size: float,
    price_scale: float,
    rng: np.random.Generator,
    logger: logging.Logger,
    cfg: Dict | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic imbalance stream aligned to the same day timeline as MBO.

    Required fields mirror the Databento imbalance schema subset:
    paired_qty, total_imbalance_qty, auction_status, ref_price, and match-price
    variants.
    """

    cfg = cfg or {}
    base_events_per_day = float(cfg.get("base_events_per_day", 18.0))
    open_spike = float(cfg.get("opening_multiplier", 2.5))
    close_spike = float(cfg.get("closing_multiplier", 2.8))
    halt_prob = float(cfg.get("halt_prob", 0.005))

    regime_mult = cfg.get(
        "regime_imbalance_scale",
        {
            0: 0.8,
            1: 2.1,
            2: 1.4,
            3: 1.0,
        },
    )

    records: List[Dict] = []
    for row in daily.itertuples(index=False):
        day = int(row.sim_day)
        regime_id = int(row.regime_id)
        mid_px = float(row.mid_close)
        sigma_eff = float(row.sigma_eff)
        ret = float(row.daily_return)
        scale = float(regime_mult.get(regime_id, 1.0))

        n_base = max(2, int(rng.poisson(base_events_per_day * scale)))
        n_open = max(1, int(rng.poisson(open_spike * scale)))
        n_close = max(1, int(rng.poisson(close_spike * scale)))

        day_start = int(start_ts_event_ns + day * NS_PER_DAY)
        offsets = []
        offsets.extend(rng.integers(0, int(0.03 * NS_PER_DAY), size=n_open))
        offsets.extend(rng.integers(int(0.03 * NS_PER_DAY), int(0.97 * NS_PER_DAY), size=n_base))
        offsets.extend(rng.integers(int(0.97 * NS_PER_DAY), NS_PER_DAY, size=n_close))
        offsets = sorted(int(v) for v in offsets)

        running_imb = 0.0
        for ns_offset in offsets:
            ts_event = day_start + ns_offset
            auction_status = _auction_status_for_time(ns_offset, NS_PER_DAY, halt_prob=halt_prob, rng=rng)

            imb_loc = 30.0 * np.tanh(45.0 * ret) + 80.0 * np.tanh(running_imb / 250.0)
            imb_scale = (90.0 + 2100.0 * sigma_eff) * scale
            if auction_status in {1, 2}:
                imb_scale *= 1.8
                imb_loc *= 1.2
            raw_imb = float(rng.normal(loc=imb_loc, scale=imb_scale))
            total_imbalance_qty = int(np.round(raw_imb))
            running_imb = 0.86 * running_imb + 0.14 * raw_imb

            paired_base = 200 + 1200 * sigma_eff + 0.35 * abs(total_imbalance_qty)
            if auction_status in {1, 2}:
                paired_base *= 1.8
            paired_qty = int(max(1, round(rng.lognormal(mean=np.log(max(2.0, paired_base)), sigma=0.45))))

            ref_noise = float(rng.normal(0.0, tick_size * (1.5 + 20.0 * sigma_eff)))
            ref_price_int = _to_price_int(mid_px + ref_noise, tick_size=tick_size, price_scale=price_scale)

            imbalance_ratio = np.clip(total_imbalance_qty / max(1.0, paired_qty), -5.0, 5.0)
            deviation_ticks = (0.6 + 2.4 * abs(imbalance_ratio)) * np.sign(imbalance_ratio)
            dev = deviation_ticks * tick_size

            ind_match_price = _to_price_int(mid_px + dev + float(rng.normal(0.0, tick_size * 0.7)), tick_size, price_scale)
            cont_book_clr_price = _to_price_int(mid_px + 0.7 * dev + float(rng.normal(0.0, tick_size * 0.5)), tick_size, price_scale)
            auct_interest_clr_price = _to_price_int(mid_px + 1.2 * dev + float(rng.normal(0.0, tick_size * 0.9)), tick_size, price_scale)

            records.append(
                {
                    "ts_event": np.uint64(ts_event),
                    "instrument_id": int(instrument_id),
                    "ref_price": int(ref_price_int),
                    "paired_qty": int(paired_qty),
                    "total_imbalance_qty": int(total_imbalance_qty),
                    "auction_status": int(auction_status),
                    "ind_match_price": int(ind_match_price),
                    "cont_book_clr_price": int(cont_book_clr_price),
                    "auct_interest_clr_price": int(auct_interest_clr_price),
                    "sim_day": int(day),
                    "regime_id": int(regime_id),
                }
            )

    out = pd.DataFrame.from_records(records, columns=IMBALANCE_FIELDS)
    out = out.sort_values("ts_event", kind="mergesort").reset_index(drop=True)
    logger.info(
        "Synthetic imbalance generated: rows=%d day_count=%d paired_qty_range=[%d, %d] imbalance_abs_max=%d",
        len(out),
        int(daily["sim_day"].max()) + 1 if not daily.empty else 0,
        int(out["paired_qty"].min()) if not out.empty else 0,
        int(out["paired_qty"].max()) if not out.empty else 0,
        int(out["total_imbalance_qty"].abs().max()) if not out.empty else 0,
    )
    return out
