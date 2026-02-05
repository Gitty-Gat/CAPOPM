"""
Synthetic Databento MBO (L3) generator for event-time simulations.

Generates a stream of order-book events with required fields:
ts_event, ts_recv, instrument_id, action, side, price (fixed-point int),
size, order_id.

Prices are derived from an internal mid price and quantized using
price_scale and tick_size so the exported `price` field is an int64.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

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
) -> pd.DataFrame:
    """
    Generate a synthetic MBO stream.

    Parameters
    ----------
    instrument_id : int
    start_ts_event_ns : int
        Starting event timestamp (ns). Monotone increasing thereafter.
    avg_events_per_day : float
        Average number of events per day (Poisson).
    initial_mid : float
        Starting mid price (float).
    tick_size : float
        Minimum price increment in float units.
    order_id_start : int
        Starting order id counter.
    price_scale : float
        Fixed-point scale (float price * (1/price_scale) -> int).
    days : float
        Total duration to simulate in days.
    max_events : int, optional
        Cap on total events to maintain memory safety.
    """

    log = logger
    seconds_per_day = 24 * 3600.0
    mean_dt = seconds_per_day / max(avg_events_per_day, 1e-6)
    total_events = int(avg_events_per_day * days)
    if max_events is not None:
        total_events = min(total_events, max_events)
    total_events = max(total_events, 10)

    ts_event = int(start_ts_event_ns)
    mid = float(initial_mid)
    order_id = int(order_id_start)

    bids: Dict[int, Dict[str, float]] = {}
    asks: Dict[int, Dict[str, float]] = {}

    records = []

    def quantize_price(price_float: float) -> Tuple[int, float]:
        """Convert float price to fixed-point int respecting tick size."""
        snapped = tick_size * round(price_float / tick_size)
        price_int = int(round(snapped / price_scale))
        return price_int, snapped

    for _ in range(total_events):
        dt = rng.exponential(mean_dt)
        ts_event += int(dt * 1e9)
        ts_recv = ts_event  # event-time only

        side_buy = rng.random() < 0.5
        side = "B" if side_buy else "S"

        # Choose an action with modest probability for fills.
        action = rng.choice(["A", "M", "C", "F"], p=[0.45, 0.10, 0.15, 0.30])

        size = float(max(1.0, rng.lognormal(mean=0.0, sigma=0.5)))

        # Helper to get top-of-book
        best_bid = max((v["price_float"] for v in bids.values()), default=np.nan)
        best_ask = min((v["price_float"] for v in asks.values()), default=np.nan)
        mid_book = (
            0.5 * (best_bid + best_ask)
            if np.isfinite(best_bid) and np.isfinite(best_ask)
            else mid
        )

        price_float = mid_book
        if action == "A":
            # Place near top of book.
            direction = -1.0 if side_buy else 1.0
            price_float = mid_book + direction * tick_size * rng.integers(1, 3)
            price_int, price_float = quantize_price(price_float)
            order_id += 1
            book = bids if side_buy else asks
            book[order_id] = {"price_int": price_int, "price_float": price_float, "size": size}
        elif action == "M":
            book = bids if side_buy else asks
            if book:
                oid = rng.choice(list(book.keys()))
                # Small price nudge and size change.
                delta_ticks = rng.integers(-1, 2)
                new_price = book[oid]["price_float"] + delta_ticks * tick_size
                price_int, price_float = quantize_price(new_price)
                new_size = max(1.0, book[oid]["size"] + rng.normal(0, 0.5))
                book[oid] = {"price_int": price_int, "price_float": price_float, "size": new_size}
                order_id = oid
            else:
                action = "A"
                continue
        elif action == "C":
            book = bids if side_buy else asks
            if book:
                oid = rng.choice(list(book.keys()))
                price_int = book[oid]["price_int"]
                price_float = book[oid]["price_float"]
                size = book[oid]["size"]
                book.pop(oid, None)
                order_id = oid
            else:
                action = "A"
                continue
        elif action == "F":
            # Trade against top of book; if empty, synthesize around mid.
            price_float = (
                best_ask if side_buy and np.isfinite(best_ask) else best_bid if not side_buy and np.isfinite(best_bid) else mid_book
            )
            impact = rng.normal(loc=0.0, scale=tick_size * 2.0)
            price_float += impact if side_buy else -impact
            price_int, price_float = quantize_price(price_float)
            mid = price_float  # simple mid update
            book = bids if side_buy else asks
            if book:
                oid = rng.choice(list(book.keys()))
                prev = book[oid]
                remaining = max(prev["size"] - size, 0.0)
                if remaining <= 0.0:
                    book.pop(oid, None)
                else:
                    book[oid]["size"] = remaining
                order_id = oid
            else:
                order_id += 1

        price_int, price_float = quantize_price(price_float)
        records.append(
            {
                "ts_event": int(ts_event),
                "ts_recv": int(ts_recv),
                "instrument_id": int(instrument_id),
                "action": str(action),
                "side": str(side),
                "price": int(price_int),
                "size": int(round(size)),
                "order_id": int(order_id),
                "price_float": float(price_float),
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("ts_event", kind="mergesort").reset_index(drop=True)
    log.info(
        "Synthetic MBO generated: %d events, instrument_id=%s, start_ts=%s",
        len(df),
        instrument_id,
        start_ts_event_ns,
    )
    return df[REQUIRED_FIELDS + ["price_float"]]

