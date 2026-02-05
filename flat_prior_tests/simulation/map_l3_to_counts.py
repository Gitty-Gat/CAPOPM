"""
Mapping from Databento L3 events to (y, n) counts for Beta-Binomial updates.

Definition of YES-leaning actions:
- Aggressive BUY fills at or above the prevailing mid price.
- Any fill whose mid-price impact is non-negative (mid_after >= mid_before),
  treating upward moves as increasing the implied probability proxy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12


def map_l3_to_counts(events_window: pd.DataFrame) -> Tuple[int, int, Dict]:
    """
    Convert an event window into (y, n) plus diagnostics.

    Parameters
    ----------
    events_window : pd.DataFrame
        Must be event-time sorted and contain Databento MBO fields.
    """

    bids: Dict[int, Dict[str, float]] = {}
    asks: Dict[int, Dict[str, float]] = {}
    diagnostics: List[Dict] = []
    mid_series: List[float] = []

    y = 0
    n = 0

    def best_bid_ask():
        bid = max((v["price"] for v in bids.values()), default=np.nan)
        ask = min((v["price"] for v in asks.values()), default=np.nan)
        return bid, ask

    for idx, row in events_window.iterrows():
        action = row["action"]
        side = row["side"]
        oid = int(row["order_id"])
        price = float(row["price"])
        size = float(row["size"])

        bid, ask = best_bid_ask()
        mid_before = 0.5 * (bid + ask) if np.isfinite(bid) and np.isfinite(ask) else np.nan

        book = bids if side == "B" else asks
        if action == "A":
            book[oid] = {"price": price, "size": size}
        elif action == "M":
            if oid in book:
                book[oid] = {"price": price, "size": size}
        elif action == "C":
            book.pop(oid, None)
        elif action == "F":
            if oid in book:
                prev = book[oid]
                remaining = max(prev["size"] - size, 0.0)
                if remaining <= EPS:
                    book.pop(oid, None)
                else:
                    book[oid]["size"] = remaining

        # Mid after applying the event
        bid, ask = best_bid_ask()
        mid_after = 0.5 * (bid + ask) if np.isfinite(bid) and np.isfinite(ask) else np.nan
        if np.isfinite(mid_after):
            mid_series.append(mid_after)

        if action == "F":
            n += 1
            yes_rationale = []
            is_yes = False

            if side == "B" and np.isfinite(mid_before) and price >= mid_before - EPS:
                is_yes = True
                yes_rationale.append("aggressive_buy_at_or_above_mid")

            if np.isfinite(mid_before) and np.isfinite(mid_after) and mid_after >= mid_before - EPS:
                is_yes = True
                yes_rationale.append("non_negative_mid_impact")

            if is_yes:
                y += 1

            diagnostics.append(
                {
                    "index": int(idx),
                    "side": side,
                    "price": price,
                    "size": size,
                    "mid_before": mid_before,
                    "mid_after": mid_after,
                    "yes": bool(is_yes),
                    "rationale": yes_rationale,
                }
            )

    diag = {
        "fills": diagnostics,
        "mid_series": mid_series,
        "y": y,
        "n": n,
    }
    return y, n, diag

