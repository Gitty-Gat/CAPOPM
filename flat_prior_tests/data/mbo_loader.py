"""
Databento L3 (MBO) ingestion utilities.

Assumptions:
- Input is a CSV or Parquet file containing the required Databento MBO fields.
- Price is stored in fixed-point units and normalized to float using `price_scale`.
- Events are strictly ordered by `ts_event` (nanoseconds since epoch) for event-time processing.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

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

OPTIONAL_FIELDS = ["flags", "channel_id", "publisher_id"]


def load_mbo_events(
    path: str,
    price_scale: float,
    required_fields: Optional[Iterable[str]] = None,
    optional_fields: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load and normalize Databento MBO data from CSV or Parquet.

    Parameters
    ----------
    path : str
        Path to CSV or Parquet file.
    price_scale : float
        Multiplier to convert fixed-point integer prices to floats.
    required_fields : iterable, optional
        Override of required fields; defaults to the Databento schema.
    optional_fields : iterable, optional
        Optional fields to include if present.

    Returns
    -------
    pd.DataFrame
        Event-time ordered dataframe with float `price` and preserved `price_raw`.
    """

    log = logger or logging.getLogger(__name__)
    req = list(required_fields) if required_fields is not None else REQUIRED_FIELDS
    opt = list(optional_fields) if optional_fields is not None else OPTIONAL_FIELDS

    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    missing: List[str] = [f for f in req if f not in df.columns]
    if missing:
        raise ValueError(f"MBO file missing required fields: {missing}")

    # Preserve raw fixed-point price and normalize.
    df = df.copy()
    df["price_raw"] = df["price"]
    df["price"] = df["price"].astype(np.float64) * float(price_scale)

    # Ensure categorical fields are uppercase for consistency.
    df["action"] = df["action"].astype(str).str.upper()
    df["side"] = df["side"].astype(str).str.upper()

    # Drop any columns not in required/optional set to keep outputs auditable.
    allowed = set(req) | set(opt) | {"price_raw"}
    extraneous = [c for c in df.columns if c not in allowed]
    if extraneous:
        log.debug("Dropping extraneous columns: %s", extraneous)
        df = df[[c for c in df.columns if c in allowed]]

    # Sort strictly by event time, breaking ties by receive time then order_id.
    df = df.sort_values(["ts_event", "ts_recv", "order_id"], kind="mergesort").reset_index(drop=True)

    log.info(
        "Loaded %d MBO events from %s (price_scale=%s). Columns: %s",
        len(df),
        path,
        price_scale,
        list(df.columns),
    )
    return df

