#!/usr/bin/env python3
"""
kalshi_capopm_step2_3_split_train_test.py

End-to-end script for CAPOPM Steps 2 & 3 on Kalshi data, specialized for:

Series: KXINXY
Test market (still OPEN): KXINXY-25DEC31-B6900

- TRAIN data: all SETTLED markets in this series (KXINXY), excluding the target
              market ticker (it won't be settled yet anyway, but we explicitly
              exclude it just in case).
- TEST data : the single OPEN market with ticker KXINXY-25DEC31-B6900.

Outputs:
    kalshi_capopm_train.csv  -> for ML prior training (historical settled)
    kalshi_capopm_test.csv   -> for CAPOPM vs Kalshi testing on the live market

Requirements:
    pip install requests pandas python-dateutil

Run:
    python kalshi_capopm_step2_3_split_train_test.py
"""

import time
from typing import Dict, List, Optional, Any

import requests
import pandas as pd
from dateutil import parser as dateparser

# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

SERIES_TICKER: Optional[str] = "KXINXY"
TARGET_MARKET_TICKER: str = "KXINXY-25DEC31-B6900"

CATEGORY_FILTER: Optional[str] = None  # e.g. "Indices", or None

STATUS_TRAIN: str = "settled"  # historical, closed markets
STATUS_TEST: str = "open"      # current live target market

LIMIT_PER_PAGE: int = 1000
MAX_MARKETS_TRAIN: Optional[int] = None  # None = all settled in this series

MIN_TRADES: int = 1  # require at least this many trades to keep a market

TRAIN_OUTPUT_CSV = "kalshi_capopm_events.csv"
TEST_OUTPUT_CSV = "kalshi_capopm_test.csv"


# =====================================================================
# LOW-LEVEL HTTP WRAPPERS
# =====================================================================

def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Thin wrapper around GET with basic error handling."""
    resp = requests.get(url, params=params, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"GET {url} failed: {resp.status_code} {resp.text}")
    return resp.json()


# =====================================================================
# FETCH MARKETS + TRADES
# =====================================================================

def fetch_markets_for_series(
    series_ticker: Optional[str],
    status: str,
    category_filter: Optional[str] = None,
    max_markets: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Pull markets for a given series and status using GET /markets with pagination.
    Used for TRAIN (settled markets).
    """
    url = f"{BASE_URL}/markets"
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {
            "status": status,
            "limit": LIMIT_PER_PAGE,
        }
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker

        data = _get(url, params)
        page_markets = data.get("markets", [])
        cursor = data.get("cursor") or None

        # Optional category filter
        if category_filter:
            page_markets = [
                m for m in page_markets
                if (m.get("category") == category_filter)
            ]

        markets.extend(page_markets)

        if max_markets is not None and len(markets) >= max_markets:
            markets = markets[:max_markets]
            break

        if not cursor or len(page_markets) == 0:
            break

        time.sleep(0.2)

    return markets


def fetch_single_market_by_ticker(
    ticker: str,
) -> List[Dict[str, Any]]:
    """
    Fetch a single market by EXACT ticker using GET /markets/{ticker}.

    Returns a list with one market dict if found, or [] if not found.
    """
    url = f"{BASE_URL}/markets/{ticker}"
    data = _get(url, params={})  # no query params for path-style endpoint

    # The API might return either:
    #   {"market": {...}}  OR just {...}
    if isinstance(data, dict):
        if "market" in data and isinstance(data["market"], dict):
            market = data["market"]
        else:
            market = data
        # sanity check ticker
        if market.get("ticker") == ticker:
            return [market]
        else:
            # ticker mismatch → treat as not found
            return []
    else:
        # unexpected structure
        return []




def fetch_all_trades_for_ticker(
    ticker: str,
    min_ts: Optional[int] = None,
    max_ts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch ALL trades for a given market ticker via GET /markets/trades with pagination.
    """
    url = f"{BASE_URL}/markets/trades"
    trades: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    while True:
        params: Dict[str, Any] = {
            "limit": LIMIT_PER_PAGE,
            "ticker": ticker,
        }
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts

        data = _get(url, params)
        page_trades = data.get("trades", [])
        cursor = data.get("cursor") or None

        trades.extend(page_trades)

        if not cursor or len(page_trades) == 0:
            break

        time.sleep(0.2)

    return trades


# =====================================================================
# MAP INTO CAPOPM EVENT OBJECTS
# =====================================================================

def parse_iso_to_unix(iso_str: Optional[str]) -> Optional[int]:
    """Convert Kalshi ISO8601 timestamps to Unix seconds, or None."""
    if not iso_str:
        return None
    try:
        dt = dateparser.isoparse(iso_str)
        return int(dt.timestamp())
    except Exception:
        return None


def aggregate_trades_to_tickets(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert raw trades into YES/NO "ticket" counts.
    """
    yes_tickets = 0
    no_tickets = 0
    yes_notional = 0
    no_notional = 0

    for tr in trades:
        side = tr.get("taker_side")
        count = tr.get("count", 0) or 0
        price = tr.get("price", 0) or 0  # cents

        if side == "yes":
            yes_tickets += count
            yes_notional += count * price
        elif side == "no":
            no_tickets += count
            no_notional += count * price

    total_tickets = yes_tickets + no_tickets
    total_notional = yes_notional + no_notional

    return {
        "yes_tickets": yes_tickets,
        "no_tickets": no_tickets,
        "total_tickets": total_tickets,
        "yes_notional_cents": yes_notional,
        "no_notional_cents": no_notional,
        "total_notional_cents": total_notional,
    }


def market_to_capopm_row(market: Dict[str, Any],
                         trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a single Kalshi market + its trades into one CAPOPM event row.
    """
    ticket_stats = aggregate_trades_to_tickets(trades)

    event_ticker = market.get("event_ticker")
    market_ticker = market.get("ticker")
    series_ticker = market.get("series")
    category = market.get("category")

    expiration_ts = parse_iso_to_unix(market.get("expiration_time"))
    settlement_ts = parse_iso_to_unix(market.get("latest_expiration_time"))
    created_ts = parse_iso_to_unix(market.get("created_time"))

    result = market.get("result")
    realized_outcome = 1 if result == "yes" else 0 if result == "no" else None

    last_price_dollars_str = market.get("last_price_dollars")
    if last_price_dollars_str is not None:
        try:
            last_price_dollars = float(last_price_dollars_str)
        except ValueError:
            last_price_dollars = None
    else:
        last_price_cents = market.get("last_price")
        last_price_dollars = (
            last_price_cents / 100.0 if isinstance(last_price_cents, (int, float)) else None
        )

    kalshi_implied_yes_prob = last_price_dollars  # [0,1] if non-null

    strike_type = market.get("strike_type")
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")
    functional_strike = market.get("functional_strike")
    rules_primary = market.get("rules_primary")
    rules_secondary = market.get("rules_secondary")

    row = {
        "series_ticker": 'KXINXY',
        "event_ticker": event_ticker,
        "market_ticker": market_ticker,
        "category": category,
        "market_title": market.get("title"),
        "market_subtitle": market.get("subtitle"),
        "yes_sub_title": market.get("yes_sub_title"),
        "no_sub_title": market.get("no_sub_title"),

        "created_ts": created_ts,
        "expiration_ts": expiration_ts,
        "settlement_ts": settlement_ts,

        "result": result,
        "realized_outcome": realized_outcome,

        "last_price_dollars": last_price_dollars,
        "kalshi_implied_yes_prob": kalshi_implied_yes_prob,
        "volume_contracts": market.get("volume"),
        "open_interest": market.get("open_interest"),
        "notional_value_cents": market.get("notional_value"),

        "strike_type": strike_type,
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "functional_strike": functional_strike,
        "rules_primary": rules_primary,
        "rules_secondary": rules_secondary,

        "yes_tickets": ticket_stats["yes_tickets"],
        "no_tickets": ticket_stats["no_tickets"],
        "total_tickets": ticket_stats["total_tickets"],
        "yes_notional_cents": ticket_stats["yes_notional_cents"],
        "no_notional_cents": ticket_stats["no_notional_cents"],
        "total_notional_cents": ticket_stats["total_notional_cents"],
    }

    return row


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=== CAPOPM x Kalshi: Steps 2 & 3 (Train/Test split) ===")
    print(f"Series ticker       : {SERIES_TICKER}")
    print(f"Train status        : {STATUS_TRAIN}")
    print(f"Test status         : {STATUS_TEST}")
    print(f"Target market ticker: {TARGET_MARKET_TICKER}")
    print()

    # -----------------------------
    # TRAIN: settled markets
    # -----------------------------
    print("Fetching SETTLED markets for training...")
    markets_train = fetch_markets_for_series(
        series_ticker=SERIES_TICKER,
        status=STATUS_TRAIN,
        category_filter=CATEGORY_FILTER,
        max_markets=MAX_MARKETS_TRAIN,
    )
    print(f"Fetched {len(markets_train)} settled markets (raw).")

    train_rows: List[Dict[str, Any]] = []

    for idx, m in enumerate(markets_train, start=1):
        ticker = m.get("ticker")
        print(f"[TRAIN {idx}/{len(markets_train)}] Ticker={ticker} ...", end="", flush=True)

        # Explicitly exclude the target ticker from training even if status ever mismatches
        if ticker == TARGET_MARKET_TICKER:
            print(" SKIPPED (target market, reserved for test).")
            continue

        try:
            trades = fetch_all_trades_for_ticker(ticker=ticker)
        except Exception as e:
            print(f" FAILED (trades error: {e})")
            continue

        if len(trades) < MIN_TRADES:
            print(f" SKIPPED (only {len(trades)} trades; minimum is {MIN_TRADES})")
            continue

        row = market_to_capopm_row(market=m, trades=trades)
        train_rows.append(row)
        print(f" OK (trades={len(trades)}, yes_tickets={row['yes_tickets']}, "
              f"no_tickets={row['no_tickets']})")

        time.sleep(0.1)

    df_train = pd.DataFrame(train_rows)
    # Drop rows with missing outcome: training requires realized_outcome
    before = len(df_train)
    df_train = df_train[df_train["realized_outcome"].notna()].copy()
    after = len(df_train)
    print(f"\nTRAIN: dropped {before - after} rows with missing realized_outcome; {after} remain.")

    # -----------------------------
    # TEST: the single OPEN target market
    # -----------------------------
    print("\nFetching OPEN target market for testing...")
    markets_test = fetch_single_market_by_ticker(TARGET_MARKET_TICKER)

    if not markets_test:
        print(
            f"ERROR: No market found with ticker {TARGET_MARKET_TICKER} via /markets/{{ticker}}.\n"
            "This usually means:\n"
            "  - The market has not been listed yet, or\n"
            "  - The ticker string is slightly different from what Kalshi uses.\n"
            "No TEST CSV will be usable until this exact ticker exists."
        )
        df_test = pd.DataFrame()
    else:
        print(f"Fetched {len(markets_test)} market(s) for target ticker.")
        test_rows: List[Dict[str, Any]] = []

        for m in markets_test:
            ticker = m.get("ticker")
            print(f"[TEST] Ticker={ticker} ...", end="", flush=True)

            try:
                trades = fetch_all_trades_for_ticker(ticker=ticker)
            except Exception as e:
                print(f" FAILED (trades error: {e})")
                continue

            if len(trades) < MIN_TRADES:
                print(f" SKIPPED (only {len(trades)} trades; minimum is {MIN_TRADES})")
                continue

            row = market_to_capopm_row(market=m, trades=trades)
            test_rows.append(row)
            print(f" OK (trades={len(trades)}, yes_tickets={row['yes_tickets']}, "
                  f"no_tickets={row['no_tickets']})")

            time.sleep(0.1)

        df_test = pd.DataFrame(test_rows)


    # IMPORTANT: do NOT drop rows with missing realized_outcome for TEST,
    # because this market is still open and we are predicting ex ante.

    # -----------------------------
    # Save outputs
    # -----------------------------
    df_train.to_csv(TRAIN_OUTPUT_CSV, index=False)
    if not df_test.empty:
        df_test.to_csv(TEST_OUTPUT_CSV, index=False)
    else:
        # write an empty CSV as a marker if you want
        df_test.to_csv(TEST_OUTPUT_CSV, index=False)

    print(f"\nSaved TRAIN dataset to: {TRAIN_OUTPUT_CSV} (rows={len(df_train)})")
    print(f"Saved TEST dataset  to: {TEST_OUTPUT_CSV} (rows={len(df_test)})")


    if not df_train.empty:
        print("\nTRAIN columns:")
        print(", ".join(df_train.columns))
    if not df_test.empty:
        print("\nTEST columns:")
        print(", ".join(df_test.columns))


if __name__ == "__main__":
    main()
