"""
kalshi_api_fetch.py
====================

This module provides a command-line utility to collect microstructure data
from the Kalshi prediction market exchange.  It pulls **order book** and
**trade‑level** information for every market in a specified series over a
specified date range.  The data is saved locally as two CSV files:

* ``orderbook_data.csv`` – one row per order book snapshot, including
  timestamp, market ticker, and lists of price/size levels for the YES and
  NO sides of the book.
* ``trades_data.csv`` – one row per executed trade, including timestamp,
  price, quantity, yes/no price decompositions, and the taker side.

The Kalshi API requires authenticated requests using RSA‑PSS signatures.  The
script reads a **key ID** and **private RSA key** from files specified on the
command line, constructs the required signature, and attaches the
authentication headers to every request.  All API calls are rate limited and
resilient to transient failures.

Usage example::

    python kalshi_api_fetch.py \
        --key-id-path /path/to/kalshi_keyID.txt \
        --private-key-path /path/to/kalshi_RSA_privateKey.txt \
        --series KXINXY \
        --start-date 2022-01-01 \
        --orderbook-out orderbook_data.csv \
        --trades-out trades_data.csv

When run, the script will:

1. Load the credentials from the provided files.
2. Query ``/markets`` to obtain all markets in the given series that were
   created or expired on or after the ``--start-date``.  Both **settled** and
   **open** markets are collected so that historical and current buckets are
   included.
3. For each market, fetch the current order book via ``/markets/{ticker}/orderbook``.
   The full book depth is requested (default depth of 0 returns all levels).
4. For each market, fetch the full trade history via ``/markets/trades`` with
   ``ticker`` and ``min_ts`` parameters.  Trades prior to the start date are
   ignored.  The response is paginated and all pages are fetched.
5. Serialize the order books and trades into CSV files.  The order book
   serializer flattens the nested lists of prices and sizes into JSON strings
   stored in a single column; trades are stored one trade per line.

Notes
-----
* This is a one‑off fetch script intended for offline analysis.  To run
  regularly or in production, consider adding exponential backoff, more
  sophisticated error handling, and timestamp persistence.
* The script assumes that the Kalshi API server is reachable at
  ``https://api.elections.kalshi.com/trade-api/v2`` as used in the official
  documentation.  Should your account use a different base URL (e.g., the
  general ``api.kalshi.com`` domain), adjust the ``BASE_URL`` constant
  accordingly.
* Parameter constraints for the structural model are documented in the
  accompanying paper.  In particular, the tempered fractional Heston model
  requires ``gamma > 0``, ``theta > 0``, ``sigma > 0``, ``alpha ∈ (1/2,1)``,
  ``lambda >= 0``, and ``V0 > 0``【205128327162733†L404-L407】.  These
  constraints will be enforced in the separate CAPOPM analysis script.

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key


###############################################################################
# Authentication helpers
###############################################################################

def load_credentials(key_id_path: Path, private_key_path: Path) -> Tuple[str, Any]:
    """Load the Kalshi key ID and RSA private key.

    Parameters
    ----------
    key_id_path : Path
        Path to the file containing the Kalshi key ID (the first line will
        be stripped and used as the key ID).  This file should contain a
        single string.
    private_key_path : Path
        Path to the file containing the RSA private key in PEM format.

    Returns
    -------
    Tuple[str, Any]
        A tuple of the key ID and the loaded private key object.
    """
    key_id = key_id_path.read_text().strip()
    private_key_bytes = private_key_path.read_bytes()
    # The PEM may be encrypted with a passphrase; we do not support passphrases
    private_key = load_pem_private_key(private_key_bytes, password=None)
    return key_id, private_key


def sign_request(private_key: Any, timestamp: str, method: str, path: str) -> str:
    """Sign an API request using RSA‑PSS.

    The signature format is defined in the Kalshi documentation: the message
    to sign is constructed by concatenating the timestamp, the HTTP method,
    and the path (excluding query parameters).  The resulting signature is
    base64 encoded and returned as a string.

    Parameters
    ----------
    private_key : Any
        An RSA private key object as returned by ``load_pem_private_key``.
    timestamp : str
        The timestamp in milliseconds as a string.
    method : str
        HTTP method (e.g., 'GET', 'POST').  Case insensitive.
    path : str
        The path component of the URL (e.g., '/markets').  Do not include
        query parameters.

    Returns
    -------
    str
        Base64 encoded signature string.
    """
    message = (timestamp + method.upper() + path).encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    import base64

    return base64.b64encode(signature).decode("utf-8")


def authenticated_get(
    session: requests.Session,
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]],
    key_id: str,
    private_key: Any,
) -> Dict[str, Any]:
    """Perform an authenticated GET request to the Kalshi API.

    This helper handles signing, header construction, sending the request,
    and basic error handling.  It will retry transient errors with a
    short exponential backoff.

    Parameters
    ----------
    session : requests.Session
        An existing session for connection pooling.
    base_url : str
        Base URL for the API (e.g. 'https://api.elections.kalshi.com/trade-api/v2').
    path : str
        Path component (e.g. '/markets').  Should start with a slash.
    params : dict or None
        Query parameters to include in the URL.
    key_id : str
        API key ID.
    private_key : Any
        Loaded RSA private key.

    Returns
    -------
    dict
        Parsed JSON response.

    Raises
    ------
    RuntimeError
        If the request ultimately fails after retries.
    """
    url = base_url + path
    backoff = 1.0
    max_attempts = 5
    for attempt in range(max_attempts):
        timestamp_ms = str(int(time.time() * 1000))
        signature = sign_request(private_key, timestamp_ms, "GET", path)
        headers = {
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }
        try:
            resp = session.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as exc:
            # connection error, retry
            print(f"Request exception {exc}, attempt {attempt+1}/{max_attempts}", file=sys.stderr)
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 429:
            # rate limited, wait and retry
            retry_after = float(resp.headers.get("Retry-After", 1))
            print(f"Rate limited. Waiting {retry_after} seconds", file=sys.stderr)
            time.sleep(retry_after)
            continue

        if not resp.ok:
            # 4xx or 5xx
            err_msg = f"Error {resp.status_code} for {path}: {resp.text}"
            if resp.status_code >= 500 and attempt < max_attempts - 1:
                # server error, retry
                print(err_msg, file=sys.stderr)
                time.sleep(backoff)
                backoff *= 2
                continue
            raise RuntimeError(err_msg)

        return resp.json()

    raise RuntimeError(f"Failed to get {path} after {max_attempts} attempts")


###############################################################################
# Data fetching
###############################################################################

def list_markets(
    session: requests.Session,
    base_url: str,
    key_id: str,
    private_key: Any,
    series: str,
    start_date: dt.datetime,
) -> List[Dict[str, Any]]:
    """List all markets for a given series created or expiring on/after start_date.

    The ``/markets`` endpoint returns a paginated list of markets.  This
    function collects both settled and open markets and filters them by
    comparing their creation time and expiration time to ``start_date``.

    Parameters
    ----------
    session : requests.Session
        Session for HTTP requests.
    base_url : str
        API base URL.
    key_id : str
        Kalshi key ID.
    private_key : Any
        RSA private key.
    series : str
        Series ticker, e.g. 'KXINXY'.
    start_date : datetime.datetime
        Only markets created or expiring on or after this date will be
        returned.

    Returns
    -------
    List[dict]
        List of market objects.
    """
    path = "/markets"
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        params: Dict[str, Any] = {
            "series_ticker": series,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        # We fetch both settled and open markets by leaving status unspecified.
        data = authenticated_get(session, base_url, path, params, key_id, private_key)
        page_markets = data.get("markets", [])
        cursor = data.get("cursor")
        for m in page_markets:
            # parse created_time and expiration_time
            created = parse_iso8601(m.get("created_time"))
            expiration = parse_iso8601(m.get("expiration_time"))
            # include market if created or expires on/after start_date
            if created and created >= start_date or (expiration and expiration >= start_date):
                markets.append(m)
        if not cursor:
            break
        # courtesy pause to avoid rate limits
        time.sleep(0.1)
    return markets


def parse_iso8601(ts: Optional[str]) -> Optional[dt.datetime]:
    """Parse ISO8601 timestamp into timezone‑aware datetime.  Returns None if input is None."""
    if ts is None:
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def fetch_orderbook(
    session: requests.Session,
    base_url: str,
    key_id: str,
    private_key: Any,
    ticker: str,
) -> Dict[str, Any]:
    """Fetch the current order book for a given market ticker.

    Returns a dict containing YES and NO sides of the book.  Raises
    ``RuntimeError`` on failure.
    """
    path = f"/markets/{ticker}/orderbook"
    params = {"depth": 0}  # 0 means return all levels
    return authenticated_get(session, base_url, path, params, key_id, private_key)


def fetch_trades(
    session: requests.Session,
    base_url: str,
    key_id: str,
    private_key: Any,
    ticker: str,
    min_ts: Optional[int],
) -> List[Dict[str, Any]]:
    """Fetch all trades for a given market since ``min_ts``.

    Parameters
    ----------
    session : requests.Session
        HTTP session.
    base_url : str
        API base URL.
    key_id : str
        Kalshi key ID.
    private_key : Any
        RSA private key.
    ticker : str
        Market ticker.
    min_ts : int or None
        Unix timestamp in seconds.  Trades created before this time are
        discarded.

    Returns
    -------
    List[dict]
        A list of trade objects.
    """
    path = "/markets/trades"
    trades: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        params: Dict[str, Any] = {
            "ticker": ticker,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = int(min_ts)
        data = authenticated_get(session, base_url, path, params, key_id, private_key)
        page_trades = data.get("trades", [])
        cursor = data.get("cursor")
        trades.extend(page_trades)
        if not cursor:
            break
        time.sleep(0.1)
    return trades


###############################################################################
# Serialization
###############################################################################

def serialize_orderbook_row(market: Dict[str, Any], orderbook: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize orderbook snapshot into a flat dictionary.

    The order book API returns lists of prices and sizes for the YES and NO
    sides.  Here we store these lists as JSON strings.  Timestamps are
    included for reproducibility.

    Parameters
    ----------
    market : dict
        The market object (from /markets listing) containing metadata such
        as ticker and series.
    orderbook : dict
        The order book JSON returned by the API.

    Returns
    -------
    dict
        Flattened order book row.
    """
    row = {
        "series_ticker": market.get("series_ticker"),
        "event_ticker": market.get("event_ticker"),
        "market_ticker": market.get("ticker"),
        "created_time": market.get("created_time"),
        "expiration_time": market.get("expiration_time"),
        "snapshot_time": dt.datetime.utcnow().isoformat() + "Z",
    }
    # The API returns orderbook.yes and orderbook.no as lists of [price, size]
    # pairs.  We'll store the lists of prices and sizes separately.
    yes_side = orderbook.get("orderbook", {}).get("yes", [])
    no_side = orderbook.get("orderbook", {}).get("no", [])
    # Split into price and size lists for each side
    yes_prices = [level[0] for level in yes_side]
    yes_sizes = [level[1] for level in yes_side]
    no_prices = [level[0] for level in no_side]
    no_sizes = [level[1] for level in no_side]
    row.update({
        "yes_prices": json.dumps(yes_prices),
        "yes_sizes": json.dumps(yes_sizes),
        "no_prices": json.dumps(no_prices),
        "no_sizes": json.dumps(no_sizes),
    })
    # The book also returns aggregated yes/no volume in dollars and contracts
    row.update({
        "yes_dollars": orderbook.get("yes_dollars"),
        "no_dollars": orderbook.get("no_dollars"),
        "yes_contracts": orderbook.get("yes_contracts"),
        "no_contracts": orderbook.get("no_contracts"),
    })
    return row


def serialize_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a single trade into a flat dictionary.

    Parameters
    ----------
    trade : dict
        Trade object as returned by the API.

    Returns
    -------
    dict
        Flattened trade row.
    """
    row = {
        "trade_id": trade.get("trade_id"),
        "ticker": trade.get("ticker"),
        "count": trade.get("count"),
        "price_cents": trade.get("price"),
        "yes_price_cents": trade.get("yes_price"),
        "no_price_cents": trade.get("no_price"),
        "yes_price_dollars": trade.get("yes_price_dollars"),
        "no_price_dollars": trade.get("no_price_dollars"),
        "taker_side": trade.get("taker_side"),
        "created_time": trade.get("created_time"),
    }
    return row


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], headers: List[str]) -> None:
    """Write rows of dictionaries to a CSV file with given headers."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


###############################################################################
# Main routine
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch orderbook and trade data from Kalshi")
    parser.add_argument("--key-id-path", type=Path, required=True,
                        help="Path to the file containing the Kalshi key ID")
    parser.add_argument("--private-key-path", type=Path, required=True,
                        help="Path to the file containing the RSA private key")
    parser.add_argument("--series", type=str, required=True,
                        help="Series ticker to fetch (e.g., KXINXY)")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Fetch markets created or expiring on/after this date (YYYY-MM-DD)")
    parser.add_argument("--orderbook-out", type=Path, default=Path("orderbook_data.csv"),
                        help="Output CSV file for orderbook snapshots")
    parser.add_argument("--trades-out", type=Path, default=Path("trades_data.csv"),
                        help="Output CSV file for trade records")
    parser.add_argument("--base-url", type=str,
                        default="https://api.elections.kalshi.com/trade-api/v2",
                        help="Base URL for the Kalshi API")
    args = parser.parse_args()

    # Parse start date
    try:
        start_date = dt.datetime.fromisoformat(args.start_date)
    except ValueError:
        print(f"Invalid start date format: {args.start_date}. Use YYYY-MM-DD.", file=sys.stderr)
        sys.exit(1)

    key_id, private_key = load_credentials(args.key_id_path, args.private_key_path)
    session = requests.Session()

    print(f"Listing markets for series {args.series} since {start_date.date()}…", file=sys.stderr)
    markets = list_markets(session, args.base_url, key_id, private_key, args.series, start_date)
    print(f"Found {len(markets)} markets", file=sys.stderr)

    # Prepare output containers
    orderbook_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []

    # Compute min_ts once; trades before this will be filtered out
    start_ts = int(start_date.timestamp())

    for idx, m in enumerate(markets, start=1):
        ticker = m.get("ticker")
        print(f"[{idx}/{len(markets)}] Processing {ticker}", file=sys.stderr)
        # Fetch orderbook (will reflect current state; historical data not available)
        try:
            ob_json = fetch_orderbook(session, args.base_url, key_id, private_key, ticker)
            orderbook_rows.append(serialize_orderbook_row(m, ob_json))
        except Exception as exc:
            print(f"Failed to fetch orderbook for {ticker}: {exc}", file=sys.stderr)
        # Fetch trades
        try:
            trades = fetch_trades(session, args.base_url, key_id, private_key, ticker, start_ts)
            for tr in trades:
                trade_rows.append(serialize_trade(tr))
        except Exception as exc:
            print(f"Failed to fetch trades for {ticker}: {exc}", file=sys.stderr)
        # Pause between markets to be polite
        time.sleep(0.1)

    # Write outputs
    if orderbook_rows:
        # Determine headers from the first row
        ob_headers = list(orderbook_rows[0].keys())
        write_csv(args.orderbook_out, orderbook_rows, ob_headers)
        print(f"Wrote {len(orderbook_rows)} orderbook snapshots to {args.orderbook_out}", file=sys.stderr)
    else:
        print("No orderbook snapshots collected", file=sys.stderr)

    if trade_rows:
        tr_headers = list(trade_rows[0].keys())
        write_csv(args.trades_out, trade_rows, tr_headers)
        print(f"Wrote {len(trade_rows)} trades to {args.trades_out}", file=sys.stderr)
    else:
        print("No trades collected", file=sys.stderr)


if __name__ == "__main__":
    main()