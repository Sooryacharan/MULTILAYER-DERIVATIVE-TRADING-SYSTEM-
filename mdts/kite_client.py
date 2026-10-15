"""Read-only Kite Connect data access. There is deliberately NO order-placement code in this module.

UNTESTED AGAINST THE LIVE API: written from Kite's public docs and exercised only with a fake.
Verify the chunk limits and interval names below on Kite's docs before relying on them.
Daily login: Kite access tokens expire every trading day; `create_session` takes the
request_token you obtain by logging in manually each morning.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Protocol

import pandas as pd

from .datastore import BarStore
from .env import require

# Max days per historical request by interval (from docs as I recall them; VERIFY).
CHUNK_DAYS = {"minute": 60, "3minute": 100, "5minute": 100, "10minute": 100,
              "15minute": 200, "30minute": 200, "60minute": 400, "day": 2000}


class KiteLike(Protocol):
    def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False, oi=False): ...


def create_session(request_token: str):
    """Exchange the morning's request_token for an authenticated client. Needs `pip install kiteconnect`."""
    try:
        from kiteconnect import KiteConnect
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install kiteconnect to use the live Kite client") from e
    kite = KiteConnect(api_key=require("KITE_API_KEY"))
    data = kite.generate_session(request_token, api_secret=require("KITE_API_SECRET"))
    kite.set_access_token(data["access_token"])
    return kite


def fetch_history(kite: KiteLike, store: BarStore, symbol: str, token: int, interval: str,
                  start: datetime, end: datetime, sleep: float = 0.4) -> int:
    """Download [start, end] in allowed-size chunks into the store. Returns rows written."""
    if interval not in CHUNK_DAYS:
        raise ValueError(f"unsupported interval {interval!r}")
    step = timedelta(days=CHUNK_DAYS[interval])
    total, cur = 0, start
    while cur <= end:
        chunk_end = min(cur + step, end)
        rows = kite.historical_data(token, cur, chunk_end, interval)
        if rows:
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)   # keep naive IST wall-clock
            total += store.upsert(symbol, interval, df.set_index("date"))
        cur = chunk_end + timedelta(seconds=1)
        time.sleep(sleep)
    return total
