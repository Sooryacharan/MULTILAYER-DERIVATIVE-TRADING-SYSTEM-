"""SQLite bar store. Point-in-time reads: `end` bounds what a caller can ever see.

Symbols are individual contract names (e.g. a specific monthly future). Stitching contracts
into a continuous series is a separate, explicit step and is NOT done here.
Timestamps are stored as naive IST strings.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

_COLS = ["open", "high", "low", "close", "volume"]


class BarStore:
    def __init__(self, path: str | Path = "data/bars.db"):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS bars (symbol TEXT, interval TEXT, ts TEXT,"
            " open REAL, high REAL, low REAL, close REAL, volume REAL,"
            " PRIMARY KEY (symbol, interval, ts))")
        self.conn.commit()

    def upsert(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        d = df.copy()
        if "volume" not in d:
            d["volume"] = 0.0
        rows = [(symbol, interval, ts.strftime("%Y-%m-%d %H:%M:%S"), *map(float, r))
                for ts, r in zip(d.index, d[_COLS].to_numpy())]
        self.conn.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def get(self, symbol: str, interval: str, start=None, end=None) -> pd.DataFrame:
        q, args = "SELECT ts,open,high,low,close,volume FROM bars WHERE symbol=? AND interval=?", [symbol, interval]
        if start is not None:
            q += " AND ts>=?"
            args.append(pd.Timestamp(start).strftime("%Y-%m-%d %H:%M:%S"))
        if end is not None:
            q += " AND ts<=?"
            args.append(pd.Timestamp(end).strftime("%Y-%m-%d %H:%M:%S"))
        df = pd.read_sql_query(q + " ORDER BY ts", self.conn, params=args)
        df.index = pd.to_datetime(df.pop("ts"))
        return df

    def last_ts(self, symbol: str, interval: str):
        r = self.conn.execute("SELECT MAX(ts) FROM bars WHERE symbol=? AND interval=?", (symbol, interval)).fetchone()[0]
        return pd.Timestamp(r) if r else None
