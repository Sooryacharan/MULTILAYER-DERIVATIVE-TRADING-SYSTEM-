"""Synthetic price generators for tests and demos. NOT market data; results say nothing about real edge."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_bars(n: int = 3000, seed: int = 0, start_price: float = 20000.0, daily_vol: float = 0.01,
              trend_persistence: float = 0.0, vol_regimes: bool = True, start: str = "2010-01-01") -> pd.DataFrame:
    """Daily OHLC. trend_persistence is the AR(1) coefficient on returns (0 => random walk, no edge)."""
    rng = np.random.default_rng(seed)
    vol = np.full(n, daily_vol)
    if vol_regimes:
        regime = np.repeat(rng.choice([0.7, 1.0, 1.8], size=n // 60 + 1, p=[0.3, 0.5, 0.2]), 60)[:n]
        vol = daily_vol * regime
    r = np.zeros(n)
    eps = rng.normal(0, 1, n) * vol
    for t in range(1, n):
        r[t] = trend_persistence * r[t - 1] + eps[t]
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.concatenate([[start_price], close[:-1]]) * np.exp(rng.normal(0, vol * 0.2))
    spread = np.abs(rng.normal(0, vol * 0.6)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
