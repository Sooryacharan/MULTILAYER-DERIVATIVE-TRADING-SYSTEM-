"""Daily-bar backtester with the Zerodha cost model, plus one parameter-light setup (Donchian breakout).

Signals form on bar i's close; entry fills at bar i+1's open (no lookahead). Stops are
evaluated intraday on subsequent bars; a gap through the stop fills at the open (worse price).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats
from .costs import FuturesCostModel
from .models import Signal, Trade


class DonchianBreakout:
    """Long above prior N-day high, short below prior N-day low.
    Exit on stop (k*ATR) or M-day opposite channel."""

    def __init__(self, entry_n: int = 20, exit_n: int = 10, atr_mult: float = 2.0, atr_n: int = 14):
        self.entry_n, self.exit_n, self.atr_mult, self.atr_n = entry_n, exit_n, atr_mult, atr_n

    def signal_at(self, symbol: str, bars: pd.DataFrame, i: int, lot_size: int) -> Signal | None:
        """Signal using bars[:i+1] only. Returns None if no setup fires."""
        if i < max(self.entry_n, self.atr_n + 1):
            return None
        w = bars.iloc[i - self.entry_n:i]
        c = bars["close"].iloc[i]
        a = stats.atr(bars["high"].to_numpy()[:i + 1], bars["low"].to_numpy()[:i + 1],
                      bars["close"].to_numpy()[:i + 1], self.atr_n)
        if np.isnan(a):
            return None
        d = bars.index[i].date()
        if c > w["high"].max():
            return Signal(symbol, d, +1, c, c - self.atr_mult * a, lot_size)
        if c < w["low"].min():
            return Signal(symbol, d, -1, c, c + self.atr_mult * a, lot_size)
        return None


def run_backtest(symbol: str, bars: pd.DataFrame, setup: DonchianBreakout, lot_size: int,
                 costs: FuturesCostModel, lots: int = 1) -> tuple[list[Trade], list[Signal]]:
    """Returns (trades, signals); signals[k] is the signal that produced trades[k]."""
    trades: list[Trade] = []
    signals: list[Signal] = []
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx = bars.index
    qty = lot_size * lots
    i, n = 0, len(bars)
    while i < n - 2:
        sig = setup.signal_at(symbol, bars, i, lot_size)
        if sig is None:
            i += 1
            continue
        entry_i = i + 1
        entry = o[entry_i]
        stop = sig.stop
        d = sig.direction
        exit_i, exit_px = n - 1, c[n - 1]
        for j in range(entry_i, n):
            stopped = (l[j] <= stop) if d > 0 else (h[j] >= stop)
            if stopped:
                gap = (o[j] <= stop) if d > 0 else (o[j] >= stop)
                exit_i, exit_px = j, (o[j] if gap else stop)
                break
            if j > entry_i and j >= setup.exit_n:
                ch = bars.iloc[j - setup.exit_n:j]
                if (d > 0 and c[j] < ch["low"].min()) or (d < 0 and c[j] > ch["high"].max()):
                    exit_i, exit_px = j, c[j]
                    break
        gross = (exit_px - entry) * d * qty
        trades.append(Trade(idx[entry_i].date(), idx[exit_i].date(), d, entry, exit_px, stop, qty,
                            gross, costs.round_trip(entry, exit_px, qty, d)))
        signals.append(Signal(symbol, sig.date, d, entry, stop, lot_size))
        i = exit_i + 1
    return trades, signals
