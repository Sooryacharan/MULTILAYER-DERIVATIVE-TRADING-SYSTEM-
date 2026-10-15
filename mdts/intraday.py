"""Intraday setup (opening-range breakout) and its minute-bar backtester.

One signal per day at most. Signal forms on a minute bar's close; entry fills at the NEXT
bar's open. Stop = opposite side of the opening range; target = rr x risk. Positions are
squared off at the EOD time. Setups whose risk exceeds the risk gate's cap are counted and
skipped (the live gate would veto them anyway).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .costs import FuturesCostModel
from .models import Signal, Trade


def to_daily(minute: pd.DataFrame) -> pd.DataFrame:
    g = minute.groupby(minute.index.normalize())
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last()})


class OpeningRangeBreakout:
    def __init__(self, or_minutes: int = 15, rr: float = 1.5, session_start: str = "09:15",
                 entry_cutoff: str = "14:30", eod_exit: str = "15:15"):
        self.or_minutes, self.rr = or_minutes, rr
        self.session_start, self.entry_cutoff, self.eod_exit = session_start, entry_cutoff, eod_exit

    def find_signal(self, symbol: str, day: pd.DataFrame, lot_size: int) -> tuple[Signal, int] | None:
        """Returns (signal, position of signal bar) using only the day's bars up to the signal bar."""
        d0 = day.index[0].normalize()
        start = d0 + pd.Timedelta(self.session_start + ":00")
        or_end = start + pd.Timedelta(minutes=self.or_minutes)
        cutoff = d0 + pd.Timedelta(self.entry_cutoff + ":00")
        or_bars = day[(day.index >= start) & (day.index < or_end)]
        if len(or_bars) != self.or_minutes:            # incomplete opening range => no trade
            return None
        hi, lo = or_bars["high"].max(), or_bars["low"].min()
        if hi <= lo:
            return None
        for i in range(len(day)):
            ts = day.index[i]
            if ts < or_end or ts > cutoff:
                continue
            c = day["close"].iloc[i]
            if c > hi:
                return Signal(symbol, d0.date(), +1, c, lo, lot_size, c + self.rr * (c - lo)), i
            if c < lo:
                return Signal(symbol, d0.date(), -1, c, hi, lot_size, c - self.rr * (hi - c)), i
        return None


def run_intraday_backtest(symbol: str, minute: pd.DataFrame, setup: OpeningRangeBreakout, lot_size: int,
                          costs: FuturesCostModel, max_risk: float | None = None, lots: int = 1,
                          stop_slippage_pct: float = 0.0001):
    """Returns (trades, signals, n_skipped_wide). signals[k] produced trades[k]."""
    trades: list[Trade] = []
    signals: list[Signal] = []
    skipped = 0
    qty = lot_size * lots
    for _, day in minute.groupby(minute.index.normalize()):
        found = setup.find_signal(symbol, day, lot_size)
        if found is None:
            continue
        sig, i = found
        if i + 1 >= len(day):
            continue
        d = sig.direction
        o, h, l, c = (day[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        entry = o[i + 1]
        if (d > 0 and entry <= sig.stop) or (d < 0 and entry >= sig.stop):
            continue
        if max_risk is not None and abs(entry - sig.stop) * lot_size > max_risk:
            skipped += 1
            continue
        eod = day.index[0].normalize() + pd.Timedelta(setup.eod_exit + ":00")
        exit_px, exit_j = c[-1], len(day) - 1
        for j in range(i + 1, len(day)):
            if day.index[j] >= eod:
                exit_px, exit_j = c[j], j
                break
            stop_hit = l[j] <= sig.stop if d > 0 else h[j] >= sig.stop
            tgt_hit = h[j] >= sig.target if d > 0 else l[j] <= sig.target
            if stop_hit:
                gapped = j > i + 1 and ((o[j] <= sig.stop) if d > 0 else (o[j] >= sig.stop))
                base = o[j] if gapped else sig.stop
                exit_px, exit_j = base * (1 - stop_slippage_pct * d), j
                break
            if tgt_hit:
                exit_px, exit_j = sig.target, j
                break
        gross = (exit_px - entry) * d * qty
        dt = sig.date
        trades.append(Trade(dt, dt, d, entry, exit_px, sig.stop, qty, gross,
                            costs.round_trip(entry, exit_px, qty, d)))
        signals.append(Signal(symbol, dt, d, entry, sig.stop, lot_size, sig.target))
    return trades, signals, skipped
