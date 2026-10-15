"""Risk gate: the single choke point for every order (paper Section 3.1).

Sizing and limits live here and nowhere else. The judgement multiplier can only shrink the
size the gate would permit. State is persisted to disk because the system restarts every
trading morning (daily login) and limits must survive restarts.

NOTE: the paper specifies the gate as a separate process; here it is an isolated class.
Process isolation is a deployment step before any live use.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RiskLimits:
    """Defaults are the owner's own limits (Rs 1,00,000 account, one lot of margin ~Rs 85,000)."""
    capital: float = 100_000.0
    margin_per_lot: float = 85_000.0
    min_free_buffer: float = 5_000.0          # free cash must never fall below this
    max_risk_per_trade: float = 2_000.0       # hard cap on loss to the stop
    max_lots: int = 1
    max_open_positions: int = 1
    max_trades_per_day: int = 1
    daily_loss_limit: float = 2_000.0         # blocks trading for the rest of the day
    weekly_loss_limit: float = 5_000.0        # blocks trading for the rest of the week
    total_drawdown_kill: float = 10_000.0     # peak-to-current equity; kills until manual reset
    max_consecutive_losses: int = 3           # pauses one trading day
    max_slippage_pct: float = 0.003           # abnormal-slippage kill trigger


@dataclass
class GateDecision:
    approved: bool
    lots: int
    reason: str


def _next_business_day(d: date) -> date:
    return np.busday_offset(d, 1, roll="forward").astype("datetime64[D]").astype(object)


class RiskGate:
    def __init__(self, limits: RiskLimits, state_path: str | Path | None = None):
        self.limits = limits
        self.state_path = Path(state_path) if state_path else None
        self.open_positions = 0
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.cum_pnl = 0.0
        self.peak_pnl = 0.0
        self.consecutive_losses = 0
        self.pause_until: date | None = None
        self.killed = False
        self.kill_reason = ""
        self._day: date | None = None
        self._week: tuple | None = None
        if self.state_path and self.state_path.exists():
            self._load(json.loads(self.state_path.read_text()))

    # ---- persistence
    def _load(self, s: dict) -> None:
        self.open_positions = s.get("open_positions", 0)
        self.trades_today = s.get("trades_today", 0)
        self.daily_pnl, self.weekly_pnl = s.get("daily_pnl", 0.0), s.get("weekly_pnl", 0.0)
        self.cum_pnl, self.peak_pnl = s.get("cum_pnl", 0.0), s.get("peak_pnl", 0.0)
        self.consecutive_losses = s.get("consecutive_losses", 0)
        pu = s.get("pause_until")
        self.pause_until = date.fromisoformat(pu) if pu else None
        d = s.get("day")
        self._day = date.fromisoformat(d) if d else None
        w = s.get("week")
        self._week = tuple(w) if w else None
        self.killed, self.kill_reason = s.get("killed", False), s.get("kill_reason", "")

    def _persist(self) -> None:
        if not self.state_path:
            return
        self.state_path.write_text(json.dumps({
            "open_positions": self.open_positions, "trades_today": self.trades_today,
            "daily_pnl": self.daily_pnl, "weekly_pnl": self.weekly_pnl,
            "cum_pnl": self.cum_pnl, "peak_pnl": self.peak_pnl,
            "consecutive_losses": self.consecutive_losses,
            "pause_until": self.pause_until.isoformat() if self.pause_until else None,
            "day": self._day.isoformat() if self._day else None,
            "week": list(self._week) if self._week else None,
            "killed": self.killed, "kill_reason": self.kill_reason}))

    # ---- state transitions
    def _roll(self, today: date) -> None:
        if self._day != today:
            self._day, self.daily_pnl, self.trades_today = today, 0.0, 0
        wk = tuple(today.isocalendar())[:2]
        if self._week != wk:
            self._week, self.weekly_pnl = wk, 0.0

    def kill(self, reason: str) -> None:
        self.killed, self.kill_reason = True, reason
        self._persist()

    def reset_kill(self) -> None:
        """Deliberate manual action after reviewing the system; never called by the system itself."""
        self.killed, self.kill_reason = False, ""
        self.peak_pnl = self.cum_pnl      # restart the drawdown envelope from here
        self._persist()

    @property
    def equity(self) -> float:
        return self.limits.capital + self.cum_pnl

    # ---- the one decision
    def evaluate(self, entry: float, stop: float, lot_size: int, today: date,
                 multiplier: float = 1.0) -> GateDecision:
        """Size an order. multiplier (from the judgement layer) is clamped to [0, 1]: it can never enlarge."""
        L = self.limits
        self._roll(today)
        if self.killed:
            return GateDecision(False, 0, f"kill switch active: {self.kill_reason}")
        if self.pause_until and today <= self.pause_until:
            return GateDecision(False, 0, f"paused after {L.max_consecutive_losses} losses until {self.pause_until}")
        if self.daily_pnl <= -L.daily_loss_limit:
            return GateDecision(False, 0, "daily loss limit hit")
        if self.weekly_pnl <= -L.weekly_loss_limit:
            return GateDecision(False, 0, "weekly loss limit hit")
        if self.trades_today >= L.max_trades_per_day:
            return GateDecision(False, 0, "max trades today")
        if self.open_positions >= L.max_open_positions:
            return GateDecision(False, 0, "max open positions")
        risk_per_lot = abs(entry - stop) * lot_size
        if not risk_per_lot > 0:
            return GateDecision(False, 0, "invalid stop distance")
        if risk_per_lot > L.max_risk_per_trade:
            max_pts = L.max_risk_per_trade / lot_size
            return GateDecision(False, 0, f"stop too wide: risk {risk_per_lot:.0f} > {L.max_risk_per_trade:.0f} "
                                          f"(max {max_pts:.1f} points)")
        lots_by_risk = math.floor(L.max_risk_per_trade / risk_per_lot)
        lots_by_margin = math.floor((self.equity - L.min_free_buffer) / L.margin_per_lot)
        base_lots = min(L.max_lots, lots_by_risk, lots_by_margin)
        m = min(1.0, max(0.0, multiplier))
        lots = math.floor(base_lots * m)
        if lots < 1:
            return GateDecision(False, 0, f"size rounds to zero (base={base_lots}, m={m:.2f}, "
                                          f"margin lots={lots_by_margin})")
        return GateDecision(True, lots, f"approved base={base_lots} m={m:.2f}")

    def on_fill(self, today: date) -> None:
        self._roll(today)
        self.open_positions += 1
        self.trades_today += 1
        self._persist()

    def on_close(self, net_pnl: float, today: date, slippage_pct: float = 0.0) -> None:
        L = self.limits
        self._roll(today)
        self.open_positions = max(0, self.open_positions - 1)
        self.daily_pnl += net_pnl
        self.weekly_pnl += net_pnl
        self.cum_pnl += net_pnl
        self.peak_pnl = max(self.peak_pnl, self.cum_pnl)
        if net_pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= L.max_consecutive_losses:
                self.pause_until = _next_business_day(today)
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0
        if self.peak_pnl - self.cum_pnl >= L.total_drawdown_kill:
            self.kill(f"total drawdown >= {L.total_drawdown_kill:.0f}")
        elif slippage_pct > L.max_slippage_pct:
            self.kill("abnormal slippage")
        self._persist()
