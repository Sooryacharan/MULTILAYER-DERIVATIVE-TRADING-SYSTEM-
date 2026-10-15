"""Risk gate: the single choke point for every order (paper Section 3.1).

Sizing and limits live here and nowhere else. The judgement multiplier can only shrink
the size the gate would permit. NOTE: the paper specifies the gate as a separate process;
this prototype keeps it as an isolated class with persisted kill-switch state. Process
isolation is a deployment step before any live use.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RiskLimits:
    capital: float = 1_000_000.0
    risk_per_trade_pct: float = 0.005      # fraction of capital risked to the stop
    daily_loss_limit_pct: float = 0.02
    max_open_positions: int = 2
    max_consecutive_losses: int = 4
    max_slippage_pct: float = 0.003        # abnormal-slippage kill trigger


@dataclass
class GateDecision:
    approved: bool
    lots: int
    reason: str


class RiskGate:
    def __init__(self, limits: RiskLimits, state_path: str | Path | None = None):
        self.limits = limits
        self.state_path = Path(state_path) if state_path else None
        self.open_positions = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.killed = False
        self.kill_reason = ""
        if self.state_path and self.state_path.exists():
            s = json.loads(self.state_path.read_text())
            self.killed, self.kill_reason = s.get("killed", False), s.get("kill_reason", "")

    def _persist(self) -> None:
        if self.state_path:
            self.state_path.write_text(json.dumps({"killed": self.killed, "kill_reason": self.kill_reason}))

    def kill(self, reason: str) -> None:
        self.killed, self.kill_reason = True, reason
        self._persist()

    def reset_kill(self) -> None:
        """Deliberate manual action only; never called by the system itself."""
        self.killed, self.kill_reason = False, ""
        self._persist()

    def new_day(self) -> None:
        self.daily_pnl = 0.0

    def evaluate(self, entry: float, stop: float, lot_size: int, multiplier: float = 1.0) -> GateDecision:
        """Size an order. multiplier (from the judgement layer) is clamped to [0, 1]: it can never enlarge."""
        L = self.limits
        if self.killed:
            return GateDecision(False, 0, f"kill switch active: {self.kill_reason}")
        if self.daily_pnl <= -L.daily_loss_limit_pct * L.capital:
            self.kill("daily loss limit")
            return GateDecision(False, 0, "daily loss limit hit")
        if self.consecutive_losses >= L.max_consecutive_losses:
            self.kill("consecutive losses")
            return GateDecision(False, 0, "consecutive loss limit hit")
        if self.open_positions >= L.max_open_positions:
            return GateDecision(False, 0, "max open positions")
        risk_per_lot = abs(entry - stop) * lot_size
        if not risk_per_lot > 0:
            return GateDecision(False, 0, "invalid stop distance")
        base_lots = math.floor(L.capital * L.risk_per_trade_pct / risk_per_lot)
        m = min(1.0, max(0.0, multiplier))
        lots = math.floor(base_lots * m)
        if lots < 1:
            return GateDecision(False, 0, f"size rounds to zero (base={base_lots}, m={m:.2f})")
        return GateDecision(True, lots, f"approved base={base_lots} m={m:.2f}")

    def on_fill(self) -> None:
        self.open_positions += 1

    def on_close(self, net_pnl: float, slippage_pct: float = 0.0) -> None:
        self.open_positions = max(0, self.open_positions - 1)
        self.daily_pnl += net_pnl
        self.consecutive_losses = self.consecutive_losses + 1 if net_pnl < 0 else 0
        if slippage_pct > self.limits.max_slippage_pct:
            self.kill("abnormal slippage")
