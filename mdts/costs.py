"""Zerodha-style charge model for NSE equity/index futures.

Rates are PARAMETERS, not facts: verify against Zerodha's brokerage calculator and
real contract notes before trusting any backtest (paper Section 5.1, Phase 1 gate).
Defaults reflect published rates as understood at time of writing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuturesCostModel:
    brokerage_pct: float = 0.0003      # 0.03% of turnover ...
    brokerage_cap: float = 20.0        # ... capped at Rs 20 per executed order
    exchange_txn_pct: float = 0.0000183   # NSE futures transaction charge (verify)
    stt_sell_pct: float = 0.0005       # on sell side only (futures)
    stamp_buy_pct: float = 0.00002     # on buy side only
    sebi_pct: float = 0.000001         # Rs 10 per crore
    gst_pct: float = 0.18              # on brokerage + exchange txn + SEBI fee

    def order_charges(self, price: float, qty: int, side: str) -> float:
        """Total charges in rupees for one executed order. side: 'BUY' or 'SELL'."""
        turnover = price * qty
        brokerage = min(self.brokerage_cap, self.brokerage_pct * turnover)
        exch = self.exchange_txn_pct * turnover
        sebi = self.sebi_pct * turnover
        gst = self.gst_pct * (brokerage + exch + sebi)
        stt = self.stt_sell_pct * turnover if side == "SELL" else 0.0
        stamp = self.stamp_buy_pct * turnover if side == "BUY" else 0.0
        return brokerage + exch + sebi + gst + stt + stamp

    def round_trip(self, entry: float, exit_: float, qty: int, direction: int) -> float:
        """Charges for entering and exiting. direction: +1 long, -1 short."""
        entry_side, exit_side = ("BUY", "SELL") if direction > 0 else ("SELL", "BUY")
        return self.order_charges(entry, qty, entry_side) + self.order_charges(exit_, qty, exit_side)
