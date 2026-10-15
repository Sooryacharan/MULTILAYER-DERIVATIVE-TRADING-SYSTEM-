"""Paper broker: simulates limit-order fills, protective stops and targets on live/replayed bars.

No real orders are ever sent from here. A live broker would implement the same two methods
(`submit`, `on_bar`) and is intentionally not written yet.
Conservative fill rules: a limit fills only if price trades through it; if stop and target are
both touched in one bar the stop is assumed first; stops fill at the open when gapped through,
plus adverse slippage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .approval import Proposal
from .costs import FuturesCostModel


@dataclass
class Position:
    proposal_id: int
    symbol: str
    direction: int
    qty: int
    entry: float
    stop: float
    target: Optional[float]
    entry_ts: object


@dataclass
class ClosedTrade:
    proposal_id: int
    symbol: str
    direction: int
    qty: int
    entry: float
    exit: float
    entry_ts: object
    exit_ts: object
    gross: float
    charges: float
    reason: str
    slippage_pct: float

    @property
    def net(self) -> float:
        return self.gross - self.charges


class PaperBroker:
    def __init__(self, costs: FuturesCostModel, stop_slippage_pct: float = 0.0001):
        self.costs = costs
        self.stop_slippage_pct = stop_slippage_pct
        self.pending: Optional[Proposal] = None
        self.position: Optional[Position] = None

    def submit(self, p: Proposal) -> None:
        if self.pending or self.position:
            raise RuntimeError("one position at a time")
        self.pending = p

    def _exit(self, ts, price: float, reason: str, intended: float) -> ClosedTrade:
        pos, self.position = self.position, None
        gross = (price - pos.entry) * pos.direction * pos.qty
        charges = self.costs.round_trip(pos.entry, price, pos.qty, pos.direction)
        slip = abs(price - intended) / intended if intended else 0.0
        return ClosedTrade(pos.proposal_id, pos.symbol, pos.direction, pos.qty, pos.entry, price,
                           pos.entry_ts, ts, gross, charges, reason, slip)

    def _check_exit(self, ts, o, h, l, entry_bar: bool):
        pos, d = self.position, self.position.direction
        stop_hit = l <= pos.stop if d > 0 else h >= pos.stop
        tgt_hit = pos.target is not None and (h >= pos.target if d > 0 else l <= pos.target)
        if stop_hit:                                   # stop wins any tie
            gapped = (not entry_bar) and ((o <= pos.stop) if d > 0 else (o >= pos.stop))
            base = o if gapped else pos.stop
            px = base * (1 - self.stop_slippage_pct * d)
            return self._exit(ts, px, "STOP", pos.stop)
        if tgt_hit:
            return self._exit(ts, pos.target, "TARGET", pos.target)
        return None

    def on_bar(self, ts, o: float, h: float, l: float, c: float) -> list[tuple[str, object]]:
        """Feed one bar. Returns events: ('FILL', Position) | ('CANCEL', Proposal) | ('CLOSE', ClosedTrade)."""
        events: list[tuple[str, object]] = []
        if self.pending is not None:
            p, self.pending = self.pending, None           # an order is valid for one bar only
            d = p.direction
            fill = None
            if d > 0 and l <= p.entry:
                fill = min(o, p.entry)
            elif d < 0 and h >= p.entry:
                fill = max(o, p.entry)
            if fill is None:
                events.append(("CANCEL", p))
                return events
            self.position = Position(p.id, p.symbol, d, p.lots * p.lot_size, fill, p.stop, p.target, ts)
            events.append(("FILL", self.position))
            closed = self._check_exit(ts, o, h, l, entry_bar=True)
            if closed:
                events.append(("CLOSE", closed))
            return events
        if self.position is not None:
            closed = self._check_exit(ts, o, h, l, entry_bar=False)
            if closed:
                events.append(("CLOSE", closed))
        return events

    def square_off(self, ts, price: float, reason: str = "EOD") -> Optional[ClosedTrade]:
        if self.position is None:
            return None
        return self._exit(ts, price, reason, price)
