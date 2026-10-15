"""Wires the layers in paper Figure 1 order for one decision:
setup signal -> event-risk tagger -> statistical judgement -> risk gate -> approval queue -> broker.
Every step is journaled. Multipliers only ever shrink; the human can only act on proposals this
pipeline created."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd

from .approval import ApprovalQueue, ConsoleNotifier, Notifier, Proposal
from .broker import ClosedTrade, PaperBroker
from .config import JudgementConfig
from .event_tagger import EventRiskTagger
from .journal import Journal
from .judgement import judge
from .models import Signal, Trade
from .risk_gate import RiskGate


class Pipeline:
    def __init__(self, cfg: JudgementConfig, gate: RiskGate, queue: ApprovalQueue, broker: PaperBroker,
                 journal: Journal, tagger: EventRiskTagger | None = None, notifier: Notifier | None = None):
        self.cfg, self.gate, self.queue, self.broker, self.journal = cfg, gate, queue, broker, journal
        self.tagger, self.notifier = tagger, notifier or ConsoleNotifier()

    def on_signal(self, signal: Signal, now: datetime, daily_bars: pd.DataFrame, trades: list[Trade],
                  headlines: list[str] | None = None, open_positions=None) -> Optional[Proposal]:
        """daily_bars must contain only COMPLETED sessions (strictly before today for an intraday signal)."""
        today = now.date()
        cap, flag = 1.0, None
        if self.tagger is not None:
            flag = self.tagger.tag(headlines or [], today)
            cap = flag.cap
            self.journal.log("event_flag", {"date": today, "action": flag.action, "reason": flag.reason, "ok": flag.ok})
        trades_before = [t for t in trades if t.exit_date < today]       # nothing from today is knowable yet
        v = judge(signal, today, daily_bars, trades_before, self.cfg, open_positions)
        self.journal.log("verdict", {"symbol": signal.symbol, **v.to_record()})
        m = min(v.multiplier, cap)                                        # can only shrink
        if v.verdict == "SKIP" or m <= 0:
            self.journal.log("no_trade", {"symbol": signal.symbol, "why": "judgement/event SKIP"})
            return None
        d = self.gate.evaluate(signal.entry, signal.stop, signal.lot_size, today, m)
        self.journal.log("gate", {"symbol": signal.symbol, "approved": d.approved, "lots": d.lots, "reason": d.reason})
        if not d.approved:
            return None
        p = self.queue.propose(signal.symbol, signal.direction, d.lots, signal.entry, signal.stop, now,
                               signal.lot_size, signal.target, v.to_record())
        self.journal.log("proposal", {"id": p.id, "symbol": p.symbol, "lots": p.lots, "entry": p.entry})
        self.notifier.notify(p)
        return p

    def decide(self, proposal_id: int, approve: bool, now: datetime) -> Proposal:
        """The human's tap. Approved proposals go to the broker as limit orders."""
        p = self.queue.decide(proposal_id, approve, now)
        self.journal.log("human_decision", {"id": p.id, "status": p.status})
        if p.status == "APPROVED":
            self.broker.submit(p)
        return p

    def on_bar(self, ts: pd.Timestamp, o: float, h: float, l: float, c: float) -> list[ClosedTrade]:
        closed = []
        for kind, obj in self.broker.on_bar(ts, o, h, l, c):
            if kind == "FILL":
                self.gate.on_fill(ts.date())
                self.journal.log("fill", {"symbol": obj.symbol, "entry": obj.entry, "qty": obj.qty})
            elif kind == "CANCEL":
                self.journal.log("cancel", {"id": obj.id, "why": "limit not reached"})
            elif kind == "CLOSE":
                self._record_close(obj, ts.date())
                closed.append(obj)
        return closed

    def square_off(self, ts: pd.Timestamp, price: float) -> Optional[ClosedTrade]:
        t = self.broker.square_off(ts, price)
        if t:
            self._record_close(t, ts.date())
        return t

    def _record_close(self, t: ClosedTrade, today: date) -> None:
        self.gate.on_close(t.net, today, t.slippage_pct)
        self.journal.log("close", {"symbol": t.symbol, "net": t.net, "reason": t.reason, "slippage_pct": t.slippage_pct})
