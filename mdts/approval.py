"""Human approval queue. The human may approve/reject a proposal the system created,
and may never inject a trade the system did not propose (paper Section 3.1).
Proposals expire (a stale breakout is not the setup that was validated) and are persisted
so a PC restart does not lose or resurrect them."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class Proposal:
    id: int
    symbol: str
    direction: int
    lots: int
    entry: float
    stop: float
    lot_size: int = 1
    target: Optional[float] = None
    verdict_record: dict = field(default_factory=dict)
    created_at: str = ""
    expires_at: str = ""
    status: str = "PENDING"      # PENDING | APPROVED | REJECTED | EXPIRED


class Notifier(Protocol):
    def notify(self, p: Proposal) -> None: ...


class ConsoleNotifier:
    def notify(self, p: Proposal) -> None:
        side = "BUY" if p.direction > 0 else "SELL"
        print(f"[approval #{p.id}] {side} {p.lots} lot {p.symbol} @ {p.entry} stop {p.stop} target {p.target} "
              f"(expires {p.expires_at})")


class ApprovalQueue:
    def __init__(self, path: str | Path | None = None, ttl_minutes: int = 5):
        self.path = Path(path) if path else None
        self.ttl = timedelta(minutes=ttl_minutes)
        self._items: dict[int, Proposal] = {}
        if self.path and self.path.exists():
            for d in json.loads(self.path.read_text()):
                self._items[d["id"]] = Proposal(**d)

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(p) for p in self._items.values()]))

    def propose(self, symbol: str, direction: int, lots: int, entry: float, stop: float, now: datetime,
                lot_size: int = 1, target: float | None = None, verdict_record: dict | None = None) -> Proposal:
        """Only called by system code after the risk gate approved."""
        pid = max(self._items, default=0) + 1
        p = Proposal(pid, symbol, direction, lots, entry, stop, lot_size, target, verdict_record or {},
                     now.isoformat(), (now + self.ttl).isoformat())
        self._items[pid] = p
        self._save()
        return p

    def decide(self, proposal_id: int, approve: bool, now: datetime) -> Proposal:
        p = self._items.get(proposal_id)
        if p is None:
            raise KeyError(f"no such proposal {proposal_id}: unproposed trades cannot be approved")
        if p.status != "PENDING":
            raise ValueError(f"proposal {proposal_id} already {p.status}")
        if now > datetime.fromisoformat(p.expires_at):
            p.status = "EXPIRED"
            self._save()
            raise TimeoutError(f"proposal {proposal_id} expired at {p.expires_at}")
        p.status = "APPROVED" if approve else "REJECTED"
        self._save()
        return p

    def pending(self) -> list[Proposal]:
        return [p for p in self._items.values() if p.status == "PENDING"]
