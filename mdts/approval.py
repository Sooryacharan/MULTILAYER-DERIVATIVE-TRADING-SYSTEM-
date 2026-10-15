"""Human approval queue. The human may approve/reject a proposal the system created,
and may never inject a trade the system did not propose (paper Section 3.1)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass
class Proposal:
    id: int
    symbol: str
    direction: int
    lots: int
    entry: float
    stop: float
    verdict_record: dict = field(default_factory=dict)
    status: str = "PENDING"      # PENDING | APPROVED | REJECTED


class ApprovalQueue:
    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self._items: dict[int, Proposal] = {}

    def propose(self, symbol: str, direction: int, lots: int, entry: float, stop: float,
                verdict_record: dict | None = None) -> Proposal:
        """Only called by system code after the risk gate approved."""
        p = Proposal(next(self._ids), symbol, direction, lots, entry, stop, verdict_record or {})
        self._items[p.id] = p
        return p

    def decide(self, proposal_id: int, approve: bool) -> Proposal:
        p = self._items.get(proposal_id)
        if p is None:
            raise KeyError(f"no such proposal {proposal_id}: unproposed trades cannot be approved")
        if p.status != "PENDING":
            raise ValueError(f"proposal {proposal_id} already {p.status}")
        p.status = "APPROVED" if approve else "REJECTED"
        return p

    def pending(self) -> list[Proposal]:
        return [p for p in self._items.values() if p.status == "PENDING"]
