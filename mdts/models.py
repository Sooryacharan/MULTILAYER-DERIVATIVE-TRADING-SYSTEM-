"""Shared data records."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class Trade:
    entry_date: date
    exit_date: date
    direction: int          # +1 long, -1 short
    entry: float
    exit: float
    stop: float
    qty: int
    gross: float
    charges: float

    @property
    def net(self) -> float:
        return self.gross - self.charges


@dataclass(frozen=True)
class Signal:
    symbol: str
    date: date              # decision date (signal formed on this bar's close)
    direction: int
    entry: float            # intended entry price
    stop: float
    lot_size: int


@dataclass
class Verdict:
    verdict: str            # TAKE | DOWNSIZE | SKIP
    multiplier: float
    statistics: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)   # (message, number) pairs
    data_as_of: Optional[str] = None
    config_version: str = ""

    def to_record(self) -> dict:
        return {"verdict": self.verdict, "multiplier": self.multiplier, "statistics": self.statistics,
                "reasons": [{"reason": r, "value": v} for r, v in self.reasons],
                "data_as_of": self.data_as_of, "config_version": self.config_version}
