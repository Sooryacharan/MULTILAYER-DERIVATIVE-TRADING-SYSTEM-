"""Append-only JSONL journal: every verdict and order event is reconstructable."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, payload: dict) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
