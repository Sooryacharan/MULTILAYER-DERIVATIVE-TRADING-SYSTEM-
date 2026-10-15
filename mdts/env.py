"""Minimal .env loader (no dependency). Secrets stay in a git-ignored .env file, never in code or chat."""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def require(name: str) -> str:
    v = os.environ.get(name, "")
    if not v:
        raise RuntimeError(f"{name} is not set; add it to your local .env file")
    return v
