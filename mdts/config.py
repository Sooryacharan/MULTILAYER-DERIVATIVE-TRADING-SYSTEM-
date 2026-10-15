"""Versioned configuration for the statistical judgement layer (paper Table 2)."""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class JudgementConfig:
    version: str = "v1"
    n_min: int = 200
    alpha: float = 0.05
    bootstrap_B: int = 10_000
    seed: int = 42
    vol_window_returns: int = 20
    vol_pct_window: int = 252
    pi_high: float = 0.80
    pi_ext: float = 0.95
    w0: float = 0.10
    w1: float = 0.25
    corr_window: int = 60
    rho0: float = 0.50
    rho1: float = 0.90
    kappa0: float = 0.8
    kappa1: float = 1.2
    delta: float = 0.5
    m_min: float = 0.25
    oos_fraction: float = 0.3
    max_data_age_days: int = 4

    @classmethod
    def load(cls, path: str | Path) -> "JudgementConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)
