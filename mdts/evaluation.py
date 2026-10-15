"""Pre-registered gated-vs-ungated comparison (paper Section 7) on identical periods.

Each trade is judged at its signal date using only trades closed before that date, so the
gated run is generated exactly as it would have been in real time. Gated PnL scales the
trade's net PnL by the multiplier (fractional lots; the live gate floors to whole lots).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import stats
from .config import JudgementConfig
from .judgement import judge
from .models import Signal, Trade
from .journal import Journal


@dataclass
class Comparison:
    n_signals: int
    counts: dict
    ungated: dict
    gated: dict
    h1_drawdown_reduced: bool
    h2_expectancy_per_risk_ok: bool


def _metrics(net: np.ndarray, capital: float, cfg: JudgementConfig) -> dict:
    if len(net) == 0:
        return {"n": 0}
    E, lo, hi = stats.bootstrap_expectancy(net, min(cfg.bootstrap_B, 2000), cfg.alpha, cfg.seed)
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {"n": int(len(net)), "E": E, "E_lo": lo, "E_hi": hi,
            "max_drawdown": stats.max_drawdown(capital + np.cumsum(net)),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
            "total_net": float(net.sum())}


def compare_gated_ungated(bars: pd.DataFrame, trades: list[Trade], signals: list[Signal],
                          cfg: JudgementConfig, oos_start, capital: float = 1_000_000.0,
                          tolerance: float = 0.2, journal: Journal | None = None,
                          intraday: bool = False) -> Comparison:
    """Evaluate on signals dated >= oos_start. `tolerance` is the pre-set H2 allowance (fraction).
    intraday=True: `bars` are daily bars; the signal day's own bar and trades are hidden (incomplete at decision time)."""
    ung, gat, counts = [], [], {"TAKE": 0, "DOWNSIZE": 0, "SKIP": 0}
    mults = []
    for tr, sg in zip(trades, signals):
        if sg.date < oos_start:
            continue
        if intraday:
            v = judge(sg, sg.date, bars.loc[bars.index < pd.Timestamp(sg.date)],
                      [t for t in trades if t.exit_date < sg.date], cfg, capital=capital)
        else:
            v = judge(sg, sg.date, bars, trades, cfg, capital=capital)
        counts[v.verdict] += 1
        if journal:
            journal.log("verdict", {"signal_date": sg.date, **v.to_record()})
        ung.append(tr.net)
        gat.append(tr.net * v.multiplier)
        mults.append(v.multiplier)
    ung_a, gat_a = np.array(ung), np.array(gat)
    mu, mg = _metrics(ung_a, capital, cfg), _metrics(gat_a, capital, cfg)
    # H2: expectancy per unit of risk taken (risk = sum of multipliers, i.e. fraction of full size deployed)
    exposure = float(np.sum(mults)) if mults else 0.0
    per_risk_g = (gat_a.sum() / exposure) if exposure > 0 else 0.0
    per_risk_u = ung_a.mean() if len(ung_a) else 0.0
    h2 = per_risk_g >= per_risk_u * (1 - tolerance) if per_risk_u > 0 else per_risk_g >= per_risk_u
    # H1 is vacuous if the layer skipped everything, so require at least one trade actually taken
    h1 = bool(exposure > 0 and mg.get("max_drawdown", 0) < mu.get("max_drawdown", 0))
    return Comparison(len(ung), counts, mu, mg, h1, bool(h2))
