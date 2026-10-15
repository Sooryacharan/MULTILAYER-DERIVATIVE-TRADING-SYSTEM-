"""Intraday demo on SYNTHETIC minute data: opening-range breakout under the owner's risk limits.
Plumbing check only; says nothing about real markets."""
from pathlib import Path

from mdts.config import JudgementConfig
from mdts.costs import FuturesCostModel
from mdts.evaluation import compare_gated_ungated
from mdts.intraday import OpeningRangeBreakout, run_intraday_backtest, to_daily
from mdts.risk_gate import RiskLimits
from mdts.synthetic import make_minute_bars

cfg = JudgementConfig.load(Path(__file__).parent / "config" / "judgement_v1.json")
costs, lim, LOT = FuturesCostModel(), RiskLimits(), 65     # Nifty lot as given by the owner


def run(label, **kw):
    minute = make_minute_bars(days=1500, **kw)
    trades, sigs, skipped = run_intraday_backtest("NIFTYFUT", minute, OpeningRangeBreakout(), LOT, costs,
                                                  max_risk=lim.max_risk_per_trade)
    daily = to_daily(minute)
    split = daily.index[len(daily) // 2].date()
    c = compare_gated_ungated(daily, trades, sigs, cfg, split, intraday=True)
    gross = sum(t.gross for t in trades)
    chg = sum(t.charges for t in trades)
    print(f"\n=== {label}: {len(trades)} trades, {skipped} skipped for stop wider than Rs {lim.max_risk_per_trade:.0f}")
    print(f"charges consumed {chg / gross:.0%} of gross profit" if gross > 0 else "gross profit <= 0")
    print("verdicts:", c.counts)
    for name, m in (("ungated", c.ungated), ("gated", c.gated)):
        if m.get("n"):
            print(f"{name:8s} E={m['E']:8.1f} [{m['E_lo']:8.1f},{m['E_hi']:8.1f}] maxDD={m['max_drawdown']:.2%} "
                  f"PF={m['profit_factor']:.2f} net={m['total_net']:.0f}")


run("no_edge", seed=1, orb_edge=0.0)
run("with_edge", seed=2, orb_edge=0.35)
