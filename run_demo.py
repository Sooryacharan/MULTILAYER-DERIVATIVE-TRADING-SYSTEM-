"""Demo on SYNTHETIC data: random walk (no edge) vs trending series (built-in edge).
A long synthetic history is used so the 200-trade minimum is met.
Shows the layer SKIPping a no-edge setup and gating a real one. Not evidence about real markets."""
from pathlib import Path

from mdts.backtest import DonchianBreakout, run_backtest
from mdts.config import JudgementConfig
from mdts.costs import FuturesCostModel
from mdts.evaluation import compare_gated_ungated
from mdts.journal import Journal
from mdts.synthetic import make_bars

cfg = JudgementConfig.load(Path(__file__).parent / "config" / "judgement_v1.json")
costs = FuturesCostModel()
LOT = 25  # illustrative lot size


def run(label, **kw):
    bars = make_bars(n=24000, **kw)
    trades, sigs = run_backtest("SYNTH", bars, DonchianBreakout(), LOT, costs)
    split = bars.index[len(bars) // 2].date()
    j = Journal(f"journal/{label}.jsonl")
    c = compare_gated_ungated(bars, trades, sigs, cfg, split, journal=j)
    print(f"\n=== {label}: {len(trades)} trades total, {c.n_signals} evaluated out-of-sample")
    print("verdicts:", c.counts)
    for name, m in (("ungated", c.ungated), ("gated", c.gated)):
        if m.get("n"):
            print(f"{name:8s} E={m['E']:9.1f} [{m['E_lo']:9.1f},{m['E_hi']:9.1f}] "
                  f"maxDD={m['max_drawdown']:.3%} PF={m['profit_factor']:.2f} net={m['total_net']:.0f}")
    print(f"H1 drawdown reduced: {c.h1_drawdown_reduced} | H2 expectancy/risk ok: {c.h2_expectancy_per_risk_ok}")


run("random_walk_no_edge", seed=1, trend_persistence=0.0)
run("trending_with_edge", seed=2, trend_persistence=0.25)
