from datetime import date

import numpy as np
import pandas as pd
import pytest

from mdts import stats
from mdts.approval import ApprovalQueue
from mdts.backtest import DonchianBreakout, run_backtest
from mdts.config import JudgementConfig
from mdts.costs import FuturesCostModel
from mdts.judgement import judge
from mdts.models import Signal, Trade
from mdts.risk_gate import RiskGate, RiskLimits
from mdts.synthetic import make_bars

CFG = JudgementConfig(bootstrap_B=500)
T = date(2020, 6, 30)
SIG = Signal("X", T, 1, 100.0, 95.0, 25)


def _bars(n=600, **kw):
    return make_bars(n=n, start="2018-01-01", **kw)


def _trades(nets, end=date(2020, 6, 29)):
    """Synthetic trade list with given net PnLs, exit dates ending at `end`."""
    n = len(nets)
    days = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    return [Trade(d.date(), d.date(), 1, 100, 100, 95, 25, float(x), 0.0) for d, x in zip(days, nets)]


def _t_for(bars):
    return bars.index[-1].date()


def test_ramp_shape():
    assert stats.ramp(0.5, 0.8, 0.95) == 1.0
    assert stats.ramp(0.95, 0.8, 0.95) == 0.0
    assert stats.ramp(0.875, 0.8, 0.95) == pytest.approx(0.5)
    assert stats.ramp(float("nan"), 0.8, 0.95) == 0.0


def test_wilson_known_value():
    lo, hi = stats.wilson_interval(50, 100, 0.05)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_bootstrap_reproducible():
    x = np.random.default_rng(0).normal(10, 100, 300)
    assert stats.bootstrap_expectancy(x, 500, 0.05, 7) == stats.bootstrap_expectancy(x, 500, 0.05, 7)


def test_cost_model_nonnegative_and_sell_has_stt():
    c = FuturesCostModel()
    assert c.order_charges(20000, 25, "SELL") > c.order_charges(20000, 25, "BUY")
    assert c.round_trip(20000, 20100, 25, 1) > 0


def test_known_answer_coin_flip_skips():
    rng = np.random.default_rng(1)
    bars = _bars()
    v = judge(SIG, _t_for(bars), bars, _trades(rng.choice([-100.0, 100.0], 400)), CFG)
    assert v.verdict == "SKIP"


def test_known_answer_strong_edge_takes():
    rng = np.random.default_rng(2)
    bars = _bars(vol_regimes=False)
    nets = rng.normal(80, 50, 400)           # strong stable edge, narrow win-rate interval
    v = judge(SIG, _t_for(bars), bars, _trades(nets), CFG)
    assert v.verdict in ("TAKE", "DOWNSIZE")
    assert v.multiplier > 0


def test_too_few_trades_skips():
    bars = _bars()
    assert judge(SIG, _t_for(bars), bars, _trades(np.full(50, 100.0)), CFG).verdict == "SKIP"


def test_lookahead_invariance():
    """Eq. 1: altering rows/trades dated after t must not change the verdict."""
    rng = np.random.default_rng(3)
    bars = _bars(vol_regimes=False)
    t = bars.index[400].date()
    trades = _trades(rng.normal(80, 50, 400), end=bars.index[399].date())
    base = judge(SIG, t, bars, trades, CFG)
    tampered = bars.copy()
    tampered.iloc[401:] *= 7.0
    future = trades + [Trade(date(2030, 1, 1), date(2030, 1, 2), 1, 1, 1, 1, 1, -1e9, 0.0)]
    again = judge(SIG, t, tampered, future, CFG)
    assert (base.verdict, base.multiplier) == (again.verdict, again.multiplier)
    assert base.statistics["E_lo"] == again.statistics["E_lo"]


@pytest.mark.parametrize("corrupt", ["nan", "stale"])
def test_missing_or_stale_data_skips(corrupt):
    rng = np.random.default_rng(4)
    bars = _bars(vol_regimes=False)
    trades = _trades(rng.normal(80, 50, 400))
    t = _t_for(bars)
    if corrupt == "nan":
        bars = bars.copy()
        bars.iloc[-3, bars.columns.get_loc("close")] = np.nan
    else:
        t = date(t.year + 1, t.month, t.day)
    assert judge(SIG, t, bars, trades, CFG).verdict == "SKIP"


def test_monotone_multiplier_bounds():
    rng = np.random.default_rng(5)
    for s in range(15):
        bars = _bars(n=500, seed=s)
        v = judge(SIG, _t_for(bars), bars, _trades(rng.normal(rng.uniform(-20, 120), 60, 300)), CFG)
        assert 0.0 <= v.multiplier <= 1.0
    gate = RiskGate(RiskLimits())
    base = gate.evaluate(100, 95, 25, 1.0).lots
    for m in (0.0, 0.3, 1.0, 5.0, 1e9):
        assert gate.evaluate(100, 95, 25, m).lots <= base


def test_correlated_open_position_downsizes_not_upsizes():
    rng = np.random.default_rng(6)
    bars = _bars(vol_regimes=False)
    trades = _trades(rng.normal(80, 50, 400))
    alone = judge(SIG, _t_for(bars), bars, trades, CFG)
    twin = {"Y": (1, bars["close"] * 1.01)}            # perfectly correlated, same direction
    with_twin = judge(SIG, _t_for(bars), bars, trades, CFG, open_positions=twin)
    assert with_twin.multiplier <= alone.multiplier
    assert with_twin.verdict == "SKIP"


def test_risk_gate_limits_and_kill_switch(tmp_path):
    g = RiskGate(RiskLimits(max_consecutive_losses=2), state_path=tmp_path / "k.json")
    assert g.evaluate(100, 95, 25).approved
    g.on_close(-100)
    g.on_close(-100)
    d = g.evaluate(100, 95, 25)
    assert not d.approved and g.killed
    assert RiskGate(RiskLimits(), state_path=tmp_path / "k.json").killed   # persists across restart
    g2 = RiskGate(RiskLimits())
    g2.on_close(0, slippage_pct=0.05)
    assert g2.killed


def test_gate_blocks_bad_order():
    g = RiskGate(RiskLimits())
    assert not g.evaluate(100, 100, 25).approved          # zero stop distance
    assert not g.evaluate(100, 20, 10_000).approved       # risk per lot exceeds budget -> zero lots


def test_approval_queue_rejects_unproposed():
    q = ApprovalQueue()
    with pytest.raises(KeyError):
        q.decide(999, True)
    p = q.propose("X", 1, 1, 100, 95)
    assert q.decide(p.id, True).status == "APPROVED"
    with pytest.raises(ValueError):
        q.decide(p.id, False)


def test_backtest_no_lookahead_entry_is_next_open():
    bars = make_bars(n=800, seed=3, trend_persistence=0.2)
    trades, sigs = run_backtest("X", bars, DonchianBreakout(), 25, FuturesCostModel())
    assert trades
    for tr, sg in zip(trades, sigs):
        assert tr.entry_date > sg.date
        assert tr.entry == bars.loc[pd.Timestamp(tr.entry_date), "open"]
