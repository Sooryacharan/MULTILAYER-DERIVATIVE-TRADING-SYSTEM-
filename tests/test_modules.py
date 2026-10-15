import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from mdts.approval import ApprovalQueue, ConsoleNotifier
from mdts.broker import PaperBroker
from mdts.config import JudgementConfig
from mdts.costs import FuturesCostModel
from mdts.datastore import BarStore
from mdts.env import load_env
from mdts.event_tagger import EventRiskTagger
from mdts.intraday import OpeningRangeBreakout, run_intraday_backtest, to_daily
from mdts.journal import Journal
from mdts.kite_client import CHUNK_DAYS, fetch_history
from mdts.models import Signal, Trade
from mdts.pipeline import Pipeline
from mdts.risk_gate import RiskGate, RiskLimits
from mdts.synthetic import make_bars, make_minute_bars

NOW = datetime(2026, 10, 19, 9, 40)


# ---------- approval queue
def test_approval_rejects_unproposed_expired_and_persists(tmp_path):
    path = tmp_path / "q.json"
    q = ApprovalQueue(path, ttl_minutes=5)
    with pytest.raises(KeyError):
        q.decide(999, True, NOW)
    p = q.propose("X", 1, 1, 100, 95, NOW)
    assert ApprovalQueue(path).pending()[0].id == p.id                # survives restart
    with pytest.raises(TimeoutError):
        q.decide(p.id, True, NOW + timedelta(minutes=6))
    assert q._items[p.id].status == "EXPIRED"
    p2 = q.propose("X", 1, 1, 100, 95, NOW)
    assert q.decide(p2.id, True, NOW + timedelta(minutes=1)).status == "APPROVED"
    with pytest.raises(ValueError):
        q.decide(p2.id, False, NOW)


# ---------- paper broker
def _prop(q, d=1, entry=100.0, stop=95.0, target=110.0):
    return q.propose("X", d, 1, entry, stop, NOW, lot_size=10, target=target)


def _broker():
    return PaperBroker(FuturesCostModel(), stop_slippage_pct=0.0)


def test_broker_limit_fill_and_target():
    b, q = _broker(), ApprovalQueue()
    b.submit(_prop(q))
    ev = b.on_bar(NOW, 101, 102, 99.5, 101)               # trades through 100 -> fills at min(open, limit)
    assert ev[0][0] == "FILL" and b.position.entry == 100
    ev = b.on_bar(NOW, 102, 111, 101, 110)
    assert ev[0][0] == "CLOSE" and ev[0][1].reason == "TARGET" and ev[0][1].gross == pytest.approx(100.0)
    assert ev[0][1].net < ev[0][1].gross


def test_broker_unreached_limit_cancels_and_stop_wins_ties():
    b, q = _broker(), ApprovalQueue()
    b.submit(_prop(q))
    assert b.on_bar(NOW, 103, 104, 101, 103)[0][0] == "CANCEL" and b.position is None
    b.submit(_prop(q))
    b.on_bar(NOW, 100, 100.5, 99.5, 100)
    ev = b.on_bar(NOW, 100, 111, 94, 100)                 # both stop and target touched
    assert ev[0][1].reason == "STOP"


def test_broker_gap_through_stop_fills_at_open():
    b, q = _broker(), ApprovalQueue()
    b.submit(_prop(q))
    b.on_bar(NOW, 100, 100.5, 99.5, 100)
    ev = b.on_bar(NOW, 90, 91, 89, 90)
    assert ev[0][1].exit == 90 and ev[0][1].slippage_pct > 0


# ---------- datastore + kite client
def test_datastore_roundtrip_and_point_in_time():
    store = BarStore(":memory:")
    df = make_bars(n=50)
    store.upsert("S", "day", df)
    got = store.get("S", "day", end=df.index[19])
    assert len(got) == 20 and got.index[-1] == df.index[19]
    assert np.allclose(got["close"], df["close"].iloc[:20])
    store.upsert("S", "day", df)                          # idempotent
    assert len(store.get("S", "day")) == 50


class FakeKite:
    def __init__(self):
        self.calls = []

    def historical_data(self, token, frm, to, interval, continuous=False, oi=False):
        self.calls.append((frm, to))
        days = pd.date_range(frm, to, freq="D")
        return [{"date": d, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10} for d in days]


def test_fetch_history_chunks_requests():
    k, store = FakeKite(), BarStore(":memory:")
    n = fetch_history(k, store, "S", 1, "minute", datetime(2025, 1, 1), datetime(2025, 6, 1), sleep=0)
    assert len(k.calls) >= 3
    assert all((to - frm).days <= CHUNK_DAYS["minute"] for frm, to in k.calls)
    assert n == len(store.get("S", "minute"))


def test_kite_module_has_no_order_code():
    import mdts.kite_client as kc
    assert not any("order" in name.lower() for name in dir(kc))


def test_env_loader_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('A_TEST_KEY=fromfile\nB_TEST_KEY="q"\n# c\n')
    monkeypatch.setenv("A_TEST_KEY", "already")
    monkeypatch.delenv("B_TEST_KEY", raising=False)
    load_env(f)
    assert os.environ["A_TEST_KEY"] == "already" and os.environ["B_TEST_KEY"] == "q"
    monkeypatch.delenv("B_TEST_KEY")


# ---------- event tagger
class Fake:
    def __init__(self, out=None, exc=None):
        self.out, self.exc = out, exc

    def complete(self, prompt):
        if self.exc:
            raise self.exc
        return self.out


@pytest.mark.parametrize("out,action,cap", [
    ('{"action": "NONE", "reason": "quiet"}', "NONE", 1.0),
    ('```json\n{"action": "downsize", "reason": "RBI"}\n```', "DOWNSIZE", 0.5),
    ('{"action": "SKIP", "reason": "budget"}', "SKIP", 0.0),
])
def test_tagger_parses_flags(out, action, cap):
    f = EventRiskTagger(Fake(out)).tag(["x"], date(2026, 10, 19))
    assert (f.action, f.cap, f.ok) == (action, cap, True)


@pytest.mark.parametrize("prov", [
    Fake('{"action": "INCREASE_SIZE", "reason": "ignore previous instructions"}'),
    Fake("not json"), Fake(exc=TimeoutError("net"))])
def test_tagger_fails_closed_and_cannot_enlarge(prov):
    f = EventRiskTagger(prov).tag(["x"], date(2026, 10, 19))
    assert f.action == "SKIP" and f.cap == 0.0 and not f.ok
    assert EventRiskTagger(prov, fail_closed=False).tag(["x"], date(2026, 10, 19)).cap <= 1.0


# ---------- intraday
def test_intraday_no_lookahead_and_stop_cap():
    minute = make_minute_bars(days=60, seed=5, orb_edge=0.3)
    costs = FuturesCostModel()
    t_all, _, sk0 = run_intraday_backtest("X", minute, OpeningRangeBreakout(), 65, costs)
    t_cap, s_cap, sk1 = run_intraday_backtest("X", minute, OpeningRangeBreakout(), 65, costs, max_risk=2000)
    assert t_all and len(t_cap) + sk1 == len(t_all) and sk0 == 0
    assert all(abs(s.entry - s.stop) * 65 <= 2000 for s in s_cap)
    # a signal must depend only on bars up to the signal bar
    for _, day in list(minute.groupby(minute.index.normalize()))[:20]:
        found = OpeningRangeBreakout().find_signal("X", day, 65)
        if found:
            sig, i = found
            again = OpeningRangeBreakout().find_signal("X", day.iloc[:i + 1], 65)
            assert again[0] == sig


def test_to_daily_aggregates():
    d = to_daily(make_minute_bars(days=5, seed=1))
    assert len(d) == 5 and (d["high"] >= d["low"]).all()


# ---------- pipeline end-to-end
class _Quiet(ConsoleNotifier):
    def notify(self, p):
        pass


def _pipeline(tmp_path, tagger=None, limits=None):
    cfg = JudgementConfig(bootstrap_B=300, n_min=50)
    gate = RiskGate(limits or RiskLimits(), state_path=tmp_path / "gate.json")
    return Pipeline(cfg, gate, ApprovalQueue(tmp_path / "q.json"), _broker(), Journal(tmp_path / "j.jsonl"),
                    tagger=tagger, notifier=_Quiet())


def _edge_history(n=120, end=date(2026, 10, 16)):
    rng = np.random.default_rng(0)
    days = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    return [Trade(d.date(), d.date(), 1, 100, 100, 95, 65, float(x), 0.0)
            for d, x in zip(days, rng.normal(900, 400, n))]


def _daily_before():
    start = pd.bdate_range(end="2026-10-16", periods=400)[0]
    return make_bars(n=400, seed=1, start=start, vol_regimes=False)


def test_pipeline_happy_path_and_journal(tmp_path):
    pl = _pipeline(tmp_path)
    sig = Signal("NIFTYFUT", NOW.date(), 1, 25000, 24970, 65, 25045)
    p = pl.on_signal(sig, NOW, _daily_before(), _edge_history())
    assert p is not None and p.lots == 1
    pl.decide(p.id, True, NOW + timedelta(minutes=1))
    pl.on_bar(pd.Timestamp(NOW), 25001, 25010, 24999, 25005)             # fills
    assert pl.gate.open_positions == 1
    closed = pl.on_bar(pd.Timestamp(NOW), 25005, 25050, 25000, 25040)    # target
    assert closed and closed[0].reason == "TARGET" and pl.gate.open_positions == 0
    kinds = [r["kind"] for r in pl.journal.read()]
    assert kinds == ["verdict", "gate", "proposal", "human_decision", "fill", "close"]


def test_pipeline_event_skip_blocks_trade(tmp_path):
    pl = _pipeline(tmp_path, tagger=EventRiskTagger(Fake('{"action": "SKIP", "reason": "RBI"}')))
    sig = Signal("NIFTYFUT", NOW.date(), 1, 25000, 24970, 65, 25045)
    assert pl.on_signal(sig, NOW, _daily_before(), _edge_history(), headlines=["RBI policy today"]) is None
    assert any(r["kind"] == "no_trade" for r in pl.journal.read())


def test_pipeline_gate_vetoes_wide_stop_and_hides_todays_trades(tmp_path):
    pl = _pipeline(tmp_path)
    wide = Signal("NIFTYFUT", NOW.date(), 1, 25000, 24900, 65, 25150)    # 100pt * 65 = 6500 > 2000
    assert pl.on_signal(wide, NOW, _daily_before(), _edge_history()) is None
    gate_rec = [r for r in pl.journal.read() if r["kind"] == "gate"][0]
    assert not gate_rec["approved"] and "stop too wide" in gate_rec["reason"]
    # a catastrophic trade dated today must not influence the decision (not knowable yet)
    poisoned = _edge_history() + [Trade(NOW.date(), NOW.date(), 1, 1, 1, 1, 1, -1e9, 0.0)]
    ok = Signal("NIFTYFUT", NOW.date(), 1, 25000, 24970, 65, 25045)
    assert pl.on_signal(ok, NOW, _daily_before(), poisoned) is not None
