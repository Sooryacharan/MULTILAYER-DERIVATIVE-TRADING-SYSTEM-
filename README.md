# Multilayer Derivative Trading System (prototype)

Reference implementation of the design in *Statistically Gated AI Trading Infrastructure for Retail
Derivatives* (draft v1.0, `ai trading system architecture.pdf`). Core rule: **AI/statistics may only make
the system more cautious, never more aggressive.**

**Status: research prototype. No broker connectivity, no live orders, no real market data. Not financial advice.**

## What's implemented

| Paper section | Module |
|---|---|
| 4 Statistical judgement layer (TAKE/DOWNSIZE/SKIP, m in [0,1]) | `mdts/judgement.py`, `mdts/stats.py`, `config/judgement_v1.json` |
| 5.1 Zerodha futures charge model | `mdts/costs.py` (rates are parameters; verify against real contract notes) |
| 5.2 Backtester + one parameter-light setup (Donchian breakout) | `mdts/backtest.py` |
| 3.1 Risk gate: single choke point, kill switch | `mdts/risk_gate.py` |
| 3.1 Human approval queue (no unproposed trades) | `mdts/approval.py` |
| 4.9 Audit journal (JSONL) | `mdts/journal.py` |
| 7 Gated vs ungated comparison (H1, H2) | `mdts/evaluation.py` |
| 5.4 Automated tests (lookahead, known-answer, monotonicity, missing data) | `tests/` |

Not implemented: Kite Connect data/execution, the LLM event-risk tagger and weekly review, Telegram
approvals, minute-data recording, H3 paper-trading stability, running the risk gate as a separate OS process
(it is an isolated class here; process isolation is required before live use).

## Run

```
pip install -r requirements.txt
python -m pytest -q
python run_demo.py        # synthetic data only
```

## Demo caveat

`run_demo.py` uses synthetic prices: a random walk (no edge) and an AR(1) trending series (built-in edge).
It shows the layer behaving as specified (SKIP when E_lo <= 0 after costs, otherwise TAKE/DOWNSIZE). It is
a plumbing check, **not evidence** the layer improves real trading. On the synthetic trending series the
gated run did *not* reduce max drawdown, which is exactly the kind of result Section 7 says to report.
Judgement defaults in `config/judgement_v1.json` are the paper's illustrative values, not tuned findings.

## Before any live use (paper Appendix B)

Verify cost rates against contract notes, re-check SEBI/Zerodha rules (static IP, daily login), obtain minute
data, pass walk-forward on real data, paper trade 8-12 weeks, start with one approval-gated lot.
