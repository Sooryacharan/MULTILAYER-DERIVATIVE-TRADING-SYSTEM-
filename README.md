# Multilayer Derivative Trading System (prototype)

Reference implementation of the design in *Statistically Gated AI Trading Infrastructure for Retail
Derivatives* (draft v1.0, `ai trading system architecture.pdf`). Core rule: **AI/statistics may only make
the system more cautious, never more aggressive.**

**Status: research prototype. Paper trading only. No live orders can be sent, no real market data is
bundled. Not financial advice.**

## What's implemented

| Paper section | Module |
|---|---|
| 4 Statistical judgement layer (TAKE/DOWNSIZE/SKIP, m in [0,1]) | `mdts/judgement.py`, `mdts/stats.py`, `config/judgement_v1.json` |
| 5.1 Zerodha futures charge model | `mdts/costs.py` (rates are parameters; verify against real contract notes) |
| 5.2 Daily backtester + Donchian setup | `mdts/backtest.py` |
| Intraday setup (opening-range breakout) + minute backtester | `mdts/intraday.py` |
| 3.1 Risk gate (owner's limits: Rs 2,000 stop cap, 1 trade/day, daily/weekly loss, drawdown kill, loss-streak pause, margin buffer; persisted state) | `mdts/risk_gate.py` |
| 3.1 Human approval queue (no unproposed trades, expiry, persistence) | `mdts/approval.py` |
| Paper broker (limit fills, stops, targets, conservative tie-breaks) | `mdts/broker.py` |
| Data store (SQLite, point-in-time reads) and read-only Kite history client | `mdts/datastore.py`, `mdts/kite_client.py` |
| AI event-risk tagger (Gemini; can only SKIP/DOWNSIZE, fails closed) | `mdts/event_tagger.py` |
| Signal -> tagger -> judgement -> gate -> approval -> broker, all journaled | `mdts/pipeline.py`, `mdts/journal.py` |
| 7 Gated vs ungated comparison (H1, H2) | `mdts/evaluation.py` |
| 5.4 Automated tests | `tests/` |

## Not done / untested

- `kite_client.py` and `GeminiProvider` are written from public docs and tested only with fakes. Verify
  against the live APIs (interval chunk limits, model name) before relying on them.
- No live broker exists by design. Going live is a separate, deliberate step after the paper phase.
- Real data: none bundled. Continuous-futures stitching, minute-data recording, H3 paper stability, Telegram
  notifier and weekly journal review are not built.
- The risk gate is an isolated class, not yet a separate OS process.

## Run

```
pip install -r requirements.txt
python -m pytest -q
python run_demo.py            # daily setup, synthetic data
python run_intraday_demo.py   # intraday setup under the Rs 2,000 stop cap, synthetic data
```

Secrets go in a local, git-ignored `.env` (copy `.env.example`). Never commit keys.

## Findings so far (synthetic data only, plumbing checks, not evidence about real markets)

- With a Rs 2,000 risk cap on a Nifty lot of 65, the stop must be within ~31 points. A 15-minute opening-range
  stop is almost always wider, so the intraday demo skipped ~97% of days for stop width in the no-edge run
  and ~83% in the with-edge run. The owner's limits and this setup do not fit; a tighter-stop setup is needed.
- With one lot, any multiplier below 1 floors to zero lots, so DOWNSIZE behaves as SKIP.
- Charges are material at this size (about 10% of gross profit even in the synthetic winning case).
- In the daily demo the gated run did not reduce max drawdown on the synthetic trending series.

## Before any live use (paper Appendix B)

Verify cost rates against contract notes, re-check SEBI/Zerodha rules (static IP, daily login), obtain minute
data, pass walk-forward on real data, paper trade 8-12 weeks, start with one approval-gated lot.
