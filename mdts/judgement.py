"""Statistical judgement layer (paper Section 4): TAKE / DOWNSIZE / SKIP with m in [0, 1].

Deterministic, no LLM calls, lookahead-free: only rows dated <= t and trades with
exit_date <= t are ever read. Missing/stale/NaN data => SKIP, never imputed.
"""
from __future__ import annotations

from datetime import date
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import stats
from .config import JudgementConfig
from .models import Signal, Trade, Verdict


def _skip(reason: str, value, cfg: JudgementConfig, t: date, statistics: dict | None = None) -> Verdict:
    return Verdict("SKIP", 0.0, statistics or {}, [(reason, value)], t.isoformat(), cfg.version)


def judge(
    signal: Signal,
    t: date,
    bars: pd.DataFrame,
    trades: Sequence[Trade],
    cfg: JudgementConfig,
    open_positions: Mapping[str, tuple[int, pd.Series]] | None = None,
    capital: float = 1_000_000.0,
) -> Verdict:
    """bars: DataFrame indexed by date with open/high/low/close.
    open_positions: symbol -> (direction, close series)."""
    open_positions = open_positions or {}

    # ---- point-in-time slice (Eq. 1): nothing after t is visible below this line
    D = bars.loc[bars.index <= pd.Timestamp(t)]
    hist = sorted((tr for tr in trades if tr.exit_date <= t), key=lambda tr: tr.exit_date)

    # ---- hard failures
    if len(D) == 0 or D[["open", "high", "low", "close"]].isna().any().any():
        return _skip("data missing or NaN", None, cfg, t)
    age = (pd.Timestamp(t) - D.index[-1]).days
    if age > cfg.max_data_age_days:
        return _skip("data stale (days)", age, cfg, t)
    if len(D) < cfg.vol_window_returns + 2:
        return _skip("insufficient price history", len(D), cfg, t)
    if len(hist) < cfg.n_min:
        return _skip("too few historical trades", len(hist), cfg, t)

    net = np.array([tr.net for tr in hist])
    E, E_lo, E_hi = stats.bootstrap_expectancy(net, cfg.bootstrap_B, cfg.alpha, cfg.seed)
    decomp = stats.payoff_decomposition(net)
    S: dict = {"n": len(hist), "E": E, "E_lo": E_lo, "E_hi": E_hi, **decomp,
               "distribution": stats.distribution_summary(net)}
    if not E_lo > 0:
        return _skip("no demonstrated after-cost edge (E_lo<=0)", E_lo, cfg, t, S)

    k = int(len(net) * (1 - cfg.oos_fraction))
    E_is, E_oos = float(net[:k].mean()), float(net[k:].mean())
    S.update(E_is=E_is, E_oos=E_oos)
    if E_is <= 0:
        return _skip("in-sample expectancy non-positive", E_is, cfg, t, S)
    Dg = E_oos / E_is
    S["degradation"] = Dg
    if Dg < cfg.delta:
        return _skip("overfit warning (OOS/IS below delta)", Dg, cfg, t, S)

    # ---- soft multipliers
    reasons: list = []
    close = D["close"].to_numpy(dtype=float)

    pi_t = stats.vol_percentile(close, cfg.vol_window_returns, cfg.vol_pct_window)
    S["vol_percentile"] = pi_t
    m_sigma = stats.ramp(pi_t, cfg.pi_high, cfg.pi_ext)
    if m_sigma < 1:
        reasons.append((f"volatility percentile (ramp {cfg.pi_high}->{cfg.pi_ext}) -> m_sigma={m_sigma:.2f}", pi_t))

    lo, hi = stats.wilson_interval(int(np.sum(net > 0)), len(net), cfg.alpha)
    width = hi - lo
    S["wilson"] = {"lo": lo, "hi": hi, "width": width}
    m_w = stats.ramp(width, cfg.w0, cfg.w1)
    if m_w < 1:
        reasons.append((f"win-rate interval width (ramp {cfg.w0}->{cfg.w1}) -> m_w={m_w:.2f}", width))

    rho_max = float("nan")
    corrs = []
    x = np.diff(np.log(close))
    for _sym, (direction, series) in open_positions.items():
        if direction != signal.direction:
            continue
        ys = series.loc[series.index <= pd.Timestamp(t)].to_numpy(dtype=float)
        corrs.append(stats.rolling_corr(x, np.diff(np.log(ys)), cfg.corr_window))
    if corrs:
        # unknown (NaN) correlation is treated as worst case, never as zero
        rho_max = float("inf") if any(np.isnan(c) for c in corrs) else max(corrs)
        m_rho = stats.ramp(rho_max, cfg.rho0, cfg.rho1)
    else:
        m_rho = 1.0
    S["rho_max"] = rho_max if corrs else None
    if m_rho < 1:
        reasons.append((f"max correlation with open positions (window {cfg.corr_window}) -> m_rho={m_rho:.2f}", rho_max))

    equity = capital + np.cumsum(net)
    train_dd = stats.max_drawdown(equity[:k]) if k > 0 else 0.0
    dd = stats.current_drawdown(equity)
    ratio = (dd / train_dd) if train_dd > 0 else (0.0 if dd == 0 else float("inf"))
    S.update(drawdown=dd, max_dd_train=train_dd, drawdown_ratio=ratio)
    m_d = stats.ramp(ratio, cfg.kappa0, cfg.kappa1)
    if m_d < 1:
        reasons.append((f"drawdown vs training envelope -> m_d={m_d:.2f}", ratio))

    # ---- Eq. 9: minimum of ramps, capped at 1, so m can never exceed 1
    m = min(1.0, m_sigma, m_w, m_rho, m_d)
    S["multipliers"] = {"sigma": m_sigma, "w": m_w, "rho": m_rho, "d": m_d}
    if m < cfg.m_min:
        verdict, m_out = "SKIP", 0.0
        reasons.append((f"multiplier below floor m_min={cfg.m_min}", m))
    elif m == 1.0:
        verdict, m_out = "TAKE", 1.0
    else:
        verdict, m_out = "DOWNSIZE", m
    return Verdict(verdict, m_out, S, reasons, D.index[-1].date().isoformat(), cfg.version)
