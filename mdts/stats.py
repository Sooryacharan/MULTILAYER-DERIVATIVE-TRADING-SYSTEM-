"""Pure statistics used by the judgement layer (paper Section 4). No I/O, no hidden state."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def ramp(x: float, a: float, b: float) -> float:
    """Eq. 4: 1 for x<=a, falling linearly to 0 at x>=b. NaN input -> 0 (fail safe)."""
    if x is None or math.isnan(x):
        return 0.0
    if b == a:
        return 1.0 if x <= a else 0.0
    return min(1.0, max(0.0, (b - x) / (b - a)))


def realised_vol(log_returns: np.ndarray) -> float:
    """Eq. 2: annualised sample stdev of daily log returns."""
    if len(log_returns) < 2:
        return float("nan")
    return math.sqrt(252.0) * float(np.std(log_returns, ddof=1))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> float:
    """Wilder ATR over n sessions (uses previous close for true range)."""
    if len(close) < n + 1:
        return float("nan")
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    a = float(np.mean(tr[:n]))
    for v in tr[n:]:
        a = (a * (n - 1) + v) / n
    return a


def vol_percentile(close: np.ndarray, vol_window: int, pct_window: int) -> float:
    """Eq. 3: percentile rank of the latest rolling vol within the last pct_window vols."""
    logret = np.diff(np.log(close))[-(pct_window + vol_window):]
    if len(logret) < vol_window + 1:
        return float("nan")
    vols = np.array([realised_vol(logret[i - vol_window:i])
                     for i in range(vol_window, len(logret) + 1)])
    vols = vols[-pct_window:]
    return float(np.mean(vols <= vols[-1]))


def wilson_interval(wins: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Eq. 5: Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    q = wins / n
    denom = 1 + z * z / n
    centre = q + z * z / (2 * n)
    half = z * math.sqrt(q * (1 - q) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def bootstrap_expectancy(net: np.ndarray, B: int, alpha: float, seed: int) -> tuple[float, float, float]:
    """Eq. 6 + percentile bootstrap (Efron). Returns (E, E_lo, E_hi). Fixed seed => reproducible."""
    net = np.asarray(net, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(net), size=(B, len(net)))
    means = net[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(net.mean()), float(lo), float(hi)


def payoff_decomposition(net: np.ndarray) -> dict:
    """q, mean win, mean absolute loss, payoff ratio b."""
    net = np.asarray(net, dtype=float)
    wins, losses = net[net > 0], net[net <= 0]
    q = len(wins) / len(net) if len(net) else float("nan")
    w = float(wins.mean()) if len(wins) else 0.0
    l = float(-losses.mean()) if len(losses) else 0.0
    return {"q": q, "avg_win": w, "avg_loss": l, "b": (w / l) if l > 0 else float("inf")}


def distribution_summary(r: np.ndarray) -> dict:
    """Section 4.2: mean, median, tail percentiles, skew, excess kurtosis."""
    r = np.asarray(r, dtype=float)
    mu, sd = r.mean(), r.std(ddof=0)
    if sd > 0:
        skew = float(np.mean(((r - mu) / sd) ** 3))
        kurt = float(np.mean(((r - mu) / sd) ** 4) - 3.0)
    else:
        skew = kurt = 0.0
    p5, p25, p75, p95 = np.quantile(r, [0.05, 0.25, 0.75, 0.95])
    return {"mean": float(mu), "median": float(np.median(r)), "p5": float(p5), "p25": float(p25),
            "p75": float(p75), "p95": float(p95), "skew": skew, "excess_kurtosis": kurt}


def rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> float:
    """Pearson correlation over the last `window` aligned observations; NaN if insufficient/degenerate."""
    n = min(len(x), len(y), window)
    if n < window:
        return float("nan")
    xs, ys = np.asarray(x[-n:], float), np.asarray(y[-n:], float)
    if xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def max_drawdown(equity: np.ndarray) -> float:
    """Max of (peak - V)/peak. Equity must be positive (use capital + cumulative pnl)."""
    equity = np.asarray(equity, dtype=float)
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def current_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    if len(equity) == 0:
        return 0.0
    return float((equity.max() - equity[-1]) / equity.max())
