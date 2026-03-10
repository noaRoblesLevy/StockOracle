"""
model.py — Shared prediction helpers used by both app.py and portfolio.py.

Functions
---------
_shrink(correct, total)             Bayesian shrinkage toward 50 %
_lr_predict_fast(train_vals, h)     LR trend extrapolation (normalised X)
_ema_predict_fast(train_ser, h)     EMA 9/21 momentum extrapolation
_rsi(closes)                        14-period Wilder RSI
_bb_pct_b(closes)                   Bollinger Band %B (0=lower, 0.5=mid, 1=upper)
_macd(closes)                       MACD (12, 26, 9) — returns (macd_line, signal_line)
_volume_ratio(volumes)              Recent volume vs. 20-day average
_quick_direction(closes, horizon)   Fast directional check for multi-horizon consensus
_52w_high_factor(closes)            Momentum amplifier near 52-week high
_fast_screen(closes, horizon)       Returns (pred_pct, conf)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


# ─── Core maths ───────────────────────────────────────────────────────────────

def _shrink(correct: int, total: int, alpha: float = 8.0) -> float:
    """Bayesian shrinkage toward 50 %: (correct + α) / (total + 2α).

    With small samples the estimate is pulled strongly toward 0.5.
    As total grows the shrinkage effect diminishes.
    """
    if total == 0:
        return 0.5
    return round((correct + alpha) / (total + 2 * alpha), 4)


def _lr_predict_fast(train_vals, horizon_days: int) -> float:
    """Linear regression trend extrapolation with normalised X axis."""
    n  = len(train_vals)
    X  = np.arange(n, dtype=float).reshape(-1, 1)
    mn, mx = X.min(), X.max()
    Xs = (X - mn) / (mx - mn + 1e-9)
    lr = LinearRegression().fit(Xs, np.asarray(train_vals, dtype=float))
    fX = (np.arange(n, n + horizon_days, dtype=float).reshape(-1, 1) - mn) / (mx - mn + 1e-9)
    return float(lr.predict(fX)[-1])


def _ema_predict_fast(train_ser: pd.Series, horizon_days: int) -> float:
    """EMA 9/21 momentum extrapolation with damped forward projection."""
    es   = float(train_ser.ewm(span=9,  adjust=False).mean().iloc[-1])
    el   = float(train_ser.ewm(span=21, adjust=False).mean().iloc[-1])
    base = float(train_ser.iloc[-1])
    momentum = (es - el) / base if base != 0 else 0
    current = base
    for i in range(horizon_days):
        current += momentum * base * 0.1 * np.exp(-i * 0.05)
    return current


# ─── Technical indicators ─────────────────────────────────────────────────────

def _rsi(closes: pd.Series, period: int = 14) -> float:
    """Wilder's RSI.  Returns 50.0 when there is insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    delta = closes.diff().dropna()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    # When both gain and loss are effectively zero (flat market), return neutral
    if float(gain.iloc[-1]) < 1e-10 and float(loss.iloc[-1]) < 1e-10:
        return 50.0
    # Replace zero loss with tiny value so rs → ∞ (RSI → 100) when all gains
    rs = gain / loss.replace(0, np.nan).fillna(1e-9)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def _bb_pct_b(closes: pd.Series, period: int = 20) -> float:
    """%B Bollinger Band indicator.

    Returns a value in [0, 1]:
      0   = price at or below the lower band (2σ below mean)
      0.5 = price at the middle band (20-day mean)
      1   = price at or above the upper band (2σ above mean)
    """
    if len(closes) < period:
        return 0.5
    mid = closes.rolling(period).mean().iloc[-1]
    std = closes.rolling(period).std().iloc[-1]
    if std == 0:
        return 0.5
    last = float(closes.iloc[-1])
    return float(np.clip((last - (mid - 2 * std)) / (4 * std), 0.0, 1.0))


def _macd(closes: pd.Series) -> tuple[float, float]:
    """MACD (12, 26, 9) — returns (macd_line, signal_line).

    Positive histogram (macd > signal) = bullish momentum.
    Requires at least 35 data points; returns (0, 0) when insufficient.
    """
    if len(closes) < 35:
        return 0.0, 0.0
    ema12     = closes.ewm(span=12, adjust=False).mean()
    ema26     = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal    = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal.iloc[-1])


def _volume_ratio(volumes: pd.Series, period: int = 20) -> float:
    """Recent volume vs. rolling average.

    > 1.0 = above-average volume (signal conviction)
    < 1.0 = below-average volume (low conviction)
    Returns 1.0 when data is unavailable.
    """
    if volumes is None or len(volumes) < period + 1:
        return 1.0
    avg = float(volumes.rolling(period).mean().iloc[-1])
    if avg == 0:
        return 1.0
    return float(volumes.iloc[-1]) / avg


def _quick_direction(closes: pd.Series, horizon: int) -> int:
    """Fast directional check — no backtest, no overlays.

    Used for multi-horizon consensus: calling _fast_screen three times would
    triple compute time due to the 12-window mini-backtest.  This is ~50× faster
    and only cares about direction (bullish/neutral/bearish).

    Returns: +1 (both LR and EMA agree bullish), -1 (both bearish), 0 (split).
    """
    if len(closes) < 30:
        return 0
    last     = float(closes.iloc[-1])
    lr_up    = _lr_predict_fast(closes.values, horizon) >= last
    ema_up   = _ema_predict_fast(closes, horizon) >= last
    if lr_up and ema_up:   return  1
    if not lr_up and not ema_up: return -1
    return 0   # models disagree — treat as neutral


def _52w_high_factor(closes: pd.Series) -> float:
    """Momentum amplifier based on 52-week high proximity.

    Stocks trading within 5 % of their 52-week high tend to break out
    (price-momentum anomaly).  Stocks more than 30 % below the high are
    in a structural downtrend and receive a mild damper.

    Returns a multiplier to apply to pred_pct.
    """
    if len(closes) < 20:
        return 1.0
    high_52w = float(closes.tail(252).max())
    last     = float(closes.iloc[-1])
    if high_52w == 0:
        return 1.0
    pct_from_high = (high_52w - last) / high_52w
    if pct_from_high <= 0.05:   return 1.10   # near high → momentum breakout
    if pct_from_high >= 0.30:   return 0.92   # deep downtrend → dampen
    return 1.0


# ─── Composite screen ─────────────────────────────────────────────────────────

def _fast_screen(closes: pd.Series, horizon: int, volumes: pd.Series = None):
    """Return (pred_pct, conf) or (None, None) when there is insufficient data.

    Signal pipeline
    ---------------
    1. LR trend extrapolation  (60 % weight when both models agree)
    2. EMA 9/21 momentum       (40 % weight when both models agree)
    3. RSI mean-reversion overlay  — amplifies oversold, dampens overbought
    4. Bollinger Band %B overlay   — dampens when price is at an extreme band
    5. MACD momentum overlay       — amplifies bullish/bearish crossover
    6. Volume confirmation         — dampens low-conviction low-volume moves
    7. 12-window mini-backtest → Bayesian-shrunk directional confidence
    """
    closes = closes.dropna()
    if len(closes) < 30:
        return None, None

    last = float(closes.iloc[-1])

    # ── Primary trend signals ──────────────────────────────────────────────────
    lr_pred  = _lr_predict_fast(closes.values, horizon)
    lr_pct   = (lr_pred - last) / last * 100

    ema_pred = _ema_predict_fast(closes, horizon)
    ema_pct  = (ema_pred - last) / last * 100

    agree    = (lr_pct > 0) == (ema_pct > 0)
    pred_pct = lr_pct * 0.6 + ema_pct * 0.4 if agree else lr_pct

    # ── RSI mean-reversion overlay ─────────────────────────────────────────────
    rsi = _rsi(closes)
    if   rsi < 30:  rsi_mult = 1.20   # strong oversold  → amplify bullish signal
    elif rsi < 40:  rsi_mult = 1.08
    elif rsi > 75:  rsi_mult = 0.70   # strong overbought → dampen bullish signal
    elif rsi > 65:  rsi_mult = 0.88
    else:           rsi_mult = 1.0
    # For a bearish prediction the RSI roles flip
    pred_pct = pred_pct * rsi_mult if pred_pct >= 0 else pred_pct * (2.0 - rsi_mult)

    # ── Bollinger Band %B overlay ──────────────────────────────────────────────
    bb_b = _bb_pct_b(closes)
    if   pred_pct > 0 and bb_b > 0.85:  pred_pct *= 0.85  # overbought, dampen bull
    elif pred_pct < 0 and bb_b < 0.15:  pred_pct *= 0.85  # oversold,   dampen bear

    # ── MACD momentum overlay ──────────────────────────────────────────────────
    macd_line, macd_signal = _macd(closes)
    if macd_line > macd_signal:    pred_pct *= 1.10   # bullish crossover — amplify
    elif macd_line < macd_signal:  pred_pct *= 0.90   # bearish crossover — dampen

    # ── 52-week high momentum overlay ──────────────────────────────────────────
    pred_pct *= _52w_high_factor(closes)

    # ── Volume confirmation overlay ────────────────────────────────────────────
    # Low volume moves have less conviction; don't dampen already-weak predictions
    if volumes is not None and abs(pred_pct) > 0.1:
        vol_ratio = _volume_ratio(volumes)
        if vol_ratio < 0.7:
            pred_pct *= 0.85   # unusually quiet — reduce conviction

    # ── 12-window mini-backtest with Bayesian shrinkage ───────────────────────
    n_win, train_w = 12, 40
    correct = total = 0
    if len(closes) >= train_w + n_win + horizon:
        for i in range(n_win):
            end       = len(closes) - n_win - horizon + i
            tr        = closes.iloc[end - train_w:end]
            actual_up = float(closes.iloc[end + horizon - 1]) >= float(tr.iloc[-1])
            base      = float(tr.iloc[-1])
            for pred_fn in (_lr_predict_fast, _ema_predict_fast):
                try:
                    arg = tr.values if pred_fn is _lr_predict_fast else tr
                    p   = pred_fn(arg, horizon)
                    correct += int((p >= base) == actual_up)
                    total   += 1
                except Exception:
                    pass

    conf = _shrink(correct, total)
    return round(pred_pct, 2), conf
