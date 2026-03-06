"""
portfolio.py — Virtual portfolio: state management + daily rebalance logic.

Strategy
--------
- Screen RANK_TICKERS with a fast LR+EMA model (Bayesian-shrunk backtest accuracy)
- Hold up to MAX_POSITIONS stocks; each ≤ MAX_POSITION_PCT of total portfolio value
- Only enter a position when predicted_pct > MIN_PRED_PCT and confidence > MIN_CONF
- Score positions by (pred_pct × confidence); sell when score turns negative or
  stock drops out of the top MAX_POSITIONS candidates
- Record every trade and a daily equity snapshot

Run daily after market close (Mon–Fri).
"""
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), 'portfolio.json')
INITIAL_BALANCE = 50_000.0
MAX_POSITIONS    = 5
MAX_POSITION_PCT = 0.20   # max 20 % of portfolio per stock
MIN_CONF         = 0.54   # Bayesian-shrunk confidence threshold
MIN_PRED_PCT     = 0.3    # minimum predicted % gain to enter

RANK_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL', 'AMZN', 'TSLA', 'AVGO',
    'AMD',  'PLTR', 'APP',  'CRWD', 'NET',   'DDOG', 'PANW', 'CRM',
    'NFLX', 'COIN', 'SMCI', 'ARM',
    'SPY',  'QQQ',  'SMH',  'ARKK',
    'JPM',  'GS',   'BAC',  'LLY',  'XOM',   'GLD',
]

# ─── Helpers (duplicated from app.py to keep this module self-contained) ──────

def _shrink(correct: int, total: int, alpha: float = 8.0) -> float:
    if total == 0:
        return 0.5
    return round((correct + alpha) / (total + 2 * alpha), 4)


def _lr_predict_fast(train_vals, horizon_days: int) -> float:
    n = len(train_vals)
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = train_vals.astype(float)
    m = LinearRegression().fit(X, y)
    return float(m.predict([[n + horizon_days - 1]])[0])


def _ema_predict_fast(train_ser: pd.Series, horizon_days: int) -> float:
    es = float(train_ser.ewm(span=9,  adjust=False).mean().iloc[-1])
    el = float(train_ser.ewm(span=21, adjust=False).mean().iloc[-1])
    momentum = (es - el) / el if el != 0 else 0
    return float(train_ser.iloc[-1]) * (1 + momentum * horizon_days / 21)


def _fast_screen(closes: pd.Series, horizon: int):
    """Returns (pred_pct, conf) or (None, None) if not enough data."""
    closes = closes.dropna()
    if len(closes) < 30:
        return None, None

    last = float(closes.iloc[-1])

    lr_pred  = _lr_predict_fast(closes.values, horizon)
    lr_pct   = (lr_pred - last) / last * 100

    ema_pred = _ema_predict_fast(closes, horizon)
    ema_pct  = (ema_pred - last) / last * 100

    agree    = (lr_pct > 0) == (ema_pct > 0)
    pred_pct = lr_pct * 0.6 + ema_pct * 0.4 if agree else lr_pct

    # 12-window mini-backtest with Bayesian shrinkage
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


# ─── Portfolio state ──────────────────────────────────────────────────────────

def _initial_portfolio() -> dict:
    today = datetime.now().strftime('%Y-%m-%d')
    return {
        'initial_balance': INITIAL_BALANCE,
        'cash':            INITIAL_BALANCE,
        'positions':       {},   # {symbol: {shares, entry_price, entry_date, entry_conf, entry_pred_pct}}
        'trades':          [],   # list of trade records
        'daily_values':    [{'date': today, 'value': INITIAL_BALANCE,
                              'cash': INITIAL_BALANCE, 'positions_value': 0.0}],
        'last_updated':    None,
        'created':         today,
    }


def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    p = _initial_portfolio()
    save_portfolio(p)
    return p


def save_portfolio(portfolio: dict) -> None:
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)


# ─── Portfolio value ──────────────────────────────────────────────────────────

def portfolio_value(portfolio: dict, price_map: dict) -> float:
    """Sum of cash + current market value of all open positions."""
    pos_val = sum(
        pos['shares'] * price_map.get(sym, pos['entry_price'])
        for sym, pos in portfolio['positions'].items()
    )
    return portfolio['cash'] + pos_val


# ─── Daily rebalance ──────────────────────────────────────────────────────────

def rebalance(portfolio: dict, horizon: str = 'week') -> tuple[dict, str]:
    """
    Download latest prices, score all RANK_TICKERS, sell losers, buy winners.
    Returns (updated_portfolio, summary_message).
    """
    horizon_map = {'day': 1, 'week': 5, 'month': 21}
    horizon_days = horizon_map.get(horizon, 5)
    today = datetime.now().strftime('%Y-%m-%d')
    weekday = datetime.now().weekday()

    if weekday >= 5:
        return portfolio, f"Weekend ({today}), no rebalance."

    if portfolio.get('last_updated') == today:
        return portfolio, f"Already rebalanced today ({today})."

    # ── Batch download 3-month history ────────────────────────────────────────
    try:
        raw = yf.download(
            RANK_TICKERS, period='3mo', interval='1d',
            auto_adjust=True, progress=False, group_by='ticker'
        )
    except Exception as exc:
        return portfolio, f"Download failed: {exc}"

    # ── Score every ticker ────────────────────────────────────────────────────
    candidates = []
    price_map  = {}
    for sym in RANK_TICKERS:
        try:
            closes = (raw[sym]['Close'] if isinstance(raw.columns, pd.MultiIndex)
                      else raw['Close']).dropna()
            if closes.empty:
                continue
            current_price = float(closes.iloc[-1])
            price_map[sym] = current_price

            pred_pct, conf = _fast_screen(closes, horizon_days)
            if pred_pct is None:
                continue

            score = pred_pct * conf if (pred_pct > MIN_PRED_PCT and conf > MIN_CONF) else -999
            candidates.append({
                'symbol':   sym,
                'pred_pct': pred_pct,
                'conf':     conf,
                'price':    current_price,
                'score':    score,
            })
        except Exception:
            continue

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top_picks = [c for c in candidates if c['score'] > 0][:MAX_POSITIONS]
    top_syms  = {c['symbol'] for c in top_picks}

    # ── Sell: positions no longer in top picks or turned bearish ──────────────
    sells = 0
    for sym in list(portfolio['positions'].keys()):
        pos   = portfolio['positions'][sym]
        price = price_map.get(sym, pos['entry_price'])
        if sym not in top_syms:
            proceeds = pos['shares'] * price
            pnl      = proceeds - pos['shares'] * pos['entry_price']
            portfolio['cash'] += proceeds
            reason = ('Signal bearish' if sym in {c['symbol'] for c in candidates
                                                   if c['pred_pct'] <= 0}
                      else 'Dropped from top picks')
            portfolio['trades'].append({
                'date':   today,
                'ticker': sym,
                'action': 'SELL',
                'shares': pos['shares'],
                'price':  round(price, 2),
                'value':  round(proceeds, 2),
                'pnl':    round(pnl, 2),
                'reason': reason,
            })
            del portfolio['positions'][sym]
            sells += 1

    # ── Buy: top picks not already held ───────────────────────────────────────
    total_val    = portfolio_value(portfolio, price_map)
    max_pos_size = total_val * MAX_POSITION_PCT
    buys = 0

    for cand in top_picks:
        sym = cand['symbol']
        if sym in portfolio['positions']:
            continue
        if len(portfolio['positions']) >= MAX_POSITIONS:
            break
        if portfolio['cash'] < 50:
            break

        alloc  = min(max_pos_size, portfolio['cash'])
        shares = int(alloc / cand['price'])
        if shares < 1:
            continue

        cost = shares * cand['price']
        portfolio['cash'] -= cost
        portfolio['positions'][sym] = {
            'shares':          shares,
            'entry_price':     round(cand['price'], 2),
            'entry_date':      today,
            'entry_conf':      round(cand['conf'], 4),
            'entry_pred_pct':  round(cand['pred_pct'], 2),
        }
        portfolio['trades'].append({
            'date':       today,
            'ticker':     sym,
            'action':     'BUY',
            'shares':     shares,
            'price':      round(cand['price'], 2),
            'value':      round(cost, 2),
            'pnl':        None,
            'pred_pct':   round(cand['pred_pct'], 2),
            'confidence': round(cand['conf'], 4),
            'reason':     f'Top pick #{top_picks.index(cand)+1}, conf={cand["conf"]:.0%}, pred={cand["pred_pct"]:+.1f}%',
        })
        buys += 1

    # ── Record daily value snapshot ───────────────────────────────────────────
    total_val     = portfolio_value(portfolio, price_map)
    positions_val = total_val - portfolio['cash']
    portfolio['daily_values'].append({
        'date':             today,
        'value':            round(total_val, 2),
        'cash':             round(portfolio['cash'], 2),
        'positions_value':  round(positions_val, 2),
    })
    portfolio['last_updated'] = today

    summary = (f"{today}: sold {sells}, bought {buys}, "
               f"positions={len(portfolio['positions'])}, "
               f"total=${total_val:,.0f}")
    return portfolio, summary
