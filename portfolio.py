"""
portfolio.py — Virtual portfolio: state management + daily rebalance logic.

Strategy
--------
- Screen RANK_TICKERS (~60 liquid names) with the shared model._fast_screen
- Hold up to MAX_POSITIONS stocks; each ≤ MAX_POSITION_PCT of total value
- Sector cap: max MAX_PER_SECTOR non-ETF positions from the same sector
- Only enter when pred_pct > MIN_PRED_PCT and confidence > MIN_CONF
- Position sizing is inverse-volatility weighted (lower-vol names get more)
- Stop-loss: sell any position that falls > STOP_LOSS_PCT from entry
- On signal sell: remove positions that dropped out of the top picks
- Record every trade and a daily equity snapshot (inc. SPY close for benchmark)

Run daily after market close (Mon–Fri), skips weekends and US holidays.
"""

import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from model import _fast_screen, _shrink

warnings.filterwarnings('ignore')

# ─── Constants ────────────────────────────────────────────────────────────────

PORTFOLIO_FILE   = os.path.join(os.path.dirname(__file__), 'portfolio.json')
INITIAL_BALANCE  = 50_000.0
MAX_POSITIONS    = 5
MAX_POSITION_PCT = 0.20    # hard cap: no single position > 20 % of portfolio
MIN_CONF         = 0.54    # Bayesian-shrunk confidence threshold
MIN_PRED_PCT     = 0.3     # minimum predicted % gain to enter
STOP_LOSS_PCT    = 0.08    # sell if position falls > 8 % below entry
MAX_PER_SECTOR   = 2       # max non-ETF positions from the same sector

# ── NYSE market holidays (observed dates) ─────────────────────────────────────
_HOLIDAYS = {
    # 2025
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18',
    '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01',
    '2025-11-27', '2025-12-25',
    # 2026
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03',
    '2026-05-25', '2026-06-19', '2026-07-03', '2026-09-07',
    '2026-11-26', '2026-12-25',
}

# ── Sector map for concentration limits ───────────────────────────────────────
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'META': 'Technology', 'GOOGL': 'Technology', 'AVGO': 'Technology',
    'AMD':  'Technology', 'PLTR': 'Technology',  'APP':  'Technology',
    'CRWD': 'Technology', 'NET':  'Technology',  'DDOG': 'Technology',
    'PANW': 'Technology', 'CRM':  'Technology',  'SMCI': 'Technology',
    'ARM':  'Technology', 'ADBE': 'Technology',  'NOW':  'Technology',
    'ORCL': 'Technology', 'INTU': 'Technology',  'SNOW': 'Technology',
    'ZS':   'Technology', 'OKTA': 'Technology',  'MDB':  'Technology',
    'WDAY': 'Technology', 'FTNT': 'Technology',  'TEAM': 'Technology',
    'SHOP': 'Technology', 'TTD':  'Technology',  'INTC': 'Technology',
    'QCOM': 'Technology', 'TXN':  'Technology',  'MU':   'Technology',
    'LRCX': 'Technology', 'AMAT': 'Technology',  'KLAC': 'Technology',
    # Consumer Discretionary
    'TSLA': 'Consumer',   'NFLX': 'Consumer',    'AMZN': 'Consumer',
    'MELI': 'Consumer',   'BABA': 'Consumer',
    # Financials
    'JPM':  'Financials', 'GS':   'Financials',  'BAC':  'Financials',
    'V':    'Financials', 'MA':   'Financials',  'COIN': 'Financials',
    'PYPL': 'Financials', 'SQ':   'Financials',
    # Healthcare
    'LLY':  'Healthcare', 'UNH':  'Healthcare',  'ABBV': 'Healthcare',
    'MRNA': 'Healthcare', 'AMGN': 'Healthcare',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy',
    # Commodities
    'GLD': 'Commodities', 'SLV': 'Commodities',
    # ETFs — no sector cap applies
    'SPY':  'ETF', 'QQQ':  'ETF', 'SMH':  'ETF', 'ARKK': 'ETF',
    'IWM':  'ETF', 'DIA':  'ETF', 'VTI':  'ETF', 'SOXX': 'ETF',
    'XLE':  'ETF', 'XLK':  'ETF', 'XLF':  'ETF', 'TLT':  'ETF',
}

# ── Screening universe ────────────────────────────────────────────────────────
RANK_TICKERS = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL', 'AMZN', 'AVGO',
    # High-growth tech
    'TSLA', 'AMD',  'PLTR', 'APP',  'CRWD', 'NET',  'DDOG', 'PANW',
    'CRM',  'NFLX', 'COIN', 'SMCI', 'ARM',  'ADBE', 'NOW',  'INTU',
    'SNOW', 'MDB',  'OKTA', 'ZS',   'WDAY', 'SHOP', 'TTD',  'FTNT',
    # Semiconductors
    'MU',   'QCOM', 'INTC', 'TXN',  'LRCX', 'AMAT', 'KLAC',
    # Broad market ETFs
    'SPY',  'QQQ',  'IWM',  'DIA',  'VTI',
    # Sector ETFs
    'SMH',  'SOXX', 'XLK',  'XLF',  'XLE',  'ARKK', 'TLT',
    # Financials
    'JPM',  'GS',   'BAC',  'V',    'MA',   'PYPL',
    # Healthcare
    'LLY',  'UNH',  'ABBV', 'AMGN',
    # Energy & commodities
    'XOM',  'CVX',  'GLD',
    # Consumer
    'MELI', 'BABA',
]

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _vol_weights(picks: list, closes_map: dict) -> dict:
    """Inverse-volatility weights: lower-volatility names get a larger share."""
    vols = {}
    for c in picks:
        sym    = c['symbol']
        closes = closes_map.get(sym)
        if closes is not None and len(closes) >= 10:
            ret = closes.pct_change().dropna().tail(20)
            vols[sym] = max(float(ret.std()), 0.005)
        else:
            vols[sym] = 0.02        # default 2 % daily vol
    inv   = {sym: 1.0 / v for sym, v in vols.items()}
    total = sum(inv.values()) or 1.0
    return {sym: v / total for sym, v in inv.items()}


def _sector_limited_picks(candidates: list, max_positions: int,
                          max_per_sector: int) -> list:
    """Return top candidates while capping non-ETF positions per sector."""
    picks         = []
    sector_counts = {}
    for cand in candidates:
        if len(picks) >= max_positions:
            break
        if cand['score'] <= 0:
            break
        sector = SECTOR_MAP.get(cand['symbol'], 'Unknown')
        # ETFs are exempt — they already represent broad diversification
        if sector == 'ETF' or sector_counts.get(sector, 0) < max_per_sector:
            picks.append(cand)
            if sector != 'ETF':
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
    return picks


# ─── Portfolio state ──────────────────────────────────────────────────────────

def _initial_portfolio() -> dict:
    today = datetime.now().strftime('%Y-%m-%d')
    return {
        'initial_balance': INITIAL_BALANCE,
        'cash':            INITIAL_BALANCE,
        'positions':       {},   # {symbol: {shares, entry_price, entry_date, ...}}
        'trades':          [],   # list of trade records
        'daily_values':    [{'date': today, 'value': INITIAL_BALANCE,
                              'cash': INITIAL_BALANCE, 'positions_value': 0.0,
                              'spy_close': None}],
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
    """Cash + current market value of all open positions."""
    pos_val = sum(
        pos['shares'] * price_map.get(sym, pos['entry_price'])
        for sym, pos in portfolio['positions'].items()
    )
    return portfolio['cash'] + pos_val


# ─── Daily rebalance ──────────────────────────────────────────────────────────

def rebalance(portfolio: dict, horizon: str = 'week') -> tuple[dict, str]:
    """
    Download latest prices, score RANK_TICKERS, apply stop-losses,
    sell dropped/bearish picks, and buy new top picks with vol-weighted sizing.
    Returns (updated_portfolio, summary_message).
    """
    horizon_map  = {'day': 1, 'week': 5, 'month': 21}
    horizon_days = horizon_map.get(horizon, 5)
    today        = datetime.now().strftime('%Y-%m-%d')
    weekday      = datetime.now().weekday()

    if weekday >= 5:
        return portfolio, f'Weekend ({today}), no rebalance.'
    if today in _HOLIDAYS:
        return portfolio, f'Market holiday ({today}), no rebalance.'
    if portfolio.get('last_updated') == today:
        return portfolio, f'Already rebalanced today ({today}).'

    # ── Batch download 3-month history ────────────────────────────────────────
    try:
        raw = yf.download(
            RANK_TICKERS, period='3mo', interval='1d',
            auto_adjust=True, progress=False, group_by='ticker',
        )
    except Exception as exc:
        return portfolio, f'Download failed: {exc}'

    # ── Build closes_map and price_map ────────────────────────────────────────
    closes_map = {}
    price_map  = {}
    for sym in RANK_TICKERS:
        try:
            closes = (raw[sym]['Close'] if isinstance(raw.columns, pd.MultiIndex)
                      else raw['Close']).dropna()
            if not closes.empty:
                closes_map[sym] = closes
                price_map[sym]  = float(closes.iloc[-1])
        except Exception:
            continue

    # ── Score every ticker ────────────────────────────────────────────────────
    candidates = []
    for sym, closes in closes_map.items():
        try:
            pred_pct, conf = _fast_screen(closes, horizon_days)
            if pred_pct is None:
                continue
            score = pred_pct * conf if (pred_pct > MIN_PRED_PCT and conf > MIN_CONF) else -999
            candidates.append({
                'symbol':   sym,
                'pred_pct': pred_pct,
                'conf':     conf,
                'price':    price_map[sym],
                'score':    score,
            })
        except Exception:
            continue

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top_picks = _sector_limited_picks(candidates, MAX_POSITIONS, MAX_PER_SECTOR)
    top_syms  = {c['symbol'] for c in top_picks}
    bearish   = {c['symbol'] for c in candidates if c['pred_pct'] <= 0}

    # ── Stop-loss sells ───────────────────────────────────────────────────────
    stop_sells = 0
    for sym in list(portfolio['positions'].keys()):
        pos      = portfolio['positions'][sym]
        price    = price_map.get(sym, pos['entry_price'])
        loss_pct = (pos['entry_price'] - price) / pos['entry_price']
        if loss_pct >= STOP_LOSS_PCT:
            proceeds = pos['shares'] * price
            pnl      = proceeds - pos['shares'] * pos['entry_price']
            portfolio['cash'] += proceeds
            portfolio['trades'].append({
                'date':   today,
                'ticker': sym,
                'action': 'SELL',
                'shares': pos['shares'],
                'price':  round(price, 2),
                'value':  round(proceeds, 2),
                'pnl':    round(pnl, 2),
                'reason': f'Stop-loss: -{loss_pct:.1%} from entry',
            })
            del portfolio['positions'][sym]
            top_syms.discard(sym)   # don't re-buy the same day
            stop_sells += 1

    # ── Signal sells (dropped from top picks or turned bearish) ───────────────
    signal_sells = 0
    for sym in list(portfolio['positions'].keys()):
        pos   = portfolio['positions'][sym]
        price = price_map.get(sym, pos['entry_price'])
        if sym not in top_syms:
            proceeds = pos['shares'] * price
            pnl      = proceeds - pos['shares'] * pos['entry_price']
            portfolio['cash'] += proceeds
            portfolio['trades'].append({
                'date':   today,
                'ticker': sym,
                'action': 'SELL',
                'shares': pos['shares'],
                'price':  round(price, 2),
                'value':  round(proceeds, 2),
                'pnl':    round(pnl, 2),
                'reason': 'Signal bearish' if sym in bearish else 'Dropped from top picks',
            })
            del portfolio['positions'][sym]
            signal_sells += 1

    sells = stop_sells + signal_sells

    # ── Buy: top picks not already held (inverse-volatility sizing) ──────────
    total_val = portfolio_value(portfolio, price_map)
    weights   = _vol_weights(top_picks, closes_map)
    buys      = 0

    for cand in top_picks:
        sym = cand['symbol']
        if sym in portfolio['positions']:
            continue
        if len(portfolio['positions']) >= MAX_POSITIONS:
            break
        if portfolio['cash'] < 50:
            break

        weight = weights.get(sym, 1 / max(len(top_picks), 1))
        alloc  = min(total_val * weight, total_val * MAX_POSITION_PCT, portfolio['cash'])
        shares = int(alloc / cand['price'])
        if shares < 1:
            continue

        cost = shares * cand['price']
        portfolio['cash'] -= cost
        portfolio['positions'][sym] = {
            'shares':         shares,
            'entry_price':    round(cand['price'], 2),
            'entry_date':     today,
            'entry_conf':     round(cand['conf'], 4),
            'entry_pred_pct': round(cand['pred_pct'], 2),
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
            'reason':     (f'Top pick #{top_picks.index(cand)+1}, '
                           f'conf={cand["conf"]:.0%}, pred={cand["pred_pct"]:+.1f}%, '
                           f'alloc={weight:.0%}'),
        })
        buys += 1

    # ── Daily value snapshot (includes SPY close for benchmark overlay) ───────
    total_val     = portfolio_value(portfolio, price_map)
    positions_val = total_val - portfolio['cash']
    spy_close     = price_map.get('SPY')

    portfolio['daily_values'].append({
        'date':            today,
        'value':           round(total_val, 2),
        'cash':            round(portfolio['cash'], 2),
        'positions_value': round(positions_val, 2),
        'spy_close':       round(spy_close, 4) if spy_close else None,
    })
    portfolio['last_updated'] = today

    summary = (
        f'{today}: stop-loss {stop_sells}, signal-sell {signal_sells}, '
        f'bought {buys}, positions={len(portfolio["positions"])}, '
        f'total=${total_val:,.0f}'
    )
    return portfolio, summary
