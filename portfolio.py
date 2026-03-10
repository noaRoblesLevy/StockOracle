"""
portfolio.py — Virtual portfolio: state management + daily rebalance logic.

Strategy
--------
- Screen RANK_TICKERS (~60 liquid names) with the shared model._fast_screen
- Hold up to MAX_POSITIONS stocks; each ≤ MAX_POSITION_PCT of total value
- Sector cap: max MAX_PER_SECTOR non-ETF positions from the same sector
- Only enter when pred_pct > MIN_PRED_PCT and confidence > MIN_CONF
- Position sizing is inverse-volatility weighted + correlation-adjusted
- Trailing stop-loss: sell any position that falls > STOP_LOSS_PCT from its peak
- Take-profit: sell when actual gain reaches the predicted gain at entry
- Portfolio circuit breaker: liquidate all positions if value drops > MAX_DRAWDOWN_PCT from ATH
- Earnings avoidance: skip buying within N_EARNINGS_DAYS of a scheduled earnings date
- On signal sell: remove positions that dropped out of the top picks
- Record every trade and a daily equity snapshot (inc. SPY close for benchmark)

Run daily after market close (Mon–Fri), skips weekends and US holidays.
"""

import json
import logging
import os
import time as _time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from model import _fast_screen, _shrink, _quick_direction

warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

PORTFOLIO_FILE   = os.path.join(os.path.dirname(__file__), 'portfolio.json')
INITIAL_BALANCE  = 50_000.0
MAX_POSITIONS    = 5
MAX_POSITION_PCT = 0.20    # hard cap: no single position > 20 % of portfolio
MIN_CONF         = 0.54    # Bayesian-shrunk confidence threshold
MIN_PRED_PCT     = 0.3     # minimum predicted % gain to enter
STOP_LOSS_PCT    = 0.08    # sell if position falls > 8 % below its peak (trailing)
MAX_PER_SECTOR   = 2       # max non-ETF positions from the same sector
MAX_DRAWDOWN_PCT = 0.15    # circuit breaker: liquidate all if portfolio down 15 % from ATH
N_EARNINGS_DAYS  = 3       # skip buying if earnings scheduled within this many calendar days

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

def _yf_download_with_retry(tickers, attempts: int = 3, **kwargs):
    """yf.download with exponential backoff — handles transient rate-limit errors."""
    last_exc = None
    for i in range(attempts):
        try:
            result = yf.download(tickers, **kwargs)
            if not result.empty:
                return result
        except Exception as exc:
            last_exc = exc
            log.warning('yf.download attempt %d/%d failed: %s', i + 1, attempts, exc)
        if i < attempts - 1:
            _time.sleep(2 ** i)   # 1 s, 2 s, …
    # Last attempt (or raise)
    if last_exc is not None:
        raise last_exc
    return yf.download(tickers, **kwargs)


def _next_earnings_date(sym: str) -> str | None:
    """Return the next scheduled earnings date (YYYY-MM-DD) or None if unknown."""
    try:
        cal = yf.Ticker(sym).calendar
        if cal is None:
            return None
        # yfinance >= 0.2: calendar is a dict with 'Earnings Date' list
        if isinstance(cal, dict):
            dates = cal.get('Earnings Date', [])
            if dates:
                return str(pd.Timestamp(dates[0]).date())
        # yfinance older: calendar is a DataFrame with 'Earnings Date' in the index
        if isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.index:
            val = cal.loc['Earnings Date'].iloc[0]
            return str(pd.Timestamp(val).date())
    except Exception:
        pass
    return None


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


def _corr_adjusted_weights(picks: list, closes_map: dict,
                           base_weights: dict) -> dict:
    """Reduce allocation for picks that are highly correlated with higher-ranked ones.

    For each pick, we compute its average absolute correlation with all picks
    ranked above it.  If that average exceeds 0.70 the weight is tapered down
    proportionally.  The floor prevents any pick from being reduced to zero.
    """
    syms = [c['symbol'] for c in picks]
    if len(syms) < 2:
        return base_weights

    rets = pd.DataFrame({
        sym: closes_map[sym].pct_change().dropna().tail(60)
        for sym in syms if sym in closes_map
    }).dropna(axis=1, how='all')

    if rets.shape[1] < 2:
        return base_weights

    corr = rets.corr()
    adjusted = {}
    for i, sym in enumerate(syms):
        w = base_weights.get(sym, 1.0 / len(syms))
        higher_ranked = syms[:i]
        for other in higher_ranked:
            try:
                r = abs(float(corr.loc[sym, other]))
            except (KeyError, TypeError):
                r = 0.0
            if r > 0.70:
                # Linear taper: r=0.70 → no reduction, r=1.0 → 40 % reduction
                w *= 1.0 - (r - 0.70) * 1.33
        adjusted[sym] = max(w, 0.05)   # floor at 5 %

    total = sum(adjusted.values()) or 1.0
    return {sym: v / total for sym, v in adjusted.items()}


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

    # ── Batch download 3-month history (with retry) ───────────────────────────
    try:
        raw = _yf_download_with_retry(
            RANK_TICKERS, period='3mo', interval='1d',
            auto_adjust=True, progress=False, group_by='ticker',
        )
    except Exception as exc:
        log.error('Download failed: %s', exc)
        return portfolio, f'Download failed: {exc}'

    # ── Build closes_map, volumes_map, and price_map ──────────────────────────
    closes_map  = {}
    volumes_map = {}
    price_map   = {}
    for sym in RANK_TICKERS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                c = raw[sym]['Close'].dropna()
                v = raw[sym].get('Volume', pd.Series()).dropna()
            else:
                c = raw['Close'].dropna()
                v = raw.get('Volume', pd.Series()).dropna()
            if not c.empty:
                closes_map[sym]  = c
                price_map[sym]   = float(c.iloc[-1])
                if not v.empty:
                    volumes_map[sym] = v
        except Exception:
            continue

    # ── Score every ticker ────────────────────────────────────────────────────
    candidates = []
    for sym, closes in closes_map.items():
        try:
            pred_pct, conf = _fast_screen(closes, horizon_days,
                                          volumes=volumes_map.get(sym))
            if pred_pct is None:
                continue

            # ── Multi-horizon consensus: require 2 of 3 horizons to agree ─────
            # _quick_direction is ~50× faster than _fast_screen (no backtest)
            day_dir   = _quick_direction(closes, 1)
            month_dir = _quick_direction(closes, 21)
            week_dir  = 1 if pred_pct > 0 else (-1 if pred_pct < 0 else 0)
            bullish_count = sum(d > 0 for d in (day_dir, week_dir, month_dir))
            bearish_count = sum(d < 0 for d in (day_dir, week_dir, month_dir))
            # Need at least 2 of 3 horizons to agree for a valid signal
            if pred_pct > 0 and bullish_count < 2:
                score = -999   # conflicting horizons — skip
            elif pred_pct > MIN_PRED_PCT and conf > MIN_CONF:
                # Sharpe-ratio-adjusted score: reward high pred/vol ratio
                ret = closes.pct_change().dropna().tail(20)
                vol = max(float(ret.std()), 0.005)
                score = (pred_pct / vol) * conf
            else:
                score = -999

            candidates.append({
                'symbol':      sym,
                'pred_pct':    pred_pct,
                'conf':        conf,
                'price':       price_map[sym],
                'score':       score,
                'horizons':    f'{day_dir:+d}/{week_dir:+d}/{month_dir:+d}',
            })
        except Exception:
            continue

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top_picks = _sector_limited_picks(candidates, MAX_POSITIONS, MAX_PER_SECTOR)
    top_syms  = {c['symbol'] for c in top_picks}
    bearish   = {c['symbol'] for c in candidates if c['pred_pct'] <= 0}

    # ── Update peak price for all open positions ───────────────────────────────
    for sym, pos in portfolio['positions'].items():
        price = price_map.get(sym, pos['entry_price'])
        current_peak = pos.get('peak_price', pos['entry_price'])
        if price > current_peak:
            pos['peak_price'] = round(price, 2)

    # ── Portfolio circuit breaker: liquidate if down > MAX_DRAWDOWN_PCT from ATH
    current_total = portfolio_value(portfolio, price_map)
    prev_ath = portfolio.get('all_time_high', portfolio['initial_balance'])
    ath = max(current_total, prev_ath)
    portfolio['all_time_high'] = round(ath, 2)

    if current_total < ath * (1 - MAX_DRAWDOWN_PCT) and portfolio['positions']:
        log.warning('Circuit breaker triggered: portfolio at $%.0f vs ATH $%.0f', current_total, ath)
        for sym in list(portfolio['positions'].keys()):
            pos      = portfolio['positions'][sym]
            price    = price_map.get(sym, pos['entry_price'])
            proceeds = pos['shares'] * price
            pnl      = proceeds - pos['shares'] * pos['entry_price']
            portfolio['cash'] += proceeds
            portfolio['trades'].append({
                'date':   today, 'ticker': sym, 'action': 'SELL',
                'shares': pos['shares'], 'price':  round(price, 2),
                'value':  round(proceeds, 2), 'pnl':    round(pnl, 2),
                'reason': f'Circuit breaker: portfolio down {(ath-current_total)/ath:.1%} from ATH',
            })
            del portfolio['positions'][sym]
        portfolio['last_updated'] = today
        total_val = portfolio_value(portfolio, price_map)
        msg = (f'{today}: CIRCUIT BREAKER — portfolio down '
               f'{(ath-current_total)/ath:.1%} from ATH ${ath:,.0f}. '
               f'Liquidated all positions. Cash=${portfolio["cash"]:,.0f}')
        log.warning(msg)
        return portfolio, msg

    # ── Trailing stop-loss sells ───────────────────────────────────────────────
    stop_sells = 0
    for sym in list(portfolio['positions'].keys()):
        pos      = portfolio['positions'][sym]
        price    = price_map.get(sym, pos['entry_price'])
        peak     = pos.get('peak_price', pos['entry_price'])
        loss_pct = (peak - price) / peak if peak > 0 else 0
        if loss_pct >= STOP_LOSS_PCT:
            proceeds = pos['shares'] * price
            pnl      = proceeds - pos['shares'] * pos['entry_price']
            portfolio['cash'] += proceeds
            portfolio['trades'].append({
                'date':   today, 'ticker': sym, 'action': 'SELL',
                'shares': pos['shares'], 'price':  round(price, 2),
                'value':  round(proceeds, 2), 'pnl':    round(pnl, 2),
                'reason': f'Trailing stop: -{loss_pct:.1%} from ${peak:.2f} peak',
            })
            log.info('Trailing stop on %s: price $%.2f, peak $%.2f, loss %.1f%%',
                     sym, price, peak, loss_pct * 100)
            del portfolio['positions'][sym]
            top_syms.discard(sym)
            stop_sells += 1

    # ── Take-profit sells ─────────────────────────────────────────────────────
    take_profit_sells = 0
    for sym in list(portfolio['positions'].keys()):
        pos      = portfolio['positions'][sym]
        price    = price_map.get(sym, pos['entry_price'])
        gain_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
        target   = pos.get('entry_pred_pct', 0)
        # Sell only when target is meaningful (≥ 1 %) and has been reached
        if target >= 1.0 and gain_pct >= target:
            proceeds = pos['shares'] * price
            pnl      = proceeds - pos['shares'] * pos['entry_price']
            portfolio['cash'] += proceeds
            portfolio['trades'].append({
                'date':   today, 'ticker': sym, 'action': 'SELL',
                'shares': pos['shares'], 'price':  round(price, 2),
                'value':  round(proceeds, 2), 'pnl':    round(pnl, 2),
                'reason': f'Take-profit: +{gain_pct:.1f}% reached target +{target:.1f}%',
            })
            log.info('Take-profit on %s: +%.1f%% vs target +%.1f%%', sym, gain_pct, target)
            del portfolio['positions'][sym]
            top_syms.discard(sym)
            take_profit_sells += 1

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
                'date':   today, 'ticker': sym, 'action': 'SELL',
                'shares': pos['shares'], 'price':  round(price, 2),
                'value':  round(proceeds, 2), 'pnl':    round(pnl, 2),
                'reason': 'Signal bearish' if sym in bearish else 'Dropped from top picks',
            })
            del portfolio['positions'][sym]
            signal_sells += 1

    sells = stop_sells + take_profit_sells + signal_sells

    # ── Buy: top picks not already held ───────────────────────────────────────
    total_val   = portfolio_value(portfolio, price_map)
    base_w      = _vol_weights(top_picks, closes_map)
    weights     = _corr_adjusted_weights(top_picks, closes_map, base_w)
    buys        = 0

    for cand in top_picks:
        sym = cand['symbol']
        if sym in portfolio['positions']:
            continue
        if len(portfolio['positions']) >= MAX_POSITIONS:
            break
        if portfolio['cash'] < 50:
            break

        # ── Earnings avoidance ────────────────────────────────────────────────
        earnings_date = _next_earnings_date(sym)
        if earnings_date:
            days_to_earnings = (pd.Timestamp(earnings_date) - pd.Timestamp(today)).days
            if 0 <= days_to_earnings <= N_EARNINGS_DAYS:
                log.info('Skipping %s — earnings in %d day(s) on %s',
                         sym, days_to_earnings, earnings_date)
                continue

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
            'peak_price':     round(cand['price'], 2),
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
        log.info('Bought %s: %d shares @ $%.2f (alloc %.0f%%)',
                 sym, shares, cand['price'], weight * 100)
        buys += 1

    # ── Position trimming: sell excess shares when position > MAX_POSITION_PCT ──
    # Winners can drift above the cap; trim back to the limit without fully exiting.
    trim_sells = 0
    total_val  = portfolio_value(portfolio, price_map)
    for sym in list(portfolio['positions'].keys()):
        pos          = portfolio['positions'][sym]
        price        = price_map.get(sym, pos['entry_price'])
        current_val  = pos['shares'] * price
        cap_val      = total_val * MAX_POSITION_PCT
        # 5 % tolerance band avoids micro-trimming on tiny drifts
        if current_val > cap_val * 1.05:
            shares_to_sell = int((current_val - cap_val) / price)
            if shares_to_sell >= 1:
                proceeds = shares_to_sell * price
                pnl      = shares_to_sell * (price - pos['entry_price'])
                portfolio['cash'] += proceeds
                portfolio['positions'][sym]['shares'] -= shares_to_sell
                portfolio['trades'].append({
                    'date':   today, 'ticker': sym, 'action': 'SELL',
                    'shares': shares_to_sell, 'price':  round(price, 2),
                    'value':  round(proceeds, 2), 'pnl':    round(pnl, 2),
                    'reason': (f'Trim: {current_val/total_val:.0%} → cap '
                               f'{MAX_POSITION_PCT:.0%}'),
                })
                log.info('Trimmed %s by %d shares (%.0f%% → %.0f%% of portfolio)',
                         sym, shares_to_sell,
                         current_val / total_val * 100, MAX_POSITION_PCT * 100)
                trim_sells += 1

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
        f'{today}: trailing-stop {stop_sells}, take-profit {take_profit_sells}, '
        f'signal-sell {signal_sells}, trim {trim_sells}, bought {buys}, '
        f'positions={len(portfolio["positions"])}, total=${total_val:,.0f}'
    )
    return portfolio, summary
