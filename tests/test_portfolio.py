"""
tests/test_portfolio.py — Unit tests for portfolio.py helper functions.

Run with:  python -m pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from portfolio import (
    _sector_limited_picks, _vol_weights, _corr_adjusted_weights,
    SECTOR_MAP, portfolio_value,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_cand(symbol, score, pred_pct=2.0, conf=0.6, price=100.0):
    return {'symbol': symbol, 'score': score, 'pred_pct': pred_pct,
            'conf': conf, 'price': price}


def make_closes(n=60, seed=0, drift=0.001):
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.02, n)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices)


# ── _sector_limited_picks ─────────────────────────────────────────────────────

def test_sector_cap_blocks_third_tech():
    """No more than 2 Technology picks should be selected."""
    candidates = [
        make_cand('AAPL', 10),   # Technology
        make_cand('MSFT', 9),    # Technology
        make_cand('NVDA', 8),    # Technology — should be blocked by cap
        make_cand('JPM',  7),    # Financials
        make_cand('BAC',  6),    # Financials
    ]
    picks = _sector_limited_picks(candidates, max_positions=5, max_per_sector=2)
    tech = [p['symbol'] for p in picks if SECTOR_MAP.get(p['symbol']) == 'Technology']
    assert len(tech) <= 2
    assert 'NVDA' not in [p['symbol'] for p in picks]


def test_sector_cap_allows_etfs_freely():
    """ETFs are exempt from the sector cap."""
    candidates = [
        make_cand('SPY',  10),  # ETF
        make_cand('QQQ',  9),   # ETF
        make_cand('SMH',  8),   # ETF
        make_cand('ARKK', 7),   # ETF
        make_cand('IWM',  6),   # ETF
    ]
    picks = _sector_limited_picks(candidates, max_positions=5, max_per_sector=2)
    assert len(picks) == 5   # all 5 ETFs should be allowed


def test_max_positions_respected():
    candidates = [make_cand(f'X{i}', 10 - i) for i in range(10)]
    # Assign dummy sectors to avoid sector cap interfering
    for c in candidates:
        # Patch into SECTOR_MAP temporarily won't work well; use ETFs
        pass
    etf_cands = [
        make_cand('SPY',  10), make_cand('QQQ',  9), make_cand('SMH',  8),
        make_cand('ARKK', 7),  make_cand('IWM',  6), make_cand('DIA',  5),
    ]
    picks = _sector_limited_picks(etf_cands, max_positions=3, max_per_sector=2)
    assert len(picks) == 3


def test_negative_score_excluded():
    """Candidates with score <= 0 should never be selected."""
    candidates = [
        make_cand('SPY', 5),
        make_cand('QQQ', 0),    # score = 0, should be excluded
        make_cand('SMH', -10),  # negative, should be excluded
    ]
    picks = _sector_limited_picks(candidates, max_positions=5, max_per_sector=2)
    syms = [p['symbol'] for p in picks]
    assert 'QQQ' not in syms
    assert 'SMH' not in syms
    assert 'SPY' in syms


def test_empty_candidates():
    picks = _sector_limited_picks([], max_positions=5, max_per_sector=2)
    assert picks == []


# ── _vol_weights ──────────────────────────────────────────────────────────────

def test_vol_weights_sum_to_one():
    picks = [
        make_cand('SPY', 10),
        make_cand('QQQ', 9),
        make_cand('GLD', 8),
    ]
    closes_map = {
        'SPY': make_closes(60, seed=0, drift=0.001),
        'QQQ': make_closes(60, seed=1, drift=0.002),
        'GLD': make_closes(60, seed=2, drift=0.0005),
    }
    weights = _vol_weights(picks, closes_map)
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9


def test_vol_weights_lower_vol_gets_more():
    """Lower-volatility asset should receive a higher weight."""
    picks = [make_cand('LOW', 10), make_cand('HIGH', 9)]
    rng = np.random.default_rng(42)
    closes_map = {
        'LOW':  pd.Series(100 + rng.normal(0, 0.5, 60).cumsum()),   # low vol
        'HIGH': pd.Series(100 + rng.normal(0, 5.0, 60).cumsum()),   # high vol
    }
    weights = _vol_weights(picks, closes_map)
    assert weights['LOW'] > weights['HIGH']


def test_vol_weights_missing_data_default():
    """Missing close data should use a default weight, not crash."""
    picks = [make_cand('SPY', 10), make_cand('MISSING', 9)]
    closes_map = {'SPY': make_closes(60)}
    weights = _vol_weights(picks, closes_map)
    assert 'SPY' in weights
    assert 'MISSING' in weights
    assert abs(sum(weights.values()) - 1.0) < 1e-9


# ── _corr_adjusted_weights ────────────────────────────────────────────────────

def test_corr_adjusted_weights_sum_to_one():
    picks = [make_cand('A', 10), make_cand('B', 9), make_cand('C', 8)]
    rng = np.random.default_rng(0)
    closes_map = {
        'A': pd.Series(100 + rng.normal(0, 2, 80).cumsum()),
        'B': pd.Series(100 + rng.normal(0, 2, 80).cumsum()),
        'C': pd.Series(100 + rng.normal(0, 2, 80).cumsum()),
    }
    base = {'A': 0.4, 'B': 0.35, 'C': 0.25}
    adj  = _corr_adjusted_weights(picks, closes_map, base)
    assert abs(sum(adj.values()) - 1.0) < 1e-9


def test_corr_adjusted_weights_reduces_correlated():
    """Highly correlated pair should result in lower weight for lower-ranked."""
    rng    = np.random.default_rng(1)
    shared = rng.normal(0, 2, 80).cumsum()
    closes_map = {
        'A': pd.Series(100 + shared),                          # rank 1
        'B': pd.Series(100 + shared + rng.normal(0, 0.01, 80)),# rank 2, nearly identical
        'C': pd.Series(100 + rng.normal(0, 2, 80).cumsum()),   # rank 3, uncorrelated
    }
    picks = [make_cand('A', 10), make_cand('B', 9), make_cand('C', 8)]
    base  = {'A': 1/3, 'B': 1/3, 'C': 1/3}
    adj   = _corr_adjusted_weights(picks, closes_map, base)
    # B is highly correlated with A (ranked higher), so B's weight should be reduced
    assert adj['B'] < base['B']


def test_corr_adjusted_weights_single_pick():
    """Single pick should return unchanged (normalised) weight."""
    picks = [make_cand('SPY', 10)]
    closes_map = {'SPY': make_closes(60)}
    base = {'SPY': 1.0}
    adj  = _corr_adjusted_weights(picks, closes_map, base)
    assert abs(adj['SPY'] - 1.0) < 1e-9


def test_corr_adjusted_weights_floor():
    """No weight should drop below the 5% floor."""
    rng    = np.random.default_rng(2)
    shared = rng.normal(0, 2, 80).cumsum()
    closes_map = {s: pd.Series(100 + shared) for s in 'ABCDE'}
    picks = [make_cand(s, 10 - i) for i, s in enumerate('ABCDE')]
    base  = {s: 0.2 for s in 'ABCDE'}
    adj   = _corr_adjusted_weights(picks, closes_map, base)
    for w in adj.values():
        assert w >= 0.05 - 1e-9


# ── portfolio_value ────────────────────────────────────────────────────────────

def test_portfolio_value_cash_only():
    p = {'cash': 50_000.0, 'positions': {}}
    assert portfolio_value(p, {}) == 50_000.0


def test_portfolio_value_with_positions():
    p = {
        'cash': 10_000.0,
        'positions': {
            'AAPL': {'shares': 10, 'entry_price': 150.0},
            'MSFT': {'shares':  5, 'entry_price': 200.0},
        }
    }
    price_map = {'AAPL': 160.0, 'MSFT': 210.0}
    val = portfolio_value(p, price_map)
    # 10_000 + 10*160 + 5*210 = 10_000 + 1_600 + 1_050 = 12_650
    assert abs(val - 12_650.0) < 0.01


def test_portfolio_value_falls_back_to_entry():
    """When price_map doesn't have a symbol, use entry_price."""
    p = {
        'cash': 0.0,
        'positions': {'AAPL': {'shares': 10, 'entry_price': 150.0}},
    }
    val = portfolio_value(p, {})   # empty price_map
    assert abs(val - 1_500.0) < 0.01
