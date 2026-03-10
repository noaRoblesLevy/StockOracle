"""
tests/test_model.py — Unit tests for model.py prediction helpers.

Run with:  python -m pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from model import _shrink, _rsi, _bb_pct_b, _macd, _volume_ratio, _fast_screen


# ── _shrink ───────────────────────────────────────────────────────────────────

def test_shrink_no_data():
    assert _shrink(0, 0) == 0.5

def test_shrink_all_correct():
    # With lots of data all correct, result approaches 1
    result = _shrink(1000, 1000)
    assert result > 0.9

def test_shrink_half_correct():
    # 50 % accuracy → shrinks toward 0.5
    result = _shrink(50, 100)
    assert 0.48 < result < 0.52

def test_shrink_small_sample():
    # 5 correct out of 5 — shrinkage pulls it well below 1
    result = _shrink(5, 5)
    assert result < 0.9


# ── _rsi ─────────────────────────────────────────────────────────────────────

def test_rsi_insufficient_data():
    closes = pd.Series([100.0] * 10)
    assert _rsi(closes) == 50.0

def test_rsi_flat_series():
    # No movement — gains and losses are both zero; RSI defaults to 50
    closes = pd.Series([100.0] * 20)
    rsi = _rsi(closes)
    assert 45 <= rsi <= 55

def test_rsi_only_gains():
    # Consistently rising prices → high RSI
    closes = pd.Series([float(i) for i in range(50, 80)])
    rsi = _rsi(closes)
    assert rsi > 70

def test_rsi_only_losses():
    # Consistently falling prices → low RSI
    closes = pd.Series([float(i) for i in range(80, 50, -1)])
    rsi = _rsi(closes)
    assert rsi < 30

def test_rsi_bounds():
    closes = pd.Series(np.random.default_rng(0).uniform(50, 200, 60))
    rsi = _rsi(closes)
    assert 0 <= rsi <= 100


# ── _bb_pct_b ─────────────────────────────────────────────────────────────────

def test_bb_pct_b_insufficient_data():
    closes = pd.Series([100.0] * 10)
    assert _bb_pct_b(closes) == 0.5

def test_bb_pct_b_at_midband():
    # Flat series — price is at the mean, so %B ≈ 0.5
    closes = pd.Series([100.0] * 30)
    result = _bb_pct_b(closes)
    # std is 0 for flat series → returns 0.5
    assert result == 0.5

def test_bb_pct_b_clamped():
    # Result is always between 0 and 1
    rng = np.random.default_rng(42)
    closes = pd.Series(rng.uniform(50, 300, 60))
    result = _bb_pct_b(closes)
    assert 0.0 <= result <= 1.0

def test_bb_pct_b_high_price():
    # Price much higher than rolling average → %B near 1
    base = pd.Series([100.0] * 25)
    spike = pd.Series([500.0])
    closes = pd.concat([base, spike], ignore_index=True)
    result = _bb_pct_b(closes)
    assert result > 0.8


# ── _macd ─────────────────────────────────────────────────────────────────────

def test_macd_insufficient_data():
    closes = pd.Series([100.0] * 20)
    macd, signal = _macd(closes)
    assert macd == 0.0 and signal == 0.0

def test_macd_types():
    closes = pd.Series(np.linspace(100, 150, 60))
    macd, signal = _macd(closes)
    assert isinstance(macd, float)
    assert isinstance(signal, float)

def test_macd_bullish_trend():
    # Steadily rising prices → MACD line should be positive
    closes = pd.Series(np.linspace(100, 200, 80))
    macd, signal = _macd(closes)
    assert macd > 0

def test_macd_bearish_trend():
    # Steadily falling prices → MACD line should be negative
    closes = pd.Series(np.linspace(200, 100, 80))
    macd, signal = _macd(closes)
    assert macd < 0


# ── _volume_ratio ─────────────────────────────────────────────────────────────

def test_volume_ratio_no_data():
    assert _volume_ratio(None) == 1.0

def test_volume_ratio_insufficient_data():
    vols = pd.Series([1_000_000.0] * 10)
    assert _volume_ratio(vols) == 1.0

def test_volume_ratio_flat():
    # Constant volume → ratio should be ~1.0
    vols = pd.Series([1_000_000.0] * 30)
    ratio = _volume_ratio(vols)
    assert abs(ratio - 1.0) < 0.01

def test_volume_ratio_spike():
    # Big spike at the end → ratio > 1
    vols = pd.Series([1_000_000.0] * 25 + [5_000_000.0])
    ratio = _volume_ratio(vols)
    assert ratio > 1.5

def test_volume_ratio_low():
    # Very low recent volume → ratio < 1
    vols = pd.Series([1_000_000.0] * 25 + [100_000.0])
    ratio = _volume_ratio(vols)
    assert ratio < 0.5


# ── _fast_screen ──────────────────────────────────────────────────────────────

def test_fast_screen_insufficient_data():
    closes = pd.Series([100.0] * 20)
    pred, conf = _fast_screen(closes, 5)
    assert pred is None and conf is None

def test_fast_screen_returns_floats():
    rng    = np.random.default_rng(1)
    closes = pd.Series(100 + rng.normal(0, 2, 80).cumsum())
    pred, conf = _fast_screen(closes, 5)
    assert pred is not None
    assert isinstance(pred, float)
    assert isinstance(conf, float)

def test_fast_screen_conf_range():
    # Confidence must be a valid probability [0, 1]
    rng    = np.random.default_rng(2)
    closes = pd.Series(100 + rng.normal(0, 3, 100).cumsum())
    _, conf = _fast_screen(closes, 5)
    assert conf is not None
    assert 0.0 <= conf <= 1.0

def test_fast_screen_with_volumes():
    # Passing volumes should not crash and returns valid output
    rng     = np.random.default_rng(3)
    closes  = pd.Series(100 + rng.normal(0, 2, 80).cumsum())
    volumes = pd.Series(rng.integers(500_000, 5_000_000, 80).astype(float))
    pred, conf = _fast_screen(closes, 5, volumes=volumes)
    assert pred is not None
    assert 0.0 <= conf <= 1.0

def test_fast_screen_horizons():
    # Should work for all supported horizons
    rng    = np.random.default_rng(4)
    closes = pd.Series(100 + rng.normal(0, 2, 100).cumsum())
    for h in (1, 5, 21):
        pred, conf = _fast_screen(closes, h)
        assert pred is not None, f'horizon={h} returned None'
