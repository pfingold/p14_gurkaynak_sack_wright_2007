"""
Unit tests for weighted mean absolute error and bid-ask hit-rate calculations
    in error_metrics.py

Tests:
- wmae: Validates that WMAE is zero when model prices match midpoints, and 
    that it correctly applies inverse-duration weighting
- hit_rate: Validates that hit rate counts the proportion of model prices 
    within bid-ask bounds, and that perfect fits yield 100% hit rate
"""
import numpy as np
import pytest

from error_metrics import wmae, hit_rate

def test_wmae_is_zero_when_model_prices_equal_midpoints():
    """WMAE should be zero when model prices exactly match bid-ask midpoints"""
    bid = np.array([100.0, 98.0])
    ask = np.array([102.0, 100.0])
    model_price = np.array([101.0, 99.0])
    duration = np.array([1.0, 2.0])

    assert wmae(model_price, bid, ask, duration) == pytest.approx(0.0)


def test_wmae_matches_manual_inverse_duration_weighting():
    """WMAE should match a hand-computed inverse-duration weighted average error"""
    bid = np.array([99.0, 99.0])
    ask = np.array([101.0, 101.0])
    model_price = np.array([102.0, 98.0])  # absolute errors = [2, 2]
    duration = np.array([1.0, 4.0])  # weights = [1, 0.25]

    expected = (1.0 * 2.0 + 0.25 * 2.0) / (1.0 + 0.25)
    assert wmae(model_price, bid, ask, duration) == pytest.approx(expected)


def test_hit_rate_counts_share_of_prices_inside_bid_ask_spread():
    """Hit rate should equal the proportion of model prices inside bid-ask bounds"""
    bid = np.array([99.0, 100.0, 101.0, 100.0])
    ask = np.array([101.0, 102.0, 103.0, 101.0])
    model_price = np.array([100.0, 99.0, 104.0, 100.5])

    assert hit_rate(model_price, bid, ask) == pytest.approx(0.5)


def test_perfect_fit_has_zero_wmae_and_unit_hit_rate():
    """Perfect midpoint pricing should produce zero WMAE and 100% hit rate"""
    bid = np.array([100.0, 99.0])
    ask = np.array([102.0, 101.0])
    mid = 0.5 * (bid + ask)
    duration = np.array([1.0, 2.0])

    assert wmae(mid, bid, ask, duration) == pytest.approx(0.0)
    assert hit_rate(mid, bid, ask) == pytest.approx(1.0)
