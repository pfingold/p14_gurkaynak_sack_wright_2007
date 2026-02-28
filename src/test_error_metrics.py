"""
Unit tests for pricing error metrics: WMAE and Hit Rate.
Tests include:
    - Basic functionality with simple examples
    - Edge case with perfect fit (zero errors)
"""

import pytest
import numpy as np
from error_metrics import wmae, hit_rate

def test_wmae():
    "Test WMAE with a simple example: midpoints = [101,100]"
    bid = np.array([100.0, 99.0])
    ask = np.array([102.0, 101.0])
    model_price = np.array([101.0, 100.0])
    duration = np.array([1.0, 2.0])
    
    #weighted avg of [1,1] should be 1
    assert wmae(model_price, bid, ask, duration) == pytest.approx(1.0)

def test_hit_rate():
    "Test Hit Rate with a simple example: 2 hits, 2 misses"
    bid = np.array([99.0, 100.0, 101.0, 100.0])
    ask = np.array([101.0, 102.0, 103.0, 101.0])
    model_price = np.array([100.0, 99.0, 104.0, 100.5]) 
    
    assert hit_rate(model_price, bid, ask) == pytest.approx(0.5)

#Edge Case: Zero Errors
def test_perfect_fit():
    "Test WMAE and Hit Rate when model price perfectly matches mid price"
    bid = np.array([100.0, 99.0])
    ask = np.array([102.0, 101.0])
    mid = 0.5 * (bid + ask)  # perfect midpoints
    duration = np.array([1.0, 2.0])
    
    assert wmae(mid, bid, ask, duration) == pytest.approx(0.0)
    assert hit_rate(mid, bid, ask) == pytest.approx(1.0)