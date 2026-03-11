"""
Unit tests for correlation metric helpers in correlation_metrics.py.

Tests:
- _safe_corr: handles edge cases (insufficient data, constant arrays) and
  returns expected correlations for valid inputs.
- _pairwise_matrix: constructs a symmetric pairwise correlation matrix with
  ones on the diagonal.
- select_representative_dates: returns low/median/high representative dates.
- _gsw_spot: returns finite values with the expected shape for valid params.
"""

import numpy as np
import pandas as pd
import pytest

from correlation_metrics import (
    _gsw_spot,
    _pairwise_matrix,
    _safe_corr,
    select_representative_dates,
)


def test_safe_corr_returns_nan_for_insufficient_valid_points():
    """_safe_corr should return NaN when fewer than 3 finite paired points exist."""
    x = np.array([1.0, np.nan, 2.0])
    y = np.array([1.0, np.nan, np.nan])

    out = _safe_corr(x, y)
    assert np.isnan(out)


def test_safe_corr_returns_nan_for_constant_series():
    """_safe_corr should return NaN when either series has near-zero variance."""
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    out = _safe_corr(x, y)
    assert np.isnan(out)


def test_safe_corr_returns_one_for_identical_series():
    """_safe_corr should return approximately 1 for identical non-constant vectors."""
    x = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    y = x.copy()

    out = _safe_corr(x, y)
    assert out == pytest.approx(1.0, abs=1e-12)


def test_pairwise_matrix_is_symmetric_with_unit_diagonal():
    """_pairwise_matrix should populate a symmetric matrix with ones on the diagonal."""
    pairwise_overall = pd.DataFrame(
        {
            "curve_type": ["spot_cc", "spot_cc", "spot_cc"],
            "method_1": ["mcc", "mcc", "fisher"],
            "method_2": ["fisher", "waggoner", "waggoner"],
            "overall_correlation": [0.85, 0.72, 0.91],
        }
    )

    mat = _pairwise_matrix(pairwise_overall, curve_type="spot_cc")
    vals = mat.to_numpy(dtype=float)

    assert mat.loc["mcc", "fisher"] == pytest.approx(0.85)
    assert mat.loc["fisher", "mcc"] == pytest.approx(0.85)
    assert mat.loc["mcc", "waggoner"] == pytest.approx(0.72)
    assert mat.loc["waggoner", "fisher"] == pytest.approx(0.91)
    assert np.allclose(np.diag(vals), 1.0)
    assert np.allclose(vals, vals.T)


def test_select_representative_dates_returns_expected_labels():
    """select_representative_dates should return low/median/high labels with valid dates."""
    summary = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31"]
            ),
            "overall_corr": [0.20, 0.40, 0.60, 0.80, 0.95],
        }
    )

    out = select_representative_dates(summary)

    assert set(out["label"]) == {"low_corr", "median_corr", "high_corr"}
    assert len(out) == 3
    assert out.loc[out["label"] == "low_corr", "overall_corr"].iloc[0] == pytest.approx(0.20)
    assert out.loc[out["label"] == "high_corr", "overall_corr"].iloc[0] == pytest.approx(0.95)


def test_gsw_spot_returns_finite_values_with_expected_shape():
    """_gsw_spot should return a finite vector of the same length as input maturities."""
    maturities = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
    params = (1.5, 4.0, 0.02, -0.01, 0.005, -0.002)

    spot = _gsw_spot(maturities, params)

    assert spot.shape == maturities.shape
    assert np.isfinite(spot).all()
