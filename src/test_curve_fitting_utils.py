"""
Unit tests for curve fitting utility helpers used in the treasury pipeline
    in curve_fitting_utils.py

Tests:
- split_in_out_sample_data: Ensures the longest-maturity bond in each date 
    bucket is assigned to in-sample
- get_cashflows_from_bonds: Validates cashflow extraction for zero-coupon 
    and short-stub bonds, including correct timing and amounts
- get_full_error_metrics: Checks that WMAE and hit rate are computed correctly 
    across defined time-to-maturity bins and for the overall sample
"""

import numpy as np
import pandas as pd
import pytest

from curve_fitting_utils import (
    get_cashflows_from_bonds,
    get_full_error_metrics,
    split_in_out_sample_data,
)


def test_split_in_out_sample_keeps_longest_maturity_in_sample_for_each_date():
    """The longest-maturity bond in each date bucket should always be assigned to in-sample."""
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-01-31",
                    "2020-01-31",
                    "2020-02-29",
                    "2020-02-29",
                    "2020-02-29",
                    "2020-02-29",
                ]
            ),
            "cusip": ["A", "B", "C", "D", "E", "F", "G"],
            "maturity_date": pd.to_datetime(
                [
                    "2020-06-30",
                    "2021-06-30",
                    "2022-06-30",
                    "2020-08-31",
                    "2021-08-31",
                    "2022-08-31",
                    "2023-08-31",
                ]
            ),
        }
    )

    in_sample, out_of_sample = split_in_out_sample_data(data.copy())

    longest_jan = in_sample.loc[in_sample["date"] == pd.Timestamp("2020-01-31"), "maturity_date"].max()
    longest_feb = in_sample.loc[in_sample["date"] == pd.Timestamp("2020-02-29"), "maturity_date"].max()

    assert longest_jan == pd.Timestamp("2022-06-30")
    assert longest_feb == pd.Timestamp("2023-08-31")
    assert len(in_sample) + len(out_of_sample) == len(data)
    assert set(in_sample["cusip"]).isdisjoint(set(out_of_sample["cusip"]))


def test_get_cashflows_from_zero_coupon_returns_single_face_payment():
    """A zero-coupon bond should return one cashflow equal to face value at maturity."""
    bonds = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")],
            "maturity_date": [pd.Timestamp("2021-01-01")],
            "coupon": [0.0],
        }
    )

    cashflows, times = get_cashflows_from_bonds(bonds)

    assert len(cashflows) == 1
    assert len(times) == 1
    assert cashflows[0].shape == (1,)
    assert times[0].shape == (1,)
    assert cashflows[0][0] == pytest.approx(100.0)
    assert times[0][0] == pytest.approx((pd.Timestamp("2021-01-01") - pd.Timestamp("2020-01-01")).days / 365.0)


def test_get_cashflows_infers_missing_first_coupon_and_applies_stub_adjustment():
    """Coupon cashflows should infer first coupon date and prorate first coupon for short stubs."""
    bonds = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-15")],
            "maturity_date": [pd.Timestamp("2021-01-31")],
            "coupon": [6.0],
            "issue_date": [pd.Timestamp("2020-01-15")],
        }
    )

    cashflows, times = get_cashflows_from_bonds(bonds, face=100, freq=2, stub_tol_days=3)
    coupon_amt = 3.0

    assert len(cashflows[0]) == 3
    assert len(times[0]) == 3
    assert np.all(np.diff(times[0]) > 0)
    assert cashflows[0][-1] > 100.0
    assert cashflows[0][0] < coupon_amt


def test_get_full_error_metrics_returns_expected_bin_values_and_labels():
    """Full error-metrics table should compute per-bin and all-sample WMAE and hit rate correctly."""
    bonds_a = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01"]),
            "cusip": ["A", "B", "C"],
            "bid": [99.0, 99.0, 99.0],
            "ask": [101.0, 101.0, 101.0],
            "duration": [1.0, 1.0, 1.0],
            "model_price": [100.0, 102.0, 98.0],
            "ttm": [0.5, 1.5, 3.5],
        }
    )
    bonds_b = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "cusip": ["D", "E"],
            "bid": [99.0, 99.0],
            "ask": [101.0, 101.0],
            "duration": [1.0, 1.0],
            "model_price": [100.0, 110.0],
            "ttm": [6.0, 12.0],
        }
    )
    results = {"run_a": {"bonds": bonds_a}, "run_b": {"bonds": bonds_b}}

    metrics = get_full_error_metrics(results)

    assert list(metrics.index) == ["0-1", "1-3", "3-5", "5-10", ">10", "All"]
    assert metrics.loc["0-1", "wmae"] == pytest.approx(0.0)
    assert metrics.loc["1-3", "wmae"] == pytest.approx(2.0)
    assert metrics.loc["3-5", "wmae"] == pytest.approx(2.0)
    assert metrics.loc["5-10", "wmae"] == pytest.approx(0.0)
    assert metrics.loc[">10", "wmae"] == pytest.approx(10.0)
    assert metrics.loc["All", "wmae"] == pytest.approx(2.8)
    assert metrics.loc["All", "hit_rate"] == pytest.approx(0.4)
