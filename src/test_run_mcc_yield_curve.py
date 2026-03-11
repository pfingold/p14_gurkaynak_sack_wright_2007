"""
Unit tests for run_mcc_yield_curve.py.

Tests:
- _collect_results: verifies expected output schemas and row counts.
- main: verifies expected in-sample and out-of-sample artifacts are written.
"""

from pathlib import Path

import pandas as pd

import run_mcc_yield_curve as mcc_run


def _fake_error_metrics_df():
    """Create a small error-metrics DataFrame with expected index/columns."""
    idx = ["0-1", "1-3", "3-5", "5-10", ">10", "All"]
    return pd.DataFrame(
        {
            "wmae": [0.1, 0.2, 0.3, 0.4, 0.5, 0.25],
            "hit_rate": [0.6, 0.5, 0.4, 0.3, 0.2, 0.45],
        },
        index=idx,
    )


def _fake_mcc_results():
    """Build a minimal McCulloch-style results dictionary for one date."""
    return {
        pd.Timestamp("2000-01-31"): {
            "curve": pd.DataFrame({"t": [0.5, 1.0], "discount": [0.99, 0.97]}),
            "nodes": pd.DataFrame({"node_t": [1.0], "node_discount": [0.97]}),
            "bonds": pd.DataFrame({"cusip": ["A"], "model_price": [100.1]}),
            "wmae": 0.11,
            "hit_rate": 0.51,
        }
    }


def test_collect_results_returns_expected_tables():
    """_collect_results should return non-empty curve/node/bond/fit-quality tables."""
    curves_df, nodes_df, bonds_df, fit_quality_df = mcc_run._collect_results(
        _fake_mcc_results()
    )

    assert not curves_df.empty
    assert not nodes_df.empty
    assert not bonds_df.empty
    assert not fit_quality_df.empty
    assert {"date", "wmae", "hit_rate"}.issubset(fit_quality_df.columns)


def test_main_writes_expected_mcc_artifacts(tmp_path, monkeypatch):
    """main should write all expected McCulloch in-sample and out-of-sample artifacts."""
    monkeypatch.setattr(mcc_run, "DATA_DIR", Path(tmp_path))

    dummy_tidy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-31", "2000-02-29"]),
            "maturity_date": pd.to_datetime(["2001-01-31", "2001-02-28"]),
            "cusip": ["A", "B"],
        }
    )

    monkeypatch.setattr(mcc_run.cfu, "load_tidy_CRSP_treasury", lambda *_: dummy_tidy)
    monkeypatch.setattr(
        mcc_run.cfu, "filter_waggoner_treasury_data", lambda df, **_: df.copy()
    )
    monkeypatch.setattr(
        mcc_run.cfu, "split_in_out_sample_data", lambda df: (df.iloc[[0]], df.iloc[[1]])
    )
    monkeypatch.setattr(mcc_run.cfu, "get_full_error_metrics", lambda *_: _fake_error_metrics_df())
    monkeypatch.setattr(
        mcc_run.mcc,
        "run_mcculloch",
        lambda *_args, **_kwargs: _fake_mcc_results(),
    )

    mcc_run.main(start_date=pd.Timestamp("2000-01-01"), end_date=pd.Timestamp("2000-12-31"), output_prefix="ut_")

    expected = [
        "ut_mcc_discount_curve.parquet",
        "ut_mcc_error_metrics.csv",
        "ut_mcc_oos_error_metrics.csv",
    ]
    for fname in expected:
        assert (tmp_path / fname).exists(), f"Missing expected artifact: {fname}"
