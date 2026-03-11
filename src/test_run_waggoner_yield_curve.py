"""
Unit tests for run_waggoner_yield_curve.py.

Tests:
- _collect_results: verifies expected output schemas and t/T normalization.
- main: verifies expected in-sample and out-of-sample artifacts are written.
"""

from pathlib import Path

import pandas as pd

import run_waggoner_yield_curve as waggoner_run


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


def _fake_waggoner_results():
    """Build a minimal Waggoner-style results dictionary for one date."""
    return {
        pd.Timestamp("2000-01-31"): {
            "curve": pd.DataFrame({"T": [0.5, 1.0], "forward": [0.029, 0.031]}),
            "nodes": pd.DataFrame({"node_t": [1.0], "node_forward": [0.031]}),
            "bonds": pd.DataFrame({"cusip": ["A"], "model_price": [100.1]}),
            "wmae": 0.09,
            "hit_rate": 0.58,
        }
    }


def test_collect_results_normalizes_curve_time_column():
    """_collect_results should rename curve column T to t and include fit metrics."""
    curves_df, nodes_df, bonds_df, fit_quality_df = waggoner_run._collect_results(
        _fake_waggoner_results()
    )

    assert "t" in curves_df.columns
    assert "T" not in curves_df.columns
    assert not nodes_df.empty
    assert not bonds_df.empty
    assert {"date", "wmae", "hit_rate"}.issubset(fit_quality_df.columns)


def test_main_writes_expected_waggoner_artifacts(tmp_path, monkeypatch):
    """main should write all expected Waggoner in-sample and out-of-sample artifacts."""
    monkeypatch.setattr(waggoner_run, "DATA_DIR", Path(tmp_path))

    dummy_tidy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-31", "2000-02-29"]),
            "maturity_date": pd.to_datetime(["2001-01-31", "2001-02-28"]),
            "cusip": ["A", "B"],
        }
    )

    monkeypatch.setattr(
        waggoner_run.cfu, "load_tidy_CRSP_treasury", lambda *_: dummy_tidy
    )
    monkeypatch.setattr(
        waggoner_run.cfu, "filter_waggoner_treasury_data", lambda df, **_: df.copy()
    )
    monkeypatch.setattr(
        waggoner_run.cfu, "split_in_out_sample_data", lambda df: (df.iloc[[0]], df.iloc[[1]])
    )
    monkeypatch.setattr(
        waggoner_run.cfu, "get_full_error_metrics", lambda *_: _fake_error_metrics_df()
    )
    monkeypatch.setattr(
        waggoner_run.waggoner,
        "run_waggoner",
        lambda *_args, **_kwargs: _fake_waggoner_results(),
    )

    waggoner_run.main(
        start_date=pd.Timestamp("2000-01-01"),
        end_date=pd.Timestamp("2000-12-31"),
        output_prefix="ut_",
    )

    expected = [
        "ut_waggoner_forward_curve.parquet",
        "ut_waggoner_forward_curve_nodes.csv",
        "ut_waggoner_bond_fits.parquet",
        "ut_waggoner_fit_quality_by_date.csv",
        "ut_waggoner_error_metrics.csv",
        "ut_waggoner_oos_bond_fits.parquet",
        "ut_waggoner_oos_fit_quality_by_date.csv",
        "ut_waggoner_oos_error_metrics.csv",
    ]
    for fname in expected:
        assert (tmp_path / fname).exists(), f"Missing expected artifact: {fname}"
