"""
Unit tests for run_fisher_yield_curve.py.

Tests:
- _collect_results: verifies expected output schemas and t/T normalization.
- main: verifies expected in-sample and out-of-sample artifacts are written.
"""

from pathlib import Path

import pandas as pd

import run_fisher_yield_curve as fisher_run


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


def _fake_fisher_results():
    """Build a minimal Fisher-style results dictionary for one date."""
    return {
        pd.Timestamp("2000-01-31"): {
            "curve": pd.DataFrame({"T": [0.5, 1.0], "forward": [0.03, 0.032]}),
            "nodes": pd.DataFrame({"node_t": [1.0], "node_forward": [0.032]}),
            "bonds": pd.DataFrame({"cusip": ["A"], "model_price": [100.1]}),
            "lambda": 0.05,
            "wmae": 0.10,
            "hit_rate": 0.56,
        }
    }


def test_collect_results_normalizes_curve_time_column():
    """_collect_results should rename curve column T to t and include lambda in fit quality."""
    curves_df, nodes_df, bonds_df, fit_quality_df = fisher_run._collect_results(
        _fake_fisher_results()
    )

    assert "t" in curves_df.columns
    assert "T" not in curves_df.columns
    assert not nodes_df.empty
    assert not bonds_df.empty
    assert {"date", "lambda", "wmae", "hit_rate"}.issubset(fit_quality_df.columns)


def test_main_writes_expected_fisher_artifacts(tmp_path, monkeypatch):
    """main should write all expected Fisher in-sample and out-of-sample artifacts."""
    monkeypatch.setattr(fisher_run, "DATA_DIR", Path(tmp_path))

    dummy_tidy = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-31", "2000-02-29"]),
            "maturity_date": pd.to_datetime(["2001-01-31", "2001-02-28"]),
            "cusip": ["A", "B"],
        }
    )

    monkeypatch.setattr(
        fisher_run.cfu, "load_tidy_CRSP_treasury", lambda *_: dummy_tidy
    )
    monkeypatch.setattr(
        fisher_run.cfu, "filter_waggoner_treasury_data", lambda df, **_: df.copy()
    )
    monkeypatch.setattr(
        fisher_run.cfu, "split_in_out_sample_data", lambda df: (df.iloc[[0]], df.iloc[[1]])
    )
    monkeypatch.setattr(
        fisher_run.cfu, "get_full_error_metrics", lambda *_: _fake_error_metrics_df()
    )
    monkeypatch.setattr(
        fisher_run.fisher,
        "run_fisher",
        lambda *_args, **_kwargs: _fake_fisher_results(),
    )

    fisher_run.main(
        start_date=pd.Timestamp("2000-01-01"),
        end_date=pd.Timestamp("2000-12-31"),
        output_prefix="ut_",
    )

    expected = [
        "ut_fisher_forward_curve.parquet",
        "ut_fisher_bond_fits.parquet",
        "ut_fisher_fit_quality_by_date.csv",
        "ut_fisher_error_metrics.csv",
        "ut_fisher_oos_bond_fits.parquet",
        "ut_fisher_oos_error_metrics.csv",
    ]
    for fname in expected:
        assert (tmp_path / fname).exists(), f"Missing expected artifact: {fname}"
