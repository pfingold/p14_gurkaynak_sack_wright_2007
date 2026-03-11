"""Tests for replication table metrics against paper benchmark values 
from Table 1a & 1b of the Waggonner (1997) paper."""

from replication_tables import (
    compute_replication_values,
    table_1a_values,
    table_1b_values,
)

# Separate tolerances by metric type (absolute differences)
# Require at least 90% of values in each metric category (across IS + OOS).
WMAE_THRESHOLD = 0.10
HIT_RATE_THRESHOLD = 0.08
MIN_PASS_RATE = 0.90


def _collect_diffs(sample_label, paper_values):
    replication_values = compute_replication_values(sample_label)
    diffs = []
    for key, paper_value in paper_values.items():
        method, metric, bucket = key
        repl_value = replication_values[key]
        diff = abs(repl_value - paper_value)
        diffs.append((sample_label, method, metric, bucket, diff))
    return diffs


def test_table_1a_replication_values_present():
    """In-sample replication dictionary contains all table 1a keys."""
    paper_values = table_1a_values()
    replication_values = compute_replication_values("in")
    assert set(paper_values).issubset(set(replication_values))


def test_table_1b_replication_values_present():
    """Out-of-sample replication dictionary contains all table 1b keys."""
    paper_values = table_1b_values()
    replication_values = compute_replication_values("out")
    assert set(paper_values).issubset(set(replication_values))


def test_replication_values_within_metric_thresholds_at_90pct():
    """
    At least 90% of values in each metric category (WMAE, Hit Rate),
    across in-sample and out-of-sample tables, must be within threshold.
    """
    all_diffs = _collect_diffs("in", table_1a_values()) + _collect_diffs(
        "out", table_1b_values()
    )

    for metric, threshold in [
        ("WMAE", WMAE_THRESHOLD),
        ("Hit Rate", HIT_RATE_THRESHOLD),
    ]:
        metric_rows = [row for row in all_diffs if row[2] == metric]
        n_total = len(metric_rows)
        n_pass = sum(row[4] <= threshold for row in metric_rows)
        n_fail = n_total - n_pass
        pass_rate = n_pass / n_total
        worst = max(metric_rows, key=lambda x: x[4])

        assert pass_rate >= MIN_PASS_RATE, (
            f"{metric=} pass_rate={pass_rate:.3f} required={MIN_PASS_RATE:.3f} "
            f"threshold={threshold:.6f} n_pass={n_pass} n_total={n_total} "
            f"worst={worst}"
        )
        # Keep thresholds meaningful: allow some misses, but not too many
        assert n_fail >= 1, (
            f"{metric=} has zero misses at current threshold={threshold:.6f}; "
            "consider tightening the threshold further."
        )
