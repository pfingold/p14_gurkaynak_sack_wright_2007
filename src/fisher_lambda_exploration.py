"""
Exploratory data analysis utilities for the Fisher (1995) lambda selection.

This module provides functions to load, classify, and analyze the regularization
parameter lambda selected by GCV across dates. Key questions addressed:

  - What is the empirical distribution of lambda, and how does it vary over time?
  - Are there macroeconomic or market-structure correlates of lambda choice?
  - How does the forward curve shape differ across lambda regimes?
  - Do the original (1970–1995) and modern (2006–2026) samples exhibit different
    lambda patterns?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_lambda_data(data_dir: Path, sample: str = "original") -> pd.DataFrame:
    """Load the Fisher fit-quality table for the requested sample.
    """
    prefix = "modern_" if sample == "modern" else ""
    path = Path(data_dir) / f"{prefix}fisher_fit_quality_by_date.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_curve_data(data_dir: Path, sample: str = "original") -> pd.DataFrame:
    """Load the Fisher forward-curve points for the requested sample.
    """
    prefix = "modern_" if sample == "modern" else ""
    path = Path(data_dir) / f"{prefix}fisher_forward_curve.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "t"]).reset_index(drop=True)
    return df


def load_bond_fits(data_dir: Path, sample: str = "original") -> pd.DataFrame:
    """Load per-bond fit results.
    """
    prefix = "modern_" if sample == "modern" else ""
    path = Path(data_dir) / f"{prefix}fisher_bond_fits.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_log_lambda(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``log10_lambda`` column to a lambda DataFrame.
    """
    out = df.copy()
    out["log10_lambda"] = np.log10(out["lambda"])
    return out

def classify_lambda_regime(
    df: pd.DataFrame,
    col: str = "log10_lambda",
    n_regimes: int = 5,
) -> pd.DataFrame:
    """Classify each date into lambda regimes using quantiles with numeric thresholds in labels."""

    out = df.copy()

    percentiles = np.linspace(0, 100, n_regimes + 1)[1:-1]
    thresholds = np.nanpercentile(out[col], percentiles)

    bins = np.concatenate(([-np.inf], thresholds, [np.inf]))

    # Build readable regime labels with numeric bounds
    labels = []
    for i in range(n_regimes):
        if i == 0:
            labels.append(f"≤ {thresholds[0]:.3f})")
        elif i == n_regimes - 1:
            labels.append(f"> {thresholds[-1]:.3f}")
        else:
            labels.append(
                f"{thresholds[i-1]:.3f} – {thresholds[i]:.3f}"
            )

    out["regime"] = pd.cut(out[col], bins=bins, labels=labels)

    for i, t in enumerate(thresholds):
        out[f"threshold_{i+1}"] = t

    return out


def add_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``decade`` column (e.g. '1970s', '1980s') to a dated DataFrame.
    """
    out = df.copy()
    out["decade"] = (out["date"].dt.year // 10 * 10).astype(str) + "s"
    return out

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def lambda_summary_by_decade(df: pd.DataFrame) -> pd.DataFrame:
    """Per-decade summary statistics of log10(lambda).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``log10_lambda`` and ``date``.

    Returns
    -------
    pd.DataFrame
        Index: decade; columns: count, mean, std, p10, p25, p50, p75, p90.
    """
    df = add_decade(df)
    stats = (
        df.groupby("decade")["log10_lambda"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            p10=lambda x: x.quantile(0.10),
            p25=lambda x: x.quantile(0.25),
            p50=lambda x: x.quantile(0.50),
            p75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.90),
        )
        .round(3)
    )
    return stats


def lambda_summary_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Per-regime summary of fit quality and market context.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``regime``, ``log10_lambda``, ``wmae``, ``hit_rate``.
        Optional: ``market_level``, ``market_slope``, ``n_bonds``.

    Returns
    -------
    pd.DataFrame
        One row per regime with mean and std of each metric.
    """
    cols_always = ["log10_lambda", "wmae", "hit_rate"]
    cols_optional = ["market_level", "market_slope", "n_bonds"]
    cols = cols_always + [c for c in cols_optional if c in df.columns]

    records = []
    for regime, grp in df.groupby("regime", observed=True):
        row: dict = {"regime": regime, "n_dates": len(grp)}
        for c in cols:
            row[f"{c}_mean"] = grp[c].mean()
            row[f"{c}_std"] = grp[c].std()
        records.append(row)

    return pd.DataFrame(records).set_index("regime").round(3)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT_DEFAULTS = {
    "template": "simple_white",
    "font": {"size": 12},
}


def plot_lambda_distribution(
    df: pd.DataFrame,
    color: str = "steelblue",
    label: str | None = None,
    bins: int = 30,
) -> go.Figure:
    """Histogram of log10(lambda) with a KDE overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``log10_lambda``.
    color : str
        Colour for both the histogram bars and the KDE line.
    label : str, optional
        Legend name for the histogram trace.
    bins : int

    Returns
    -------
    go.Figure
    """
    from scipy.stats import gaussian_kde

    x = df["log10_lambda"].dropna().to_numpy()
    kde = gaussian_kde(x)
    xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 300)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x,
        nbinsx=bins,
        histnorm="probability density",
        name=label or "Distribution",
        marker_color=color,
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=xs,
        y=kde(xs),
        mode="lines",
        line={"color": color, "width": 2},
        name="KDE",
        showlegend=False,
    ))
    fig.update_layout(
        xaxis_title="log\u2081\u2080(\u03bb)",
        yaxis_title="Density",
        title="Distribution of GCV-Selected \u03bb",
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def plot_lambda_heatmap_by_year_month(df: pd.DataFrame) -> go.Figure:
    """Calendar heatmap of log10(lambda) (month × year grid).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and ``log10_lambda``.

    Returns
    -------
    go.Figure
    """
    _MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    pivot = df.copy()
    pivot["year"] = pivot["date"].dt.year
    pivot["month"] = pivot["date"].dt.month
    mat = pivot.pivot_table(index="month", columns="year", values="log10_lambda", aggfunc="mean")

    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=[str(c) for c in mat.columns],
        y=_MONTH_LABELS[: len(mat.index)],
        colorscale="RdYlBu_r",
        colorbar={"title": "log\u2081\u2080(\u03bb)"},
        hovertemplate="Year: %{x}<br>Month: %{y}<br>log\u2081\u2080(\u03bb): %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="log\u2081\u2080(\u03bb) by Year and Month",
        xaxis={"title": "Year", "tickangle": -90},
        yaxis={"title": "Month"},
        height=350,
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def plot_compare_distributions(
    orig_df: pd.DataFrame,
    modern_df: pd.DataFrame,
) -> go.Figure:
    """Overlaid KDE + histogram comparison of log10(lambda) across two samples.

    Parameters
    ----------
    orig_df : pd.DataFrame
        Original sample; must contain ``log10_lambda``.
    modern_df : pd.DataFrame
        Modern sample; must contain ``log10_lambda``.

    Returns
    -------
    go.Figure
    """
    from scipy.stats import gaussian_kde

    fig = go.Figure()
    samples = [
        (orig_df,   "steelblue", "Original (1970\u20131995)"),
        (modern_df, "tomato",    "Modern (2006\u20132026)"),
    ]
    for df, color, name in samples:
        x = df["log10_lambda"].dropna().to_numpy()
        kde = gaussian_kde(x)
        xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 300)

        fig.add_trace(go.Histogram(
            x=x,
            nbinsx=30,
            histnorm="probability density",
            name=name,
            marker_color=color,
            opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=xs,
            y=kde(xs),
            mode="lines",
            line={"color": color, "width": 2},
            name=f"{name} KDE",
            showlegend=False,
        ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title="log\u2081\u2080(\u03bb)",
        yaxis_title="Density",
        title="Distribution of log\u2081\u2080(\u03bb): Original vs Modern Sample",
        **_PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------

def _wrap_latex_table(tabular: str, caption: str = "", label: str = "") -> str:
    """Wrap a pandas-generated tabular string with standard LaTeX boilerplate."""
    caption_line = f"\\caption{{{caption}}}\n" if caption else ""
    label_line = f"\\label{{{label}}}\n" if label else ""
    return (
        "\\begingroup\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\scriptsize\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{tabular.rstrip()}\n"
        "}\n"
        "\\endgroup\n"
    )


def format_decade_table_latex(
    df: pd.DataFrame,
    caption: str = r"$\log_{10}(\lambda)$ by Decade",
    label: str = "tab:lambda_decade",
) -> str:
    """Format the output of ``lambda_summary_by_decade()`` as a LaTeX string.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`lambda_summary_by_decade`.
    caption, label : str
        LaTeX caption and label strings.

    Returns
    -------
    str
        Complete LaTeX fragment (``\\begingroup`` … ``\\endgroup``).
    """
    display = df.copy()
    col_rename = {
        "count": "N",
        "mean":  r"$\bar{x}$",
        "std":   r"$\sigma$",
        "p10":   "p10",
        "p25":   "p25",
        "p50":   "p50",
        "p75":   "p75",
        "p90":   "p90",
    }
    display = display.rename(columns={k: v for k, v in col_rename.items() if k in display.columns})
    display.index.name = "Decade"
    display.columns = [f"\\textbf{{{c}}}" for c in display.columns]

    tabular = display.to_latex(
        index=True,
        escape=False,
        na_rep="",
        float_format="{:.3f}".format,
        column_format="l" + "r" * len(display.columns),
    )
    return _wrap_latex_table(tabular, caption=caption, label=label)


def format_regime_table_latex(
    df: pd.DataFrame,
    caption: str = r"$\log_{10}(\lambda)$ and Fit Quality by Regime",
    label: str = "tab:lambda_regime",
) -> str:
    """Format the output of ``lambda_summary_by_regime()`` as a LaTeX string.

    Only mean columns are included (std dropped for compactness). Hit rate is
    displayed as a percentage.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`lambda_summary_by_regime`.
    caption, label : str

    Returns
    -------
    str
        Complete LaTeX fragment.
    """
    mean_cols = ["n_dates"] + [c for c in df.columns if c.endswith("_mean")]
    display = df[mean_cols].copy()

    col_rename = {
        "n_dates":           "N",
        "log10_lambda_mean": r"$\overline{\log_{10}\lambda}$",
        "wmae_mean":         r"$\overline{\mathrm{WMAE}}$",
        "hit_rate_mean":     r"$\overline{\mathrm{HR}}$",
        "market_level_mean": r"$\overline{y_{\mathrm{lvl}}}$",
        "market_slope_mean": r"$\overline{y_{\mathrm{slp}}}$",
        "n_bonds_mean":      r"$\overline{N_{\mathrm{bonds}}}$",
    }
    display = display.rename(columns={k: v for k, v in col_rename.items() if k in display.columns})
    display.index.name = "Regime"

    hr_col = r"$\overline{\mathrm{HR}}$"
    if hr_col in display.columns:
        display[hr_col] = display[hr_col].apply(
            lambda x: f"{100 * x:.1f}\\%" if pd.notna(x) else ""
        )

    display.columns = [f"\\textbf{{{c}}}" for c in display.columns]

    tabular = display.to_latex(
        index=True,
        escape=False,
        na_rep="",
        float_format="{:.3f}".format,
        column_format="l" + "r" * len(display.columns),
    )
    return _wrap_latex_table(tabular, caption=caption, label=label)


def export_lambda_table(latex_str: str, out_path: Path) -> None:
    """Write a LaTeX string to disk, creating parent directories as needed.

    Parameters
    ----------
    latex_str : str
        LaTeX content to write.
    out_path : Path
        Destination file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex_str, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Build and export Fisher lambda summary tables for both samples.

    Reads DATA_DIR and OUTPUT_DIR from the project settings. Writes four
    ``.tex`` files to OUTPUT_DIR::

        lambda_decade_table_original.tex
        lambda_regime_table_original.tex
        lambda_decade_table_modern.tex
        lambda_regime_table_modern.tex
    """
    from settings import config

    data_dir = Path(config("DATA_DIR"))
    out_dir  = Path(config("OUTPUT_DIR"))
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("original", "1970\u20131995"),
        ("modern",   "2006\u20132026"),
    ]
    for sample, date_range in samples:
        df = load_lambda_data(data_dir, sample=sample)
        df = add_log_lambda(df)
        df = classify_lambda_regime(df)

        decade_df = lambda_summary_by_decade(df)
        decade_latex = format_decade_table_latex(
            decade_df,
            caption=rf"$\log_{{10}}(\lambda)$ by Decade \textemdash\ {date_range} Sample",
            label=f"tab:lambda_decade_{sample}",
        )
        export_lambda_table(decade_latex, out_dir / f"lambda_decade_table_{sample}.tex")

        regime_df = lambda_summary_by_regime(df)
        regime_latex = format_regime_table_latex(
            regime_df,
            caption=rf"$\log_{{10}}(\lambda)$ and Fit Quality by Regime \textemdash\ {date_range} Sample",
            label=f"tab:lambda_regime_{sample}",
        )
        export_lambda_table(regime_latex, out_dir / f"lambda_regime_table_{sample}.tex")

    print("Fisher lambda tables written to", out_dir)


if __name__ == "__main__":
    main()
