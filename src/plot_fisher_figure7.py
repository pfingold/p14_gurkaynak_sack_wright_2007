"""
Build a Waggoner Paper Fisher Figure 7 style replication 
plot for February 28, 1977 and output the resulting 
curve in docs/charts
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from settings import config

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CHARTS_DIR = ROOT_DIR / "docs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(config("DATA_DIR"))

TARGET_DATE = pd.Timestamp("1977-02-28")
OUT_HTML = CHARTS_DIR / "fisher_figure7_1977_02_28.html"


def _nearest_curve_point(curve, x_val):
    """Compute nearest curve point."""
    idx = (curve["t"] - float(x_val)).abs().idxmin()
    return float(curve.loc[idx, "t"]), float(curve.loc[idx, "forward_pct"])


def _annotation_block(curve, x_val, label, ax, ay):
    """Compute annotation block."""
    x_anchor, y_anchor = _nearest_curve_point(curve, x_val)
    return dict(
        x=x_anchor,
        y=y_anchor,
        xref="x",
        yref="y",
        text=label,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.0,
        ax=ax,
        ay=ay,
        font=dict(size=16, color="black"),
        align="left",
    )


def main():
    curve_path = DATA_DIR / "fisher_forward_curve.parquet"
    in_sample_path = DATA_DIR / "fisher_bond_fits.parquet"
    out_sample_path = DATA_DIR / "fisher_oos_bond_fits.parquet"

    curve_all = pd.read_parquet(curve_path)
    in_sample_all = pd.read_parquet(in_sample_path)
    out_sample_all = pd.read_parquet(out_sample_path)

    curve_all["date"] = pd.to_datetime(curve_all["date"]).dt.normalize()
    in_sample_all["date"] = pd.to_datetime(in_sample_all["date"]).dt.normalize()
    out_sample_all["date"] = pd.to_datetime(out_sample_all["date"]).dt.normalize()

    curve = curve_all.loc[curve_all["date"] == TARGET_DATE].copy()
    in_sample = in_sample_all.loc[in_sample_all["date"] == TARGET_DATE].copy()
    out_sample = out_sample_all.loc[out_sample_all["date"] == TARGET_DATE].copy()

    if curve.empty:
        raise ValueError(f"No Fisher curve available for {TARGET_DATE.date()}.")
    if in_sample.empty:
        raise ValueError(f"No in-sample Fisher bonds available for {TARGET_DATE.date()}.")
    if out_sample.empty:
        raise ValueError(f"No out-of-sample Fisher bonds available for {TARGET_DATE.date()}.")

    curve = curve.sort_values("t").reset_index(drop=True)
    curve["forward_pct"] = 100.0 * curve["forward"]
    x_min = float(curve["t"].min())
    x_max = float(curve["t"].max())
    y_min = float(curve["forward_pct"].min())
    y_max = float(curve["forward_pct"].max())

    # Add padding so the curve never clips at the panel boundaries.
    x_pad = max(0.2, 0.03 * (x_max - x_min))
    y_pad = max(0.6, 0.08 * (y_max - y_min))

    in_sorted = in_sample.sort_values("ttm").reset_index(drop=True)
    out_sorted = out_sample.sort_values("ttm").reset_index(drop=True)

    last_in = float(in_sorted["ttm"].iloc[-1])
    next_last_in = float(in_sorted["ttm"].iloc[-2]) if len(in_sorted) >= 2 else float(in_sorted["ttm"].iloc[-1])
    last_out = float(out_sorted["ttm"].iloc[-1])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["t"],
            y=curve["forward_pct"],
            mode="lines",
            name="Fisher Forward Curve",
            line=dict(color="black", width=4),
            hovertemplate="Maturity=%{x:.2f}y<br>Percent=%{y:.2f}<extra></extra>",
        )
    )

    annotations = [
        _annotation_block(curve, last_in, "Last in-sample bond", ax=-190, ay=-40),
        _annotation_block(curve, next_last_in, "Next to last in-sample bond", ax=-200, ay=110),
        _annotation_block(curve, last_out, "Last out-of-sample bond", ax=-70, ay=-220),
    ]

    fig.update_layout(
        title=dict(
            text="Figure 7: Fisher Yield Curve (Replication) for February 28, 1977",
            x=0.5,
            xanchor="center",
            font=dict(size=24, family="Times New Roman"),
        ),
        template="plotly_white",
        xaxis=dict(
            title="Years to Maturity",
            range=[x_min - x_pad, x_max + x_pad],
            dtick=2,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            title_font=dict(size=16, family="Times New Roman"),
            tickfont=dict(size=13, family="Times New Roman"),
        ),
        yaxis=dict(
            title="Percent",
            range=[y_min - y_pad, y_max + y_pad],
            dtick=5,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="#a7a7a7",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            title_font=dict(size=16, family="Times New Roman"),
            tickfont=dict(size=13, family="Times New Roman"),
        ),
        showlegend=False,
        margin=dict(l=70, r=40, t=70, b=60),
        annotations=annotations,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print("Wrote Fisher figure to:", OUT_HTML.resolve())


if __name__ == "__main__":
    main()
