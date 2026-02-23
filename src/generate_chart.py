"""
File for creating charts:

 1. CRSP treasury data (price over time for four different ten-year bonds)
 2. Fed yield curve data (comparing parameterized and plotted curves for last 10 yrs)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import plot

from settings import config

from pull_yield_curve_data import load_fed_yield_curve_all
from pull_CRSP_treasury import load_CRSP_treasury_consolidated
from gsw2006_yield_curve import spot

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def CRSP_treasury_plot():
    prices_maturities = load_CRSP_treasury_consolidated(DATA_DIR, with_runness=False)

    # fix datetime column
    prices_maturities["mcaldt"] = pd.to_datetime(prices_maturities["mcaldt"])

    # find all treasury issues that were originally 10yr bonds
    ten_year_issues = (
        prices_maturities
        .loc[prices_maturities["original_maturity"] == 10, "kytreasno"]
        .unique()
    )

    # choose four issues from across the full sample (that have full data from issue to maturity present)
    issue_meta = (
        prices_maturities[prices_maturities["kytreasno"].isin(ten_year_issues)]
        .groupby("kytreasno")[["tdatdt", "tmatdt"]]
        .min()
        .reset_index()
        .sort_values("tdatdt")
    )
    DATA_START = prices_maturities["mcaldt"].min()
    DATA_END = prices_maturities["mcaldt"].max()
    issue_meta = issue_meta.loc[(issue_meta["tdatdt"] >= DATA_START) &
                                (issue_meta["tmatdt"] <= DATA_END)]

    selected_issues = issue_meta.iloc[
        [0, len(issue_meta)//3, 2*len(issue_meta)//3, -1]
    ]["kytreasno"].tolist()

    # get the data for the selected issues and format nicely for plotly
    df_plot = prices_maturities[prices_maturities["kytreasno"].isin(selected_issues)].copy()

    df_plot["years_since_issue"] = (
        df_plot["caldt"] - df_plot["tdatdt"]
    ).dt.days / 365.25

    df_plot["tdatdt"] = df_plot["tdatdt"].dt.date

    # create plotly line chart
    fig = px.line(
        df_plot,
        x="years_since_issue",
        y="price",
        color="tdatdt",
        title="Lifecycle of Selected 10-Year Treasury Issues"
    )

    fig.update_layout(
        xaxis_title="Years Since Issuance",
        yaxis_title="Clean Price",
        legend_title_text="Treasury Issue Date",
        template="plotly_white"
    )

    fig.update_layout(
        xaxis=dict(
            range=[
                df_plot["years_since_issue"].min() - 0.5,
                df_plot["years_since_issue"].max() + 0.5
            ]
        ),
        yaxis=dict(
            range=[
                df_plot["price"].min(),
                df_plot["price"].max()
            ]
        )
    )


    output_path = OUTPUT_DIR / "crsp_treasury_sample_plot.html"
    plot(fig, filename=str(output_path), auto_open=False)
    print(f'Saved interactive plot as html to: {output_path}')


def fed_yield_curve_plot():
    fed_yield_curve = load_fed_yield_curve_all(DATA_DIR)

    # get parameterized curve
    PARAMS = ["TAU1", "TAU2", "BETA0", "BETA1", "BETA2", "BETA3"]
    fed_params = fed_yield_curve[PARAMS]

    # get coordinates of curve
    ZERO_COUP_COLUMNS = [c for c in fed_yield_curve.columns if c[:5] == "SVENY"]
    fed_zero_coupon = fed_yield_curve[ZERO_COUP_COLUMNS]

    # get the x-axis labels
    MATURITIES = [int(z[-2:]) for z in ZERO_COUP_COLUMNS]

    # build plotly-friendly data of implied and reported yield curves over time
    rows = []

    for date in fed_zero_coupon["2016"::].index:
        for m, y_obs, y_fit in zip(
            MATURITIES,
            fed_zero_coupon.loc[date],
            spot(MATURITIES, fed_params.loc[date])
        ):
            rows.append({
                "maturity": m,
                "rate": y_obs,
                "series": "Observed (Zero Coupon)",
                "date": date
            })
            rows.append({
                "maturity": m,
                "rate": y_fit,
                "series": "Fitted (Spot Curve)",
                "date": date
            })

    df = pd.DataFrame(rows)

    # resample curve data to monthly touchpoints
    df = df.loc[df.date.dt.day == df.date.iloc[-1].day]

    # create plotly figure
    fig = px.line(
    df,
    x="maturity",
    y="rate",
    color="series",
    animation_frame="date",
    title="Observed vs Fitted Zero-Coupon Curve from Fed Data"
    )

    fig.update_layout(
        xaxis_title="Maturity (Years)",
        yaxis_title="Rate",
        template="plotly_white"
    )

    y_min = df["rate"].min()
    y_max = df["rate"].max()

    x_min = df["maturity"].min()
    x_max = df["maturity"].max()

    fig.update_layout(
        yaxis=dict(range=[y_min, y_max]),
        xaxis=dict(range=[x_min, x_max])
    )

    output_path = OUTPUT_DIR / "fed_yield_curve_sample_plot.html"
    plot(fig, filename=str(output_path), auto_open=False)
    print(f'Saved interactive plot as html to: {output_path}')

if __name__ == "__main__":
    CRSP_treasury_plot()
    fed_yield_curve_plot()
