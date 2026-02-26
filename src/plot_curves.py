"""
Executes the replication pipeline and plots the resulting curves:
    - Load & filter CRSP Treausry data (curve_fitting_utils.py)
    - Build MCC discount curves (mcc1975_yield_curve.py)
    - Convert discount to spot & forward (curve_conversions.py)
    - Plot the curves for selected dates (this file)

"""
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import curve_fitting_utils as cfu
import curve_conversions as cc
import mcc1975_yield_curve as mcc

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CHARTS_DIR = ROOT_DIR / "docs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    
def pick_dates(df, n=3):
    "Pick n evenly spaced dates from the available sample dates in the DataFrame"
    dates = pd.to_datetime(sorted(df["date"].unique()))
    if len(dates) == 0:
        raise ValueError("No dates found in the DataFrame.")
    if n >= len(dates):
        return list(dates)
    idx = np.linspace(0, len(dates) - 1, n).round().astype(int)
    return list(dates[idx])

def build_converted_curve(treasury_filtered, dates_to_plot, dt_fwd=0.25):
    "Build MCC curve for selected dates and add spot/forward conversions"
    sample = treasury_filtered.copy()
    sample['date'] = pd.to_datetime(sample['date'])
    sample = sample.loc[sample['date'].isin(dates_to_plot)].copy()

    #Run MCC Curve Construction
    results = mcc.run_mcculloch(sample)

    out = {}
    for d in dates_to_plot:
        key = None
        for k in results.keys():
            if pd.to_datetime(k) == pd.to_datetime(d):
                key = k
                break
        if key is None:
            raise ValueError(f"Could not find matching key for date {d}")

        curve_df = results[key]["curve"]
        curve_conv = cc.add_spot_and_forwards(curve_df, 
                        dt=dt_fwd, t_col="T", d_col="discount")
        out[pd.to_datetime(d)] = curve_conv

    return out


### Plotting Functions ###
def plot_curves(curves_dict, 
                x_col, y_col,
                title, y_axis_title, out_html):
    """Plot curves from the curves_dict with Plotly"""
    fig = go.Figure()
    for date, df in curves_dict.items():
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], 
                    mode='lines', name=str(pd.to_datetime(date).date())))
    fig.update_layout(title=title,
                      xaxis_title='Time to Maturity (Years)',
                      yaxis_title=y_axis_title,
                      legend_title='Date',
                      template='plotly_white')
    fig.write_html(out_html, include_plotlyjs='cdn')
    return fig

def main():
    #Load & filter data
    treasury = cfu.load_tidy_CRSP_treasury()
    treasury_filtered = cfu.filter_waggoner_treasury_data(treasury)

    #Pick dates to plot
    dates_to_plot = pick_dates(treasury_filtered, n=3)

    #Build curves with conversions
    curves_dict = build_converted_curve(treasury_filtered, dates_to_plot, dt_fwd=0.25)

    #Plot Discount Factors
    plot_curves(curves_dict, 
                x_col="T", y_col="discount",
                title="MCC Discount Curves",
                y_axis_title="Discount Factor",
                out_html=CHARTS_DIR / "mcc_discount_factors.html")
    #Plot Spot Rates
    plot_curves(curves_dict, 
                x_col="T", y_col="spot_cc",
                title="Spot (Zero) Curves (Continuously Compounded)",
                y_axis_title="Spot Rate (Continuously Compounded)",
                out_html=CHARTS_DIR / "mcc_spot_rates_cc.html")
    #Plot forward rates
    plot_curves(curves_dict, 
                x_col="T", y_col="forward_instant_cc",
                title="Instantaneous Forward Curves (Continuously Compounded)",
                y_axis_title="Instantaneous Forward Rate",
                out_html=CHARTS_DIR / "mcc_forward_rates.html")
    
    print(f'Discount, Spot, and Forward curve plots saved to {CHARTS_DIR}')

if __name__ == "__main__":   
    main()