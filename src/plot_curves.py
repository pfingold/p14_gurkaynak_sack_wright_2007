"""
Executes the replication pipeline and plots the resulting curves:
    - Load & filter CRSP Treausry data (curve_fitting_utils.py)
    - Build MCC discount curves (mcc1975_yield_curve.py)
    - Convert discount to spot & forward (curve_conversions.py)
    - Overlays the GSW curve for comparison (gsw2007_yield_curve.py)
    - Plot the curves for selected dates (this file)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import curve_fitting_utils as cfu
import curve_conversions as cc
import mcc1975_yield_curve as mcc

from settings import config 
import pull_yield_curve_data

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
CHARTS_DIR = ROOT_DIR / "docs" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(config("DATA_DIR"))
    
def pick_dates(df, n=3):
    "Pick n evenly spaced dates from the available sample dates in the DataFrame"
    dates = pd.to_datetime(sorted(df["date"].unique()))
    if len(dates) == 0:
        raise ValueError("No dates found in the DataFrame.")
    if n >= len(dates):
        return list(dates)
    idx = np.linspace(0, len(dates) - 1, n).round().astype(int)
    return list(dates[idx])

def build_converted_curve(treasury_filtered, dates_to_plot, dt_fwd):
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

### GSW Overlay ###
def _gsw_spot(maturities, params):
    tau1, tau2, beta1, beta2, beta3, beta4 = params
    t = np.asarray(maturities, dtype=float)
    t_safe = np.where(t == 0.0, 1e-6, t)  #handle t=0 case

    #avoid divide by zero error
    tau1_exp = (1 - np.exp(-t_safe / tau1)) / (t_safe / tau1)
    tau2_exp = (1 - np.exp(-t_safe / tau2)) / (t_safe / tau2)

    return (
        beta1 +
        beta2 * tau1_exp +
        beta3 * (tau1_exp - np.exp(-t_safe / tau1)) +
        beta4 * (tau2_exp - np.exp(-t_safe / tau2))
    )

def build_converted_gsw_curve(gsw_date, t_grid, dt_fwd=0.25):
    """Builds a curve DF with columns T, discount from GSW params for gsw_date,
    and then applies curve_conversions to get spot & forward columns"""
    gsw_date = pd.to_datetime(gsw_date)

    #Load Fed GSW/NSS parameters
    params_raw = pull_yield_curve_data.load_fed_yield_curve_all(data_dir=Path(DATA_DIR))

    #Ensure datetime index
    if not isinstance(params_raw.index, pd.DatetimeIndex):
        if "date" in params_raw.columns:
            params_df = params_raw.copy()
            params_df["date"] = pd.to_datetime(params_df["date"])
            params_df = params_df.set_index("date")
        else:
            raise ValueError("Fed parameter table must have a DatetimeIndex or a 'date' column.")
    else:
        params_df = params_raw.copy()

    params_df = params_df.sort_index()

    #exact date if available; else previous available ("asof")
    if gsw_date in params_df.index:
        row = params_df.loc[gsw_date]
    else:
        idx = params_df.index[params_df.index <= gsw_date]
        if len(idx) == 0:
            raise ValueError(f"No GSW params found on/before {gsw_date.date()}.")
        gsw_date = idx[-1]
        row = params_df.loc[gsw_date]
    
    #Maps Fed Column Names to Model Parameters
    tau1 = float(row["TAU1"])
    tau2 = float(row["TAU2"])

    b0 = float(row["BETA0"])
    b1 = float(row["BETA1"])
    b2 = float(row["BETA2"])
    b3 = float(row["BETA3"])

    #Scale to decimals if betas are percentages
    scale = 0.01 if max(abs(b0), abs(b1), abs(b2), abs(b3)) > 1.0 else 1.0
    beta1 = b0 * scale
    beta2 = b1 * scale
    beta3 = b2 * scale
    beta4 = b3 * scale

    params = (tau1, tau2, beta1, beta2, beta3, beta4)

    spot = _gsw_spot(t_grid, params)
    discount = np.exp(-spot * t_grid)

    gsw_curve = pd.DataFrame({
        "T": t_grid,
        "discount": discount
    })
    gsw_conv = cc.add_spot_and_forwards(gsw_curve, dt=dt_fwd, t_col="T", d_col="discount")
    return gsw_date, gsw_conv


### Plotting Functions ###
def plot_curves(curves_dict, 
                x_col, y_col,
                title, y_axis_title, out_html,
                extra_traces=None):
    """Plot curves from the curves_dict with Plotly"""
    fig = go.Figure()
    for date, df in curves_dict.items():
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], 
                    mode='lines', name=str(pd.to_datetime(date).date())))

    #Account for any extra traces (e.g. GSW overlay)
    if extra_traces:
        for tr in extra_traces:
            df2 = tr["df"]
            xcol = tr.get("x_col", x_col)
            ycol = tr.get("y_col", y_col)
            fig.add_trace(
                go.Scatter(
                    x=df2[xcol],
                    y=df2[ycol],
                    mode="lines",
                    name=tr["name"],
                    line=dict(dash=tr.get("dash", "dash")),
                )
            )

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
    GSW_DATE = max(dates_to_plot)  # Date of GSW curve to overlay for comparison

    t_grid = np.linspace(0.0, 30.0, 30*48 +1) #approx weekly grid for 0-30y
    DT_FWD = t_grid[1] - t_grid[0]  # Time step for forward rate conversions (in years)

    #Build curves with conversions
    curves_dict = build_converted_curve(treasury_filtered, dates_to_plot, dt_fwd=DT_FWD)

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
    
    print(f'MCC Discount, Spot, and Forward curve plots saved to {CHARTS_DIR}')

    ### Overlay GSW curve for comparison ###

        ### (1) Date-specific MCC vs GSW plots (3 dates × 3 curve types = 9 files) ###
    for d in dates_to_plot:
        d = pd.to_datetime(d)

        # MCC curve for this date only
        mcc_one = {d: curves_dict[d]}

        # GSW curve "asof" this date
        gsw_actual_date, gsw_df = build_converted_gsw_curve(d, t_grid, DT_FWD)
        gsw_label = f"GSW ({gsw_actual_date.date()})"

        date_tag = d.strftime("%Y-%m-%d")

        # Discount
        plot_curves(
            mcc_one,
            x_col="T", y_col="discount",
            title=f"Discount Curves: MCC vs GSW ({date_tag})",
            y_axis_title="Discount Factor",
            out_html=CHARTS_DIR / f"discount_mcc_vs_gsw_{date_tag}.html",
            extra_traces=[{"name": gsw_label, "df": gsw_df, "dash": "dash"}],
        )

        # Spot (cc)
        plot_curves(
            mcc_one,
            x_col="T", y_col="spot_cc",
            title=f"Spot Curves (CC): MCC vs GSW ({date_tag})",
            y_axis_title="Spot Rate (Continuously Compounded)",
            out_html=CHARTS_DIR / f"spot_cc_mcc_vs_gsw_{date_tag}.html",
            extra_traces=[{"name": gsw_label, "df": gsw_df, "dash": "dash"}],
        )

        # Instantaneous forward (cc)
        plot_curves(
            mcc_one,
            x_col="T", y_col="forward_instant_cc",
            title=f"Instantaneous Forward Curves (CC): MCC vs GSW ({date_tag})",
            y_axis_title="Instantaneous Forward Rate",
            out_html=CHARTS_DIR / f"fwd_instant_cc_mcc_vs_gsw_{date_tag}.html",
            extra_traces=[{"name": gsw_label, "df": gsw_df, "dash": "dash"}],
        )

    print(f"Date-specific MCC vs GSW plots saved to {CHARTS_DIR}")

if __name__ == "__main__":   
    main()