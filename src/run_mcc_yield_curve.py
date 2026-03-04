"""
Runs the McCulloch (1975) yield curve replication method (a discount-curve representation 
of the yield curve to compute spot/zero rates and forward rates) and saves the outputs for analysis

Input:
    - DATA_DIR/TFZ_with_runness.parquet
Outputs:
    - DATA_DIR/mcc_discount_curve.parquet
    - DATA_DIR/mcc_discount_curve_nodes.parquet
    - DATA_DIR/mcc_bond_fits.parquet
    - DATA_DIR/mcc_fit_quality_by_date.parquet
    - DATA_DIR/mcc_error_metrics.parquet
"""

from pathlib import Path
import pandas as pd
import numpy as np
from settings import config
import curve_fitting_utils as cfu
import mcc1975_yield_curve as mcc

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

def main():
    input_path = OUTPUT_DIR / "tidy_CRSP_treasury.parquet"
    df = pd.read_parquet(input_path)

    #Check column existence & types
    if "date" not in df.columns:
        raise KeyError("Input DataFrame must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    #Mid Price
    if "mid_price" not in df.columns:
        if not {"bid", "ask"}.issubset(df.columns):
            raise ValueError("Need either mid_price OR both bid and ask.")
        df["mid_price"] = 0.5 * (df["bid"].astype(float) + df["ask"].astype(float))

    #Create Sample
    sample_cols = [
    # required for pricing + cashflows
    "date", "cusip", "maturity_date", "coupon",
    # optional but helpful for coupon schedule accuracy
    "first_coupon_date",
    # quote inputs
    "bid", "ask", "mid_price", "accrued_interest",
    # maturity + duration
    "ttm_days", "ttm_years", "duration",
    ]
    sample_cols = [c for c in sample_cols if c in df.columns]

    sample = df[sample_cols].dropna(subset=["date", "cusip", "maturity_date", "coupon", "mid_price"]).copy()
    sample = sample.loc[(sample["ttm_days"] > 0) & (sample["duration"] > 0)]

    #Run MCC Replication
    results = mcc.run_mcculloch(sample)

    curves= []
    nodes = []
    bond_fits = []
    fit_quality = []

    for dt, out in results.items():
        c = out["curve"].copy()
        c["date"] = pd.to_datetime(dt)
        curves.append(c)

        n = out["nodes"].copy()
        n["date"] = pd.to_datetime(dt)
        nodes.append(n)

        b = out["bonds"].copy()
        b["date"] = pd.to_datetime(dt)
        bond_fits.append(b)

        fit_quality.append({
            "date": pd.to_datetime(dt),
            "wmae": float(out["wmae"]),
            "hit_rate": float(out["hit_rate"]),
        })

    curves_df = pd.concat(curves, ignore_index=True)
    nodes_df = pd.concat(nodes, ignore_index=True)
    bonds_df = pd.concat(bond_fits, ignore_index=True)
    fit_quality_df = pd.DataFrame(fit_quality).sort_values("date")

    # Overall + by maturity-bin error metrics
    err_df = cfu.get_full_error_metrics(results).reset_index().rename(columns={"index": "bucket"})

    # Save Outputs
    (DATA_DIR / "mcc_discount_curve.parquet").write_bytes(b"")  # ensures parent exists in some envs
    curves_df.to_parquet(DATA_DIR / "mcc_discount_curve.parquet", index=False)
    nodes_df.to_parquet(DATA_DIR / "mcc_discount_curve_nodes.parquet", index=False)
    bonds_df.to_parquet(DATA_DIR / "mcc_bond_fits.parquet", index=False)
    fit_quality_df.to_parquet(DATA_DIR / "mcc_fit_quality_by_date.parquet", index=False)
    err_df.to_parquet(DATA_DIR / "mcc_error_metrics.parquet", index=False)


if __name__ == "__main__":
    main()