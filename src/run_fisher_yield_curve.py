"""
Runs the Fisher (1995) forward curve replication method
 and saves the outputs for analysis

Inputs:
  - OUTPUT_DIR/tidy_CRSP_treasury.parquet   (produced by tidy_CRSP_treasury.py)

Outputs (to DATA_DIR):
  - fisher_forward_curve.parquet
  - fisher_forward_curve_nodes.parquet
  - fisher_bond_fits.parquet
  - fisher_fit_quality_by_date.parquet
  - fisher_error_metrics.parquet

Notes:
- fisher1995_yield_curve.run_fisher expects the sample DataFrame to contain:
    date, cusip, ttm_days, mid_price, accrued_interest, bid, ask, duration,
    maturity_date, coupon (needed for cashflows via curve_fitting_utils)
  and optionally first_coupon_date.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from settings import config
import error_metrics
import curve_fitting_utils as cfu
import fisher1995_yield_curve as fisher

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

def _require_exists(p, label):
    if not p.exists():
        raise FileNotFoundError(f"Required input file for {label} not found at {Path(p).resolve()}")

def main():
    #Load tidy CRSP treasury data
    df = cfu.load_tidy_CRSP_treasury(OUTPUT_DIR)
    df_filtered = cfu.filter_waggoner_treasury_data(df)
    in_sample, out_of_sample = cfu.split_in_out_sample_data(df_filtered)

    #Run Fisher Replication
    results = fisher.run_fisher(in_sample)

    curves= []
    nodes = []
    bond_fits = []
    fit_quality = []

    for dt, out in results.items():
        c = out["curve"].copy()
        if "T" in c.columns:
            c = c.rename(columns={"T": "t"})
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
            "lambda": float(out.get("lambda", np.nan)),
            "wmae": float(out["wmae"]),
            "hit_rate": float(out["hit_rate"]),
        })

    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    nodes_df = pd.concat(nodes, ignore_index=True) if nodes else pd.DataFrame()
    bonds_df = pd.concat(bond_fits, ignore_index=True) if bond_fits else pd.DataFrame()
    fit_quality_df = pd.DataFrame(fit_quality).sort_values("date") if fit_quality else pd.DataFrame()

    # Overall + by maturity-bin error metrics
    err_df = cfu.get_full_error_metrics(results).reset_index().rename(columns={"index": "bucket"})

    # Save outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    curves_df.to_parquet(DATA_DIR / "fisher_forward_curve.parquet", index=False)
    nodes_df.to_parquet(DATA_DIR / "fisher_forward_curve_nodes.parquet", index=False)
    bonds_df.to_parquet(DATA_DIR / "fisher_bond_fits.parquet", index=False)
    fit_quality_df.to_parquet(DATA_DIR / "fisher_fit_quality_by_date.parquet", index=False)
    err_df.to_parquet(DATA_DIR / "fisher_error_metrics.parquet", index=False)

    print("Wrote Fisher outputs to:", DATA_DIR.resolve())

if __name__ == "__main__":
    main()