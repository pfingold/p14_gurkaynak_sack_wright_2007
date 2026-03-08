"""
Runs the McCulloch (1975) yield curve replication method (a discount-curve representation
of the yield curve to compute spot/zero rates and forward rates) and saves the outputs for analysis

Inputs:
  - DATA_DIR/tidy_CRSP_treasury.parquet   (produced by tidy_CRSP_treasury.py)

Outputs (in-sample):
    - DATA_DIR/mcc_discount_curve.parquet
    - DATA_DIR/mcc_discount_curve_nodes.csv
    - DATA_DIR/mcc_bond_fits.parquet
    - DATA_DIR/mcc_fit_quality_by_date.csv
    - DATA_DIR/mcc_error_metrics.csv

Outputs (out-of-sample):
    - DATA_DIR/mcc_oos_bond_fits.parquet
    - DATA_DIR/mcc_oos_fit_quality_by_date.csv
    - DATA_DIR/mcc_oos_error_metrics.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from settings import config
import curve_fitting_utils as cfu
import mcc1975_yield_curve as mcc

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def _collect_results(results):
    """Collect model outputs into consolidated result tables."""
    curves, nodes, bond_fits, fit_quality = [], [], [], []
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

    return (
        pd.concat(curves, ignore_index=True),
        pd.concat(nodes, ignore_index=True),
        pd.concat(bond_fits, ignore_index=True),
        pd.DataFrame(fit_quality).sort_values("date"),
    )


def main(start_date=None, end_date=None, output_prefix=""):
    """Run the module's main workflow."""
    df = cfu.load_tidy_CRSP_treasury(DATA_DIR)
    filter_kwargs = {}
    if start_date is not None:
        filter_kwargs["start_date"] = start_date
    if end_date is not None:
        filter_kwargs["end_date"] = end_date
    df_filtered = cfu.filter_waggoner_treasury_data(df, **filter_kwargs)
    in_sample, out_of_sample = cfu.split_in_out_sample_data(df_filtered)

    p = output_prefix

    # --- In-sample ---
    print("Running McCulloch in-sample...")
    in_sample_results = mcc.run_mcculloch(in_sample)

    curves_df, nodes_df, bonds_df, fit_quality_df = _collect_results(in_sample_results)
    err_df = cfu.get_full_error_metrics(in_sample_results).reset_index().rename(columns={"index": "bucket"})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    curves_df.to_parquet(DATA_DIR / f"{p}mcc_discount_curve.parquet", index=False)
    nodes_df.to_csv(DATA_DIR / f"{p}mcc_discount_curve_nodes.csv", index=False)
    bonds_df.to_parquet(DATA_DIR / f"{p}mcc_bond_fits.parquet", index=False)
    fit_quality_df.to_csv(DATA_DIR / f"{p}mcc_fit_quality_by_date.csv", index=False)
    err_df.to_csv(DATA_DIR / f"{p}mcc_error_metrics.csv", index=False)

    # --- Out-of-sample ---
    print("Running McCulloch out-of-sample...")
    oos_results = mcc.run_mcculloch(out_of_sample, pre_trained_results=in_sample_results)

    _, _, oos_bonds_df, oos_fit_quality_df = _collect_results(oos_results)
    oos_err_df = cfu.get_full_error_metrics(oos_results).reset_index().rename(columns={"index": "bucket"})

    oos_bonds_df.to_parquet(DATA_DIR / f"{p}mcc_oos_bond_fits.parquet", index=False)
    oos_fit_quality_df.to_csv(DATA_DIR / f"{p}mcc_oos_fit_quality_by_date.csv", index=False)
    oos_err_df.to_csv(DATA_DIR / f"{p}mcc_oos_error_metrics.csv", index=False)

    print("Wrote McCulloch outputs to:", DATA_DIR.resolve())


if __name__ == "__main__":
    main()