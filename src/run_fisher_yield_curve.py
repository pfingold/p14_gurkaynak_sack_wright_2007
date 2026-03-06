"""
Runs the Fisher (1995) forward curve replication method and saves the outputs for analysis

Inputs:
  - DATA_DIR/tidy_CRSP_treasury.parquet   (produced by tidy_CRSP_treasury.py)

Outputs (in-sample):
  - DATA_DIR/fisher_forward_curve.parquet
  - DATA_DIR/fisher_forward_curve_nodes.csv
  - DATA_DIR/fisher_bond_fits.parquet
  - DATA_DIR/fisher_fit_quality_by_date.csv
  - DATA_DIR/fisher_error_metrics.csv

Outputs (out-of-sample):
  - DATA_DIR/fisher_oos_bond_fits.parquet
  - DATA_DIR/fisher_oos_fit_quality_by_date.csv
  - DATA_DIR/fisher_oos_error_metrics.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from settings import config
import curve_fitting_utils as cfu
import fisher1995_yield_curve as fisher

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def _collect_results(results):
    curves, nodes, bond_fits, fit_quality = [], [], [], []
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

    return (
        pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        pd.concat(nodes, ignore_index=True) if nodes else pd.DataFrame(),
        pd.concat(bond_fits, ignore_index=True) if bond_fits else pd.DataFrame(),
        pd.DataFrame(fit_quality).sort_values("date") if fit_quality else pd.DataFrame(),
    )


def main(start_date=None, end_date=None, output_prefix=""):
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
    print("Running Fisher in-sample...")
    in_sample_results = fisher.run_fisher(in_sample)

    curves_df, nodes_df, bonds_df, fit_quality_df = _collect_results(in_sample_results)
    err_df = cfu.get_full_error_metrics(in_sample_results).reset_index().rename(columns={"index": "bucket"})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    curves_df.to_parquet(DATA_DIR / f"{p}fisher_forward_curve.parquet", index=False)
    nodes_df.to_csv(DATA_DIR / f"{p}fisher_forward_curve_nodes.csv", index=False)
    bonds_df.to_parquet(DATA_DIR / f"{p}fisher_bond_fits.parquet", index=False)
    fit_quality_df.to_csv(DATA_DIR / f"{p}fisher_fit_quality_by_date.csv", index=False)
    err_df.to_csv(DATA_DIR / f"{p}fisher_error_metrics.csv", index=False)

    # --- Out-of-sample ---
    print("Running Fisher out-of-sample...")
    oos_results = fisher.run_fisher(out_of_sample, pre_trained_results=in_sample_results)

    _, _, oos_bonds_df, oos_fit_quality_df = _collect_results(oos_results)
    oos_err_df = cfu.get_full_error_metrics(oos_results).reset_index().rename(columns={"index": "bucket"})

    oos_bonds_df.to_parquet(DATA_DIR / f"{p}fisher_oos_bond_fits.parquet", index=False)
    oos_fit_quality_df.to_csv(DATA_DIR / f"{p}fisher_oos_fit_quality_by_date.csv", index=False)
    oos_err_df.to_csv(DATA_DIR / f"{p}fisher_oos_error_metrics.csv", index=False)

    print("Wrote Fisher outputs to:", DATA_DIR.resolve())


if __name__ == "__main__":
    main()