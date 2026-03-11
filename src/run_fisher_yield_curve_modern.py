"""
Runs the Fisher (1995) forward curve on a rolling modern sample
(last 20 years of available data) and saves the outputs for analysis

Inputs:
  - DATA_DIR/tidy_CRSP_treasury.parquet   (produced by tidy_CRSP_treasury.py)

Outputs (in-sample):
  - DATA_DIR/modern_fisher_forward_curve.parquet
  - DATA_DIR/modern_fisher_bond_fits.parquet
  - DATA_DIR/modern_fisher_fit_quality_by_date.csv
  - DATA_DIR/modern_fisher_error_metrics.csv

Outputs (out-of-sample):
  - DATA_DIR/modern_fisher_oos_bond_fits.parquet
  - DATA_DIR/modern_fisher_oos_error_metrics.csv
"""
from pathlib import Path
from pandas.tseries.offsets import DateOffset
from settings import config
import curve_fitting_utils as cfu
from run_fisher_yield_curve import main

DATA_DIR = Path(config("DATA_DIR"))

if __name__ == "__main__":
    df = cfu.load_tidy_CRSP_treasury(DATA_DIR)
    end_date = df["date"].max()
    start_date = end_date - DateOffset(years=20)
    main(start_date=start_date, end_date=end_date, output_prefix="modern_", node_ratio=6)
