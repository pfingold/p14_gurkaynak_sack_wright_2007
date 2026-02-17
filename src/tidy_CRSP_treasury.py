"""
Generate a tidy data set of the data used before Yield Curve construction
& analysis. Data is sourced from CRSP Treasury data, and is available in 
the `data/` directory.

Data Sources:
- TFZ_DAILY: daily quotes
- TFZ_INFO: bond characteristics
- TFZ_consolidated: merged quotes + characteristics
- TFZ_with_runness: merged quotes + characteristics + runness indicators

Here, use TFZ_with_runness to generate a tidy data set, as it contains
    all the relevant fields for curve estimation & sample selection
"""

from pathlib import Path
import numpy as np
import pandas as pd

def load_CRSP_treasury_data(data_dir):
    "Load CRSP Treasury data from the given directory, and return a tidy DataFrame."
    path = data_dir / "TFZ_with_runness.parquet"
    df = pd.read_parquet(path)
    return df

def standardize_column_names(df):
    "Standardize column names for consistency and ease of use."
    output_df = df.copy()
    # Clean column names
    output_df = output_df.rename(columns={
        'caldt': 'date',
        'tcusip': 'cusip',
        'tcouprt': 'coupon',
        'tdduratn': 'duration',
        'tdatdt': 'issue_date',
        'tmatdt': 'maturity_date',
        'tdbid': 'bid',
        'tdask': 'ask',
        'tdaccint': 'accrued_interest',
        'tdyld': 'yield',
        'price': 'price_raw',
        'itype': 'itype',
        'run': 'run',
        #'original_maturity': 'original_maturity',
        'days_to_maturity': 'ttm_days',
        'years_to_maturity': 'ttm_years',
        'callable': 'callable',
        'kytreasno': 'kytreasno',
        'kycrspid': 'kycrspid',
    })
    # Convert date columns to datetime format
    output_df['date'] = pd.to_datetime(output_df['date'])
    output_df['issue_date'] = pd.to_datetime(output_df['issue_date'])
    output_df['maturity_date'] = pd.to_datetime(output_df['maturity_date'])

    # Convert numeric columns to appropriate types
    numeric_columns = [
        'coupon', 'duration', 'bid', 'ask', 'accrued_interest', 'yield', 'price_raw',
        'ttm_days', 'ttm_years', 'run', 'itype'
    ]
    for col in numeric_columns:
        if col in output_df.columns:
            output_df[col] = pd.to_numeric(output_df[col], errors='coerce')

    # Convert callable column to boolean if it exists
    if 'callable' in output_df.columns:
        output_df['callable'] = output_df['callable'].astype(bool)

    return output_df

def add_relevant_fields(df):
    """Creates relevant fields required for curve estimation & analysis, including:
    - mid price
    - time-to-maturity
    - runness indicators
    Additionally provies 'sanity checks' to ensure no errors arise later
    """
    output_df = df.copy()
    # Calculate mid price
    output_df['mid_price'] = (output_df['bid'] + output_df['ask']) / 2
    
    # Calculate time-to-maturity (if not in data)
    if 'ttm_days' not in output_df.columns:
        output_df['ttm_days'] = (output_df['maturity_date'] - output_df['date']).dt.days
    if 'ttm_years' not in output_df.columns:
        output_df['ttm_years'] = output_df['ttm_days'] / 365.0
   
    # Sanity Checks:
    output_df['valid_quote'] = (
        np.isfinite(output_df['bid']) &
        np.isfinite(output_df['ask']) &
        (output_df['bid'] > 0) &
        (output_df['ask'] > 0) &
        (output_df['ask'] >= output_df['bid'])
    )
    output_df['nonnegative_maturity'] = np.isfinite(output_df['ttm_days']) & (output_df['ttm_days'] >= 0)

    output_df['clean'] = output_df['valid_quote'] & output_df['nonnegative_maturity']
    
    # Runness Indicators (GSW)
    output_df['is_on_the_run'] = output_df['run'] == 0
    output_df['is_first_off_the_run'] = output_df['run'] == 1
    output_df['is_off_the_run'] = output_df['run'] >= 2

    # Waggoner uses Months
    output_df['month'] = output_df['date'].dt.to_period('M').dt.to_timestamp('M')

    # Maturity Flags (Waggoner excludes under 30 days and under 1y from curve estimation)
    output_df['is_under_30d'] = output_df['ttm_days'] < 30
    output_df['is_under_3m'] = output_df['ttm_days'] < 90
    output_df['is_under_1y'] = output_df['ttm_days'] < 365

    # GSW also excludes 20yr maturity post 1996
    output_df['is_20y'] = (output_df['ttm_years'] > 18) & (output_df['ttm_years'] < 22)
    output_df['is_20yr_post_1996'] = (output_df['date'] >= '1996-01-01') & (output_df['is_20y'])

    # Instrument Type Flags
    output_df['is_itype1'] = output_df['itype'] == 1
    output_df['is_itype2'] = output_df['itype'] == 2

    # Callable Flag (GSW excludes callable bonds from curve estimation)
    output_df['is_callable'] = output_df['callable']
   
    return output_df

def select_relevant_cols(df):
    "Keeps the relevant columns for curve estimation & analysis."
    cols = [
        # identifiers
        'date', 'cusip', 'month', 'kytreasno', 'kycrspid',
        # bond chracteristics
        'issue_date', 'maturity_date', 'coupon', 'itype', 'run',
        # quotes & prices
        'bid', 'ask', 'mid_price', 'accrued_interest', 'yield', 'price_raw',
        # maturity measures & duration
        'ttm_days', 'ttm_years', 'duration',
        # flags
        'is_on_the_run', 'is_first_off_the_run', 'is_off_the_run',
        'is_under_30d', 'is_under_3m', 'is_under_1y',
        'is_20y', 'is_20yr_post_1996',
        'is_itype1', 'is_itype2',
        'is_callable',
        'valid_quote', 'nonnegative_maturity', 'clean',
    ]
    cols = [col for col in cols if col in df.columns]
    output_df = df[cols].sort_values(['date', 'ttm_years', 'cusip'])
    return output_df

def generate_tidy_CRSP_treasury_data(tidy_df, output_dir):
    "Generates the tidy CRSP Treasury data set"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tidy_CRSP_treasury.parquet"
    tidy_df.to_parquet(output_path, index=False)
    return output_path


def main(data_dir = '../_data', output_dir = '../_output'):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Load raw data
    df = load_CRSP_treasury_data(data_dir)

    # Standardize column names
    df = standardize_column_names(df)

    # Add relevant fields for curve estimation & analysis
    df = add_relevant_fields(df)

    # Select relevant columns for curve estimation & analysis
    tidy_df = select_relevant_cols(df)

    # Generate tidy CRSP Treasury data set
    output_path = generate_tidy_CRSP_treasury_data(tidy_df, output_dir)
    print(f"Wrote tidy CRSP Treasury data set saved to: {output_path}")

if __name__ == "__main__":
    main()