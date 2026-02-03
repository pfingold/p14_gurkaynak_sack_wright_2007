"""
Example plot using CRSP stock data:
- Boxplot of Market Capitalization by Primary Exchange
"""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

def main():
    crsp_data = pd.read_parquet(DATA_DIR / "CRSP_stock_ciz.parquet")

    df = crsp_data.copy()
    df['shrout'] = df['shrout'] * 1000  # convert to actual shares
    df['mktcap'] = df['mthprc'].abs() * df['shrout']
    
    #Filter out non-positive market caps
    df = df.loc[
        (df['mktcap'] > 0) & (df['mktcap'].notna())
        & (df['primaryexch'].notna())
    ]

    df['log_mktcap'] = np.log(df['mktcap'])
    # Order exchanges by median log market cap
    order = (
        df.groupby('primaryexch')['log_mktcap']
        .median()
        .sort_values()
        .index
    )

    organized_df = [df.loc[df['primaryexch'] == exch, 'log_mktcap'] for exch in order]
    exchanges = {
        'N': 'NYSE',
        'A': 'AMEX',
        'Q': 'NASDAQ',
        'B': 'BATS',
        'R': 'NYSE ARCA',
        'I': 'IEX',
        'X': 'UNKNOWN'
    }
    labels = [exchanges.get(code, code) for code in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(organized_df, tick_labels=labels, showfliers=False)

    ax.set_title('Log Market Capitalization by Primary Exchange')
    ax.set_xlabel('Primary Exchange')
    ax.set_ylabel('Log Market Capitalization (USD)')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    outpath = OUTPUT_DIR / "crsp_stock_log_mktcap_by_exchange.png"
    fig.savefig(outpath)
    print(f"Saving plot to {outpath}")
    plt.close(fig)

if __name__ == "__main__":
    main()
