"""
Example plot using CRSP stock data:
- Boxplot of Market Capitalization by Primary Exchange
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(DATA_DIR / "CRSP_stock_ciz.parquet")

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

    exchanges = {
        'N': 'NYSE',
        'A': 'AMEX',
        'Q': 'NASDAQ',
        'B': 'BATS',
        'R': 'NYSE ARCA',
        'I': 'IEX',
        'X': 'UNKNOWN'
    }
    fig = go.Figure()

    for exch in order:
        fig.add_trace(
            go.Box(
                y=df.loc[df['primaryexch'] == exch, 'log_mktcap'],
                name=exchanges.get(exch, exch),
                boxpoints=False
            )
        )

    fig.update_layout(
        title="Log Market Capitalization by Primary Exchange",
        yaxis_title="Log Market Capitalization (USD)",
        xaxis_title="Primary Exchange",
        template="plotly_white"
    )

    outpath = OUTPUT_DIR / "crsp_stock_log_mktcap_by_exchange.html"
    plot(fig, filename=str(outpath), auto_open=False)
    print(f'Saved interactive plot as html to: {outpath}')

if __name__ == "__main__":
    main()
