"""
Thank you to Younghun Lee for preparing this script for use in class.
"""
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from settings import config

DATA_DIR = config("DATA_DIR")

def process_issue_date(df):
    """process_issue_date 
    returns a DataFrame with totalTendered and totalAccepted for each issueDate.

    Args:
        df (DataFrame): auction_query.csv
    """

    return df.groupby('issueDate').sum(numeric_only=True)[['totalTendered', 'totalAccepted']]

def process_ontherun(df, start_date='1800-01-01'):
    """
    Processes a DataFrame to return the most recently issued CUSIP for each date, 
    excluding those that are expired.

    Args:
        df (pd.DataFrame): DataFrame containing bond auction data. Expected to have 
                           columns like 'date', 'original_maturity', 'cusip', 'maturityDate', 'issueDate'.
        start_date (str, optional): The starting date from which to consider data. 
                                    Defaults to '1800-01-01'. The actual starting date will be
                                    set as start_date = max(start_date, df.issueDate.min()).

    Returns:
        pd.DataFrame: A DataFrame with columns ['date', 'run', 'original_maturity', 'cusip'], 
                      containing the most recent CUSIPs for each original_maturity and date, 
                      up to the offruns limit, if applicable.
    """

    COLS = ['date', 'run', 'original_maturity', 'type', 'cusip']
    if df.empty:
        return pd.DataFrame(columns=COLS)

    def calculate_run_byterm(df, original_maturity, start_date, dates):
        temp_df = df[(df.original_maturity == original_maturity) & (df.maturityDate >= start_date)].sort_values(
            'issueDate', ascending=False)
        res = []
        for d in dates:
            row = temp_df[(temp_df.issueDate <= d) & (
                d <= temp_df.maturityDate)][['original_maturity', 'type', 'cusip']]
            # if offruns != -1:
            #     row = row.iloc[:offruns+1]
            row = row[~row.duplicated(subset='cusip', keep='first')]
            row['date'] = d
            row['run'] = np.arange(len(row))
            res.append(row)
        res = pd.concat(res).reset_index(drop=True)[COLS]
        return res

    lastday = np.min([pd.Timestamp.today(), df.maturityDate.max()])
    start_date = np.max([pd.to_datetime(start_date), df.issueDate.min()])

    dates = pd.bdate_range(start_date, lastday, name='date')
    types = df.type.unique().tolist()

    firstissue = df.sort_values(['maturityDate', 'issueDate']).groupby(
        'cusip').first().reset_index()

    res = []
    for t in types:
        temp = firstissue[firstissue.type == t]
        terms = temp.original_maturity.unique().tolist()
        for original_maturity in terms:
            res.append(calculate_run_byterm(temp, original_maturity, start_date, dates))
    res = pd.concat(res).sort_values(
        by=['date', 'run', 'original_maturity', 'type']).reset_index(drop=True)
    return res


if __name__ == '__main__':

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
        dat = pd.read_csv(DATA_DIR / 'auction_query.csv',
                          parse_dates=['issueDate', 'maturityDate'])

    sub_cols = ['cusip', 'issueDate', 'maturityDate',
                'type', 'original_maturity', 'totalTendered', 'totalAccepted']

    dat = dat[sub_cols][dat['type'].isin(['Note', 'Bond'])]
    # dat = dat[sub_cols]

    issue_dates = process_issue_date(dat)
    issue_dates.to_csv(DATA_DIR / 'issue_dates.csv')

    preload = process_ontherun(dat)
    preload.to_csv(DATA_DIR / 'ontherun.csv', index=False)
