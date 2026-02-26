from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset, MonthEnd

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))

START_DATE = "1970-01-01"
END_DATE = "1995-12-31"

def load_tidy_CRSP_treasury(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Loads the tidy CRSP treasury data from the specified output directory"""
    treasury_path = output_dir / "tidy_CRSP_treasury.parquet"
    treasury = pd.read_parquet(treasury_path)
    return treasury

def filter_waggoner_treasury_data(treasury):
    """
    Filters the treasury data according to the following criteria:
        - Bills: exclude those with maturity under 30 days
        - Notes: exclude those with maturity under 1 year
        - Bonds: exclude those with maturity under 1 year
        - Exclude any security flagged as a "flower"
        - Restrict to data between START_DATE and END_DATE
    """
    treasury_filtered = treasury.loc[((treasury.is_bill) & (~treasury.is_under_30d) |
                                    (treasury.is_note) & (~treasury.is_under_1y)) |
                                    (treasury.is_bond) & (~treasury.is_under_1y)]

    treasury_filtered = treasury_filtered.loc[(treasury_filtered.date <= END_DATE) &
                                              (treasury_filtered.date >= START_DATE)]

    treasury_filtered = treasury_filtered.loc[~treasury_filtered.is_flower]

    return treasury_filtered

def split_in_out_sample_data(treasury_filtered):
    """Splits the filtered treasury data into in-sample and out-of-sample groups 
    according to Waggoner's method"""
    # 1) Sort so cumcount follows maturity order within each month
    treasury_filtered = treasury_filtered.sort_values(["date", "maturity_date"])

    # 2) Rank within month by maturity (0,1,2,...)
    treasury_filtered["maturity_rank_in_month"] = treasury_filtered.groupby("date").cumcount()

    # 3) Every other security -> parity of the rank
    treasury_filtered["group"] = (treasury_filtered["maturity_rank_in_month"] % 2)  # 0 or 1

    # Ensure longest maturity each month ends up in the in-sample group
    def enforce_longest_in_sample(group):
        """
        Ensure the longest-maturity security is in group 0.
        If not, flip the entire month's assignment.
        """
        # Longest maturity = last row (already sorted ascending)
        if group.iloc[-1]["group"] == 1:
            group["group"] = 1 - group["group"]
        return group

    treasury_filtered = treasury_filtered.groupby("date", group_keys=False).apply(enforce_longest_in_sample)

    # Create in-sample indicator
    treasury_filtered["is_in_sample"] = treasury_filtered["group"] == 0

    # Build dataframes of in-sample and out-of-sample data
    in_sample = treasury_filtered.loc[treasury_filtered.is_in_sample].copy()
    in_sample = in_sample.drop(["maturity_rank_in_month", "group", "is_in_sample"], axis=1)
    out_of_sample = treasury_filtered.loc[~treasury_filtered.is_in_sample]
    out_of_sample = out_of_sample.drop(["maturity_rank_in_month", "group", "is_in_sample"], axis=1)

    return in_sample, out_of_sample

def get_cashflows_from_bonds(bonds, face=100, freq=2, stub_tol_days=3):
    """Returns cashflows and times for each bond in the input dataframe."""
    cashflows, times = [], []

    months = int(12 / freq)

    for _, row in bonds.iterrows():
        settle = row["date"]
        maturity_date = row["maturity_date"]
        first_coupon = row.get("first_coupon_date", pd.NaT) #account for missing first_coupon_date (treat as NaT, which will be handled as regular schedule)
        cpn_rate = row["coupon"] / 100.0
        coupon_amt = face * cpn_rate / freq

        # Infer first_coupon_date if missing for coupon bonds
        if coupon_amt > 0 and pd.isna(first_coupon):
            month_end = (maturity_date == maturity_date + MonthEnd(0))
            d = maturity_date
            while d > settle:
                prev = d - DateOffset(months=months)
                if month_end:
                    prev = prev + MonthEnd(0)
                if prev <= settle < d:
                    first_coupon = d
                    break
                d = prev
            if pd.isna(first_coupon):
                raise ValueError(f"Could not determine first coupon date for bond with maturity {maturity_date} and settle {settle}")

        if coupon_amt > 0:
            # build coupon schedule forward from first_coupon
            month_end = (first_coupon == first_coupon + MonthEnd(0))

            payment_dates = []
            d = first_coupon
            while d.to_period("M") <= maturity_date.to_period("M"):
                if d > settle:
                    payment_dates.append(d)
                d = d + DateOffset(months=months) + MonthEnd(0) if month_end else d + DateOffset(months=months)

            payment_dates = pd.to_datetime(payment_dates)
            
            # Handle the case of no payment dates
            if len(payment_dates) == 0:
                maturity_t = (maturity_date - settle).days / 365.0
                payment_times = np.array([maturity_t], dtype=float)
                cf = np.array([coupon_amt + face], dtype=float)
            else:
                payment_times = np.array([(dt - settle).days / 365.0 for dt in payment_dates], dtype=float)
                cf = np.full(payment_times.shape[0], coupon_amt, dtype=float)
                cf[-1] += face

                # Stub handling
                first_pay = payment_dates[0]
                prev_coupon = first_pay - DateOffset(months=months)
                if month_end:
                    prev_coupon = prev_coupon + MonthEnd(0)

                regular_days = (first_pay - prev_coupon).days

                issue_date = row.get("issue_date", pd.NaT)
                accrual_start = max(issue_date, prev_coupon) if pd.notna(issue_date) else prev_coupon
                stub_days = (first_pay - accrual_start).days

                if abs(stub_days - regular_days) > stub_tol_days and regular_days != 0:
                    cf[0] = coupon_amt * (stub_days / regular_days)

        else:
            maturity_t = (maturity_date - settle).days / 365.0
            payment_times = np.array([maturity_t], dtype=float)
            cf = np.array([face], dtype=float)

        cashflows.append(cf)
        times.append(payment_times)

    return cashflows, times

