"""DOCSTRING"""

import numpy as np
import pandas as pd

import error_metrics
import curve_fitting_utils

ERROR_COLS = ["bid", "ask", "duration", "model_price", "ttm"]
ID_COLS = ["date", "cusip"]

def get_nodes(bonds, maturities):
    """DOCSTRING"""
    n_bonds = len(bonds)
    n_knots = int(np.sqrt(n_bonds))
    Tmax = max(maturities)
    ncoef = n_knots + 1
    interior_knots = []
    for j in range(2, n_knots):
        h = int(np.floor((j-1)*n_bonds/(ncoef-2)))
        theta = (j-1)*n_bonds/(ncoef-2) - h
        d_j = maturities[h] + theta*(maturities[h+1] - maturities[h])
        interior_knots.append(d_j)

    d = np.r_[
        [0.0],
        interior_knots,
        [Tmax]
    ]

    return d, ncoef

def build_basis_matrix(m_grid, d, k):
    """
    Build the McCulloch (Appendix A) cubic spline basis matrix F where
    F[:, j-1] = f_j(m) for j=1..k, using formulas (A.2)-(A.6).

    Parameters
    ----------
    m_grid : array-like
        Points m at which to evaluate the basis (e.g., cashflow times).
    d : array-like
        Knot points d_1,...,d_{k-1}. Must be length k-1.
    k : int
        Number of coefficients/basis functions.

    Returns
    -------
    F : np.ndarray shape (len(m_grid), k)
        Basis matrix with columns f_1,...,f_k evaluated at m_grid.
    """
    m = np.asarray(m_grid, dtype=float)
    d = np.asarray(d, dtype=float)

    if len(d) != k - 1:
        raise ValueError(f"Need {k-1} knots for k={k}, got {len(d)}.")
    if np.any(np.diff(d) < 0):
        raise ValueError("Knots d must be nondecreasing.")

    F = np.zeros((m.size, k), dtype=float)

    def f_j(mvals, j):
        """
        Evaluate f_j(m) for j=1..k-1 using A.2-A.5 (paper indexing, 1-based).
        """
        # Special-case note in paper: set d_{j-1} = d_j = 0 when j=1.
        if j == 1:
            d_jm1 = 0.0      # d_0 = 0
            d_j   = d[0]     # d_1 (should be 0 if you want the classic setup)
            d_jp1 = d[1]     # d_2
        else:
            d_jm1 = d[j-2]
            d_j   = d[j-1]
            d_jp1 = d[j] if (j < k-1) else d[-1]

        out = np.zeros_like(mvals, dtype=float)

        # (A.2) m < d_{j-1} => 0 (already)

        # (A.3) d_{j-1} <= m < d_j
        mask2 = (mvals >= d_jm1) & (mvals < d_j)
        if np.any(mask2):
            denom = 6.0 * (d_j - d_jm1)
            if denom != 0.0:
                out[mask2] = ((mvals[mask2] - d_jm1) ** 3) / denom

        # (A.4) d_j <= m < d_{j+1}
        mask3 = (mvals >= d_j) & (mvals < d_jp1)
        if np.any(mask3):
            c = d_j - d_jm1
            e = mvals[mask3] - d_j
            denom = 6.0 * (d_jp1 - d_j)
            if denom != 0.0:
                out[mask3] = (c**2)/6.0 + (c*e)/2.0 + (e**2)/2.0 - (e**3)/denom
            else:
                out[mask3] = (c**2)/6.0 + (c*e)/2.0 + (e**2)/2.0

        # (A.5) d_{j+1} <= m
        mask4 = (mvals >= d_jp1)
        if np.any(mask4):
            out[mask4] = (d_jp1 - d_jm1) * (
                (2.0 * d_jp1 - d_j - d_jm1) / 6.0 + (mvals[mask4] - d_jp1) / 2.0
            )

        return out

    # Columns 1..k-1: f_1..f_{k-1}
    for j in range(1, k):
        F[:, j - 1] = f_j(m, j)

    # Last column: (A.6) f_k(m) = m
    F[:, k - 1] = m
    return F

def discount(beta, t, d, ncoef):
    t = np.asarray(t, dtype=float)

    # F should be shape (len(t_i), ncoef)
    F = build_basis_matrix(t, d, ncoef)

    # discount at each cashflow date
    D = 1.0 + F @ beta

    return D

def predict_prices(beta, cashflows, times, d, ncoef, accrued_interest):
    """
    cashflows: list (or iterable) of 1D arrays, cf_i[j] cashflow at time t_i[j]
    times:     list (or iterable) of 1D arrays, t_i[j] cashflow time in years for bond i
    """
    beta = np.asarray(beta, dtype=float)
    model_prices = []

    for i, (t_i, cf_i) in enumerate(zip(times, cashflows)):
        D = discount(beta, t_i, d, ncoef)
        
        P_i_dirty = np.sum(cf_i * D)
        P_i = P_i_dirty - accrued_interest[i]
        model_prices.append(P_i)

    return np.asarray(model_prices, dtype=float)

import numpy as np

def fit(cashflows, times, prices_clean, ai, d, ncoef):
    X_rows = []
    rhs = []
    for cf_i, t_i, P_i, ai_i in zip(cashflows, times, prices_clean, ai):
        F_i = build_basis_matrix(t_i, d, ncoef)      # (n_cf, ncoef)
        C_i = np.sum(cf_i)
        X_i = np.sum(cf_i[:, None] * F_i, axis=0)        # (ncoef,)
        X_rows.append(X_i)
        rhs.append(P_i - (C_i - ai_i))
    X = np.vstack(X_rows)
    rhs = np.asarray(rhs)
    beta_hat, *_ = np.linalg.lstsq(X, rhs, rcond=None)
    return beta_hat

def discount_curve(bonds, beta_hat, d, ncoef):
    """DOCSTRING"""
    T_grid = np.linspace(0, np.ceil(max(bonds["ttm"])), 200)

    F_nodes = build_basis_matrix(d, d, ncoef)
    D_nodes = 1.0 + F_nodes @ beta_hat

    F = build_basis_matrix(T_grid, d, ncoef)
    D = 1.0 + F @ beta_hat
    
    nodes_df = pd.DataFrame({"T": d,
                             "discount": D_nodes})

    curve_df = pd.DataFrame({"T": T_grid,
                             "discount": D})
    
    return curve_df, nodes_df


def run_mcculloch(sample):
    results = {}

    dates = sample["date"].unique()

    for idx, DATE in enumerate(dates):
        if idx % 50 == 0:
            print(f"{DATE.to_period("M")}: {idx} / {len(dates)} ({int(idx/len(dates)*100)}%)")

        bonds = sample.loc[sample["date"] == DATE].reset_index(drop=True)

        bonds["ttm"] = bonds["ttm_days"] / 365

        bonds = bonds.sort_values(by="ttm", ascending=True)

        prices = bonds["mid_price"].to_numpy()
        acc_int = bonds["accrued_interest"].to_numpy()
        maturities = bonds["ttm"].to_numpy()

        cashflows, times = curve_fitting_utils.get_cashflows_from_bonds(bonds)

        d, ncoef = get_nodes(bonds, maturities)

        beta_hat = fit(cashflows, times, prices, acc_int, d, ncoef)
        
        P_hat = predict_prices(beta_hat, cashflows, times, d, ncoef, acc_int)
        resid = P_hat - prices

        bonds["model_price"] = P_hat
        bonds["residual"] = resid

        curve_df, nodes_df = discount_curve(bonds, beta_hat, d, ncoef)
        
        wmae = error_metrics.wmae(bonds["model_price"], bonds["bid"], bonds["ask"], bonds["duration"])
        hit_rate = error_metrics.hit_rate(bonds["model_price"], bonds["bid"], bonds["ask"])
        
        
        results[DATE] = {"beta_hat": beta_hat,
                        "bonds": bonds,
                        "curve": curve_df,
                        "nodes": nodes_df,
                        "wmae": wmae,
                        "hit_rate": hit_rate,
                        }
        
    return results

def get_full_error_metrics(results, id_cols=ID_COLS, error_cols=ERROR_COLS):
    ttm_bins = [(0, 1), (1, 3), (3, 5), (5, 10), (10, np.inf)]
    wmae_list = []
    hit_rate_list = []

    preds = pd.concat([results[r]["bonds_df"][id_cols + error_cols].set_index(id_cols) for r in results], axis=0)

    for start, stop in ttm_bins:
        preds_bin = preds.loc[(preds["ttm"] >= start) & (preds["ttm"] < stop)]

        wmae = error_metrics.wmae(
            preds_bin["model_price"],
            preds_bin["bid"],
            preds_bin["ask"],
            preds_bin["duration"]
            )
        wmae_list.append(wmae)

        hit_rate = error_metrics.hit_rate(
            preds_bin["model_price"],
            preds_bin["bid"],
            preds_bin["ask"]
            )
        hit_rate_list.append(hit_rate)

    wmae_list.append(
        error_metrics.wmae(
            preds["model_price"],
            preds["bid"],
            preds["ask"],
            preds["duration"]
        ))
    
    hit_rate_list.append(
        error_metrics.hit_rate(
            preds["model_price"],
            preds["bid"],
            preds["ask"]
        ))

    labels = [f"{start}—{stop}"
              if stop < np.inf
              else f">{start}"
              for start, stop in ttm_bins
              ] + ["All"]

    return pd.DataFrame({
        "wmae": wmae_list,
        "hit_rate": hit_rate_list},
        index=labels)

