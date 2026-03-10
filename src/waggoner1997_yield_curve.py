"""
Utilities for waggoner1997 yield curve in this project:
- run_waggoner: main wrapper to run the Waggoner (1997) variable roughness penalty 
    yield curve fitting procedure on the provided sample data, with optional pre-trained 
    results for nodes and beta.
- fit_fisher_forward_variable_lambda: core fitting function that implements the variable 
    roughness penalty optimization for fitting the forward rate curve using a cubic spline basis.
- vrp_roughness_matrix: computes the variable roughness penalty matrix K based on the 
    piecewise constant lambda(t) specified in W
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import curve_fitting_utils
import error_metrics
from fisher1995_yield_curve import (
    bspline_basis_list, integrated_basis_matrix,
    sqrt_penalty_from_K, price_and_jac,
    fisher_nodes_equal_counts, bspline_knots_from_nodes,
    fisher_curve_points_to_dfs, fisher_predict_prices,
)

def vrp_roughness_matrix(knots, degree: int = 3, grid_size: int = 1000):
    """
    Waggoner (1997) Variable Roughness Penalty matrix:

        K_vrp_ij = integral lambda(t) B''_i(t) B''_j(t) dt

    where lambda(t) is piecewise constant:
        lambda(t) = 0.1       for 0 <= t <= 1
        lambda(t) = 100       for 1  < t <= 10
        lambda(t) = 100,000   for t  > 10
    """
    knots = np.asarray(knots, float)
    basis = bspline_basis_list(knots, degree)
    d2 = [b.derivative(2) for b in basis]

    t_min = knots[degree]
    t_max = knots[-degree - 1]
    t = np.linspace(t_min, t_max, grid_size)

    lam_t = np.where(t <= 1.0, 0.1, np.where(t <= 10.0, 100.0, 100_000.0))

    dt = np.diff(t)
    w = np.empty_like(t)
    w[0] = 0.5 * dt[0]
    w[-1] = 0.5 * dt[-1]
    w[1:-1] = 0.5 * (dt[:-1] + dt[1:])

    w_vrp = lam_t * w
    B2 = np.column_stack([b(t) for b in d2])  # (grid_size, p)
    return (B2.T * w_vrp) @ B2


def fit_fisher_forward_variable_lambda(
    bonds,
    knots,
    degree: int = 3,
    A_all=None,
    idx_slices=None,
    K=None,
    beta0=None,
    c_all=None,
    bond_idx=None,
):
    """
    Fits beta using Waggoner (1997) Variable Roughness Penalty (VRP).
    Returns fit dict incl. RSS, pricing Jacobian, etc.
    """

    knots = np.asarray(knots, float)
    basis = bspline_basis_list(knots, degree)
    p = len(basis)

    # Build shared structures if not passed
    if A_all is None or idx_slices is None:
        times_all = np.concatenate([b["times"] for b in bonds])
        idx_slices = []
        start = 0
        for b in bonds:
            n = len(b["times"])
            idx_slices.append(slice(start, start + n))
            start += n
        A_all = integrated_basis_matrix(times_all, basis, t0=0.0)

    if K is None:
        K = vrp_roughness_matrix(knots, degree=degree)

    if c_all is None or bond_idx is None:
        c_all = np.concatenate([b["c"] for b in bonds])
        bond_idx = np.repeat(np.arange(len(bonds)), [len(b["c"]) for b in bonds])

    # L s.t. L^T L = K_vrp; lambda weights are baked into K_vrp
    # so ||L beta||^2 = beta^T K_vrp beta = integral lambda(t) [f''(t)]^2 dt
    L = sqrt_penalty_from_K(K, eps=1e-12)

    P_obs = np.array([b["P"] for b in bonds], float)
    N = len(P_obs)

    if beta0 is None:
        beta0 = np.zeros(p)

    # Cache price_and_jac between fun() and jac() — scipy calls both with the
    # same beta on each iteration, so we avoid computing it twice.
    _cache = {"beta": None, "P_hat": None, "Jp": None}

    def _eval(beta):
        if _cache["beta"] is None or not np.array_equal(beta, _cache["beta"]):
            _cache["P_hat"], _cache["Jp"] = price_and_jac(beta, c_all, bond_idx, A_all, N)
            _cache["beta"] = beta.copy()

    def fun(beta):
        _eval(beta)
        r_price = P_obs - _cache["P_hat"]
        r_pen = L @ beta
        return np.concatenate([r_price, r_pen])

    def jac(beta):
        _eval(beta)
        Jr_price = -_cache["Jp"]
        Jr_pen = L
        return np.vstack([Jr_price, Jr_pen])

    res = least_squares(fun, beta0, jac=jac, method="trf")

    beta_hat = res.x
    P_hat, J_price = price_and_jac(beta_hat, c_all, bond_idx, A_all, N)
    r_price = P_obs - P_hat
    RSS = float(r_price @ r_price)

    return {
        "beta": beta_hat,
        "success": bool(res.success),
        "message": res.message,
        "cost": float(res.cost),
        "P_hat": P_hat,
        "resid_price": r_price,
        "RSS": RSS,
        "J_price": J_price,   # dP_hat/dbeta (N x p)
        "knots": knots,
        "degree": degree,
        "A_all": A_all,
        "idx_slices": idx_slices,
        "K": K,
    }


# ----------------
# Full wrapper for Waggoner
# ----------------

def run_waggoner(sample, pre_trained_results=None, node_ratio:int = 3):
    """Runs the Waggoner (1997) variable roughness penalty yield curve fitting procedure 
    on the provided sample data, with optional pre-trained results for nodes and beta."""
    results = {}

    dates = sample["date"].unique()

    prev_beta = None   # warm-start: reuse previous date's solution as beta0

    for idx, DATE in enumerate(dates):
        if idx % 50 == 0:
            print(f"{DATE.to_period('M')}: {idx} / {len(dates)} ({int(idx/len(dates)*100)}%)")

        bonds = sample.loc[sample["date"] == DATE].reset_index(drop=True)

        bonds["ttm"] = bonds["ttm_days"] / 365
        bonds.sort_values(by="ttm", inplace=True)
        n_bonds = bonds.shape[0]

        cashflows, times = curve_fitting_utils.get_cashflows_from_bonds(bonds)

        maturities = bonds["ttm"].to_numpy()
        prices = bonds["mid_price"].to_numpy()
        acc_int = bonds["accrued_interest"].to_numpy()

        bonds_dict = [{"P": prices[i] + acc_int[i], "times": times[i], "c": cashflows[i]} for i in range(n_bonds)]

        if pre_trained_results is not None:
            pre = pre_trained_results[DATE]
            beta_hat = np.asarray(pre["beta_hat"], float)
            knots = np.asarray(pre["knots"], float)

            P_hat_dirty, resid = fisher_predict_prices(beta_hat, knots, bonds_dict, degree=3)
            bonds["model_price"] = P_hat_dirty - acc_int
            bonds["resid"] = resid

            fit_for_curve = {"beta": beta_hat, "knots": knots, "degree": 3}
            curve_df, nodes_df = fisher_curve_points_to_dfs(fit_for_curve)
        else:
            nodes = fisher_nodes_equal_counts(maturities, node_ratio=node_ratio)
            knots = bspline_knots_from_nodes(nodes, degree=3)

            p = len(knots) - 3 - 1
            beta0 = prev_beta if (prev_beta is not None and len(prev_beta) == p) else None
            out = fit_fisher_forward_variable_lambda(bonds_dict, knots, degree=3, beta0=beta0)
            beta_hat = out["beta"]

            bonds["model_price"] = out["P_hat"] - acc_int
            bonds["resid"] = out["resid_price"]

            curve_df, nodes_df = fisher_curve_points_to_dfs(out)
            prev_beta = beta_hat

        wmae = error_metrics.wmae(bonds["model_price"], bonds["bid"], bonds["ask"], bonds["duration"])
        hit_rate = error_metrics.hit_rate(bonds["model_price"], bonds["bid"], bonds["ask"])

        results[DATE] = {
            "beta_hat": beta_hat,
            "knots": knots,
            "bonds": bonds,
            "curve": curve_df,
            "nodes": nodes_df,
            "wmae": wmae,
            "hit_rate": hit_rate,
        }

    return results