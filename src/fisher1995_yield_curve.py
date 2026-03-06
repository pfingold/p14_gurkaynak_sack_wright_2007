import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, minimize_scalar
import curve_fitting_utils
import error_metrics

# -----------------------------
# Build knot vector
# -----------------------------

def fisher_nodes_equal_counts(maturities: np.ndarray) -> np.ndarray:
    m = np.asarray(maturities, dtype=float)
    m = m[np.isfinite(m) & (m >= 0)]
    if m.size < 2:
        raise ValueError("Need at least 2 maturities")
    m_sorted = np.sort(m)
    n = m_sorted.size

    n_nodes = max(2, int(n / 3))      # Fisher rule of thumb
    n_interior = max(0, n_nodes - 2)  # excluding 0 and Tmax

    nodes = [0.0]
    if n_interior > 0:
        for j in range(1, n_interior + 1):
            frac = j / (n_interior + 1)
            idx = int(round(frac * (n - 1)))
            nodes.append(float(m_sorted[idx]))
    nodes.append(float(m_sorted[-1]))

    return np.unique(np.array(nodes, dtype=float))

def bspline_knots_from_nodes(nodes: np.ndarray, degree: int = 3) -> np.ndarray:
    x = np.asarray(nodes, dtype=float)
    x = np.unique(x)
    if x.size < 2 or x[0] != 0.0:
        raise ValueError("nodes must include 0 and Tmax")
    if np.any(np.diff(x) <= 0):
        raise ValueError("nodes must be strictly increasing")

    k = int(degree)
    left = np.repeat(x[0], k + 1)
    right = np.repeat(x[-1], k + 1)
    interior = x[1:-1]
    return np.concatenate([left, interior, right]).astype(float)

def n_basis_from_knots(knots: np.ndarray, degree: int = 3) -> int:
    k = int(degree)
    return len(knots) - k - 1

import numpy as np

# -----------------------------
# Basis + integrated basis matrix
# -----------------------------
def bspline_basis_list(knots: np.ndarray, degree: int = 3):
    """
    Return list of BSpline basis functions N_k(t) for given knot vector.
    knots must cover the domain of all cashflow times.
    """
    knots = np.asarray(knots, float)
    p = len(knots) - degree - 1  # number of basis functions
    basis = []
    eye = np.eye(p)
    for k in range(p):
        basis.append(BSpline(knots, eye[k], degree, extrapolate=False))
    return basis

def integrated_basis_matrix(times: np.ndarray, basis: list[BSpline], t0: float = 0.0):
    """
    A[m,k] = ∫_{t0}^{times[m]} N_k(s) ds using exact antiderivative.
    """
    times = np.asarray(times, float)
    A = np.empty((len(times), len(basis)), float)
    for k, Nk in enumerate(basis):
        Ik = Nk.antiderivative()
        A[:, k] = Ik(times) - Ik(t0)
    if np.isnan(A).any():
        raise ValueError("Some times fall outside spline domain. Check knots / clamping.")
    return A

# -----------------------------
# Roughness penalty matrix K = ∫ N''_k N''_l
# -----------------------------
def roughness_matrix(knots: np.ndarray, degree: int = 3, grid_size: int = 5000):
    """
    K_{kl} = ∫ N''_k(t) N''_l(t) dt approximated with trapezoid on a dense grid.
    """
    knots = np.asarray(knots, float)
    basis = bspline_basis_list(knots, degree)
    d2 = [b.derivative(2) for b in basis]

    t_min = knots[degree]
    t_max = knots[-degree - 1]
    t = np.linspace(t_min, t_max, grid_size)
    dt = np.diff(t)

    B2 = np.column_stack([b(t) for b in d2])  # (G, p)

    # trapezoid weights
    w = np.empty_like(t)
    w[0] = 0.5 * dt[0]
    w[-1] = 0.5 * dt[-1]
    w[1:-1] = 0.5 * (dt[:-1] + dt[1:])

    # K ≈ B2^T diag(w) B2
    K = (B2.T * w) @ B2
    return K

# -----------------------------
# Price + Jacobian (w.r.t beta)
# -----------------------------
def price_and_jac(beta, bonds, A_all, idx_slices):
    """
    Returns:
      P_hat: (N,)
      J: (N, p) with J[i,k] = d P_hat_i / d beta_k
    """
    p = len(beta)
    N = len(bonds)
    P_hat = np.zeros(N)
    J = np.zeros((N, p))

    for i, b in enumerate(bonds):
        sl = idx_slices[i]
        A = A_all[sl, :]   # (n_cf_i, p)
        c = b["c"]         # (n_cf_i,)

        z = A @ beta
        d = np.exp(-z)

        P_hat[i] = c @ d

        # dP/dbeta = - sum_j c_j exp(-z_j) A_j
        w = c * d
        J[i, :] = -(w @ A)

    return P_hat, J

# -----------------------------
# Fit for a fixed lambda (Fisher objective via LS on augmented residuals)
# -----------------------------
def sqrt_penalty_from_K(K, eps=1e-12):
    # Symmetrize
    Ksym = 0.5 * (K + K.T)

    # Eigen-decompose (K should be symmetric)
    w, Q = np.linalg.eigh(Ksym)

    # Floor small/negative eigenvalues
    w_clipped = np.clip(w, eps, None)

    # Return L such that L.T @ L ≈ K (with flooring)
    # If you want exactly: beta^T K beta = ||L beta||^2
    L = (np.sqrt(w_clipped)[:, None] * Q.T)   # shape (p, p)
    return L

def fit_fisher_forward_fixed_lambda(
    bonds,
    knots,
    lam: float,
    degree: int = 3,
    A_all=None,
    idx_slices=None,
    K=None,
    beta0=None,
):
    """
    Fits beta for fixed lam and returns fit dict incl. RSS, pricing Jacobian, etc.
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
        K = roughness_matrix(knots, degree=degree)

    # Cholesky for penalty residuals: ||sqrt(lam)*L beta||^2 = lam * beta^T K beta
    jitter = 1e-12
    L = sqrt_penalty_from_K(K, eps=1e-12)

    P_obs = np.array([b["P"] for b in bonds], float)
    N = len(P_obs)

    if beta0 is None:
        beta0 = np.zeros(p)

    def fun(beta):
        P_hat, _ = price_and_jac(beta, bonds, A_all, idx_slices)
        r_price = P_obs - P_hat
        r_pen = np.sqrt(lam) * (L @ beta)
        return np.concatenate([r_price, r_pen])

    def jac(beta):
        _, Jp = price_and_jac(beta, bonds, A_all, idx_slices)
        # r_price = P_obs - P_hat => dr/dbeta = -dP_hat/dbeta = -Jp
        Jr_price = -Jp
        Jr_pen = np.sqrt(lam) * L
        return np.vstack([Jr_price, Jr_pen])

    res = least_squares(fun, beta0, jac=jac, method="trf")

    beta_hat = res.x
    P_hat, J_price = price_and_jac(beta_hat, bonds, A_all, idx_slices)
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

# -----------------------------
# Effective parameters + GCV
# -----------------------------
def effective_params(J_price: np.ndarray, K: np.ndarray, lam: float) -> float:
    """
    ep ≈ tr( J^T J (J^T J + lam K)^{-1} )

    Here J_price is dP_hat/dbeta (N x p). Residual Jacobian differs by sign,
    but J^T J is sign-invariant.
    """
    J = np.asarray(J_price, float)
    K = np.asarray(K, float)
    JTJ = J.T @ J  # (p, p)

    # Solve (JTJ + lam K) X = JTJ  => ep = tr(X)
    A = JTJ + lam * K

    # add tiny jitter for numerical stability
    jitter = 1e-12
    A = A + jitter * np.eye(A.shape[0])

    X = np.linalg.solve(A, JTJ)
    return float(np.trace(X))

def gcv_score(RSS: float, N: int, ep: float, theta: float = 2.0) -> float:
    denom = (N - theta * ep)
    if denom <= 0:
        return np.inf
    return RSS / (denom * denom)

# -----------------------------
# GCV wrapper
# -----------------------------
from scipy.optimize import minimize_scalar

def select_lambda_gcv_fisher_grid_and_linesearch(
    bonds,
    knots,
    degree: int = 3,
    lambda_grid=None,
    theta: float = 2.0,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    warm_start: bool = True,
    # refinement controls
    n_refine_starts: int = 3,         # refine around best K grid points
    refine_halfwidth_decades: float = 0.75,  # +/- decades around grid point
    refine_maxiter: int = 30,
    # numerical safety
    enforce_success: bool = False,     # if True, ignore failed fits
):
    """
    Waggoner-style GCV selection:
      1) Grid search over lambdas (log-spaced)
      2) Local line search (Brent) in log(lambda) around best grid minima

    Returns
    -------
    out : dict with keys:
      - best_lambda (refined)
      - best_fit (refined fit dict)
      - best_gcv
      - grid_table : structured array with (lambda, RSS, ep, gcv, success)
      - refined_table : structured array with (start_lambda, opt_lambda, gcv, success)
    """
    knots = np.asarray(knots, float)

    # Default grid: wide, but you should tune based on units (years vs days).
    if lambda_grid is None:
        lambda_grid = np.logspace(-2, 8, 41)
    lambda_grid = np.asarray(list(lambda_grid), float)

    # Apply bounds if provided
    if lambda_min is not None:
        lambda_grid = lambda_grid[lambda_grid >= float(lambda_min)]
    if lambda_max is not None:
        lambda_grid = lambda_grid[lambda_grid <= float(lambda_max)]
    if lambda_grid.size < 3:
        raise ValueError("Need at least 3 lambdas in grid after applying bounds.")

    # Precompute shared structures (reuse your existing helpers)
    basis = bspline_basis_list(knots, degree)
    p = len(basis)

    times_all = np.concatenate([b["times"] for b in bonds])
    idx_slices = []
    start = 0
    for b in bonds:
        n = len(b["times"])
        idx_slices.append(slice(start, start + n))
        start += n
    A_all = integrated_basis_matrix(times_all, basis, t0=0.0)
    K = roughness_matrix(knots, degree=degree)

    P_obs = np.array([b["P"] for b in bonds], float)
    N = len(P_obs)

    # --------
    # Stage 1: grid search
    # --------
    grid_rows = []
    grid_fits = {}  # store fits for reuse
    beta0 = np.zeros(p)

    best_grid = None

    for lam in lambda_grid:
        fit = fit_fisher_forward_fixed_lambda(
            bonds=bonds,
            knots=knots,
            lam=float(lam),
            degree=degree,
            A_all=A_all,
            idx_slices=idx_slices,
            K=K,
            beta0=beta0,
        )

        # optionally require solver success
        if enforce_success and not fit["success"]:
            ep = np.nan
            gcv = np.inf
        else:
            ep = effective_params(fit["J_price"], K, float(lam))
            gcv = gcv_score(fit["RSS"], N=N, ep=ep, theta=theta)

        grid_rows.append((float(lam), float(fit["RSS"]), float(ep), float(gcv), bool(fit["success"])))
        grid_fits[float(lam)] = fit

        if warm_start and fit["success"]:
            beta0 = fit["beta"]

        if best_grid is None or gcv < best_grid["gcv"]:
            best_grid = {"lambda": float(lam), "gcv": float(gcv), "fit": fit}

    grid_table = np.array(
        grid_rows,
        dtype=[("lambda", "f8"), ("RSS", "f8"), ("ep", "f8"), ("gcv", "f8"), ("success", "?")],
    )

    # Identify candidate starting points for refinement:
    # pick the n_refine_starts smallest gcv values from the grid
    order = np.argsort(grid_table["gcv"])
    order = order[np.isfinite(grid_table["gcv"][order])]
    order = order[: max(1, int(n_refine_starts))]
    start_lams = grid_table["lambda"][order]

    # Helper: evaluate GCV at a lambda (re-fit each time, but can warm-start locally)
    def eval_gcv_at_lambda(lam: float, beta_init=None):
        lam = float(lam)
        fit = fit_fisher_forward_fixed_lambda(
            bonds=bonds,
            knots=knots,
            lam=lam,
            degree=degree,
            A_all=A_all,
            idx_slices=idx_slices,
            K=K,
            beta0=(beta_init if beta_init is not None else np.zeros(p)),
        )
        if enforce_success and not fit["success"]:
            return np.inf, fit
        ep = effective_params(fit["J_price"], K, lam)
        gcv = gcv_score(fit["RSS"], N=N, ep=ep, theta=theta)
        return float(gcv), fit

    # --------
    # Stage 2: local line-search refinements in log-lambda
    # --------
    refined_rows = []
    best = {"lambda": best_grid["lambda"], "gcv": best_grid["gcv"], "fit": best_grid["fit"]}

    for lam0 in start_lams:
        lam0 = float(lam0)

        # bracket around lam0 in log10 space
        lo = 10 ** (np.log10(lam0) - refine_halfwidth_decades)
        hi = 10 ** (np.log10(lam0) + refine_halfwidth_decades)

        # Respect global bounds if provided
        if lambda_min is not None:
            lo = max(lo, float(lambda_min))
        if lambda_max is not None:
            hi = min(hi, float(lambda_max))
        if not (lo < hi):
            continue

        # optional warm-start for this local search: use beta from nearest grid lambda
        beta_init = grid_fits.get(lam0, best_grid["fit"])["beta"]

        # optimize in log space for stability
        def objective(loglam):
            lam = 10.0 ** float(loglam)
            gcv, _ = eval_gcv_at_lambda(lam, beta_init=beta_init)
            return gcv

        res = minimize_scalar(
            objective,
            bounds=(np.log10(lo), np.log10(hi)),
            method="bounded",
            options={"maxiter": int(refine_maxiter)},
        )

        lam_star = 10.0 ** float(res.x)
        gcv_star, fit_star = eval_gcv_at_lambda(lam_star, beta_init=beta_init)

        refined_rows.append((lam0, float(lam_star), float(gcv_star), bool(fit_star["success"])))

        if gcv_star < best["gcv"]:
            best = {"lambda": float(lam_star), "gcv": float(gcv_star), "fit": fit_star}

    refined_table = np.array(
        refined_rows,
        dtype=[("start_lambda", "f8"), ("opt_lambda", "f8"), ("gcv", "f8"), ("success", "?")],
    )

    return {
        "best_lambda": best["lambda"],
        "best_gcv": best["gcv"],
        "best_fit": best["fit"],
        "grid_table": grid_table,
        "refined_table": refined_table,
    }

# -----------------
# Points for plotting
# -----------------

def fisher_curve_points_to_dfs(
    fit: dict,
    t_max: float | None = None,
    n_grid: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce the point data (as DataFrames) that would be used to plot Fisher curves.

    Returns
    -------
    curve_df : pd.DataFrame
            - 't' (maturity in years)
            - 'forward' (rate)
    nodes_df : pd.DataFrame
            - 't' (node maturity in years)
            - 'forward' (forward at node)

    Notes
    -----
    - Spline domain is [knots[degree], knots[-degree-1]].
    - Discount factor is computed by exp(-∫ f(s) ds) using trapezoid integration on the grid.
      (This mirrors your plotting logic; if you want exact integration via antiderivatives, say so.)
    """
    beta = np.asarray(fit["beta"], dtype=float)
    knots = np.asarray(fit["knots"], dtype=float)
    degree = int(fit["degree"])

    # ---- spline domain ----
    t_min = float(knots[degree])
    t_max_spline = float(knots[-degree - 1])
    if t_max is None:
        t_max = t_max_spline
    t_max = float(min(t_max, t_max_spline))

    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")

    t_grid = np.linspace(t_min, t_max, n_grid)

    # ---- build basis ----
    p = beta.size
    eye = np.eye(p)
    basis = [BSpline(knots, eye[k], degree, extrapolate=False) for k in range(p)]

    # ---- forward curve f(t) ----
    f_vals = np.zeros_like(t_grid)
    for k, Nk in enumerate(basis):
        f_vals += beta[k] * Nk(t_grid)

    # ---- discount factor δ(t) = exp(-∫_0^t f(s) ds) on the grid ----
    dt = np.diff(t_grid)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (f_vals[1:] + f_vals[:-1]) * dt)])
    delta_vals = np.exp(-integral)

    # ---- spot z(t) = -log δ(t) / t ----
    spot_vals = np.full_like(t_grid, np.nan)
    spot_vals[1:] = -np.log(delta_vals[1:]) / t_grid[1:]


    # ---- curve_df ----
    data = {"T": t_grid, "forward": f_vals}
    curve_df = pd.DataFrame(data)

    # ---- nodes_df ----
    nodes_df = pd.DataFrame(columns=["T", "forward"])
    # "unique" interior knots for clamped cubic: strip repeated boundaries
    # (keep the same convention you used in the plotting function)
    unique_knots = knots[degree:-degree]
    unique_knots = unique_knots[unique_knots <= t_max]

    f_nodes = np.zeros_like(unique_knots, dtype=float)
    for k, Nk in enumerate(basis):
        f_nodes += beta[k] * Nk(unique_knots)

    nodes_df = pd.DataFrame(
        {
            "t": unique_knots.astype(float),
            "forward": (f_nodes).astype(float),
        }
    )

    return curve_df, nodes_df

# ----------------
# Out-of-sample pricing from pre-trained coefficients
# ----------------

def fisher_predict_prices(beta, knots, bonds_dict, degree=3):
    """
    Compute model prices and residuals from pre-trained Fisher coefficients.

    Parameters
    ----------
    beta : array-like
        Pre-trained spline coefficients.
    knots : array-like
        Knot vector used during training.
    bonds_dict : list of dicts
        Each dict with keys 'P' (dirty price), 'times' (cashflow times), 'c' (cashflows).
    degree : int
        B-spline degree.

    Returns
    -------
    P_hat : np.ndarray
        Model dirty prices.
    resid : np.ndarray
        Residuals P_obs - P_hat.
    """
    beta = np.asarray(beta, float)
    knots = np.asarray(knots, float)
    basis = bspline_basis_list(knots, degree)

    times_all = np.concatenate([b["times"] for b in bonds_dict])
    idx_slices = []
    start = 0
    for b in bonds_dict:
        n = len(b["times"])
        idx_slices.append(slice(start, start + n))
        start += n
    A_all = integrated_basis_matrix(times_all, basis, t0=0.0)

    P_obs = np.array([b["P"] for b in bonds_dict], float)
    P_hat, _ = price_and_jac(beta, bonds_dict, A_all, idx_slices)
    resid = P_obs - P_hat

    return P_hat, resid


# ----------------
# Full wrapper for Fisher
# ----------------


def run_fisher(sample, pre_trained_results=None):
    """DOCSTRING"""
    results = {}

    dates = sample["date"].unique()

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
            best_lam = pre["lambda"]

            P_hat_dirty, resid = fisher_predict_prices(beta_hat, knots, bonds_dict, degree=3)
            bonds["model_price"] = P_hat_dirty - acc_int
            bonds["resid"] = resid

            fit_for_curve = {"beta": beta_hat, "knots": knots, "degree": 3}
            curve_df, nodes_df = fisher_curve_points_to_dfs(fit_for_curve)
        else:
            nodes = fisher_nodes_equal_counts(maturities)
            knots = bspline_knots_from_nodes(nodes, degree=3)

            gcv_out = select_lambda_gcv_fisher_grid_and_linesearch(bonds_dict, knots, degree=3)

            out = gcv_out["best_fit"]
            best_lam = gcv_out["best_lambda"]
            beta_hat = out["beta"]

            bonds["model_price"] = out["P_hat"] - acc_int
            bonds["resid"] = out["resid_price"]

            curve_df, nodes_df = fisher_curve_points_to_dfs(out)

        wmae = error_metrics.wmae(bonds["model_price"], bonds["bid"], bonds["ask"], bonds["duration"])
        hit_rate = error_metrics.hit_rate(bonds["model_price"], bonds["bid"], bonds["ask"])

        results[DATE] = {"beta_hat": beta_hat,
                        "knots": knots,
                        "lambda": best_lam,
                        "bonds": bonds,
                        "curve": curve_df,
                        "nodes": nodes_df,
                        "wmae": wmae,
                        "hit_rate": hit_rate,
                        }
        
    return results