"""
Convert a discount-curve representation into:
    - spot/zero rates (continuously compounded, annualized)
    - forward rates (instantaneous, discrete forward over a tenor)

Compiles a set of functions to perform these conversions, and a 
wrapper to add all the columns to the curve DataFrame at once.
"""

import pandas as pd
import numpy as np

### Helper Functions ###
def _as_arrays(curve, t_col, d_col):
    """Convert the curve DataFrame into arrays of times and discount factors."""
    if t_col not in curve.columns or d_col not in curve.columns:
        raise KeyError(f"Curve DataFrame must contain columns '{t_col}' and '{d_col}'.")

    T = np.asarray(curve[t_col], dtype=float)
    D = np.asarray(curve[d_col], dtype=float)

    #sanity checks
    if np.any(np.isnan(T)) or np.any(np.isnan(D)):
        raise ValueError("Input curve contains NaN values.")
    if np.any(T < 0) or np.any(D <= 0):
        raise ValueError("Times must be non-negative and discount factors must be positive.")
    
    idx = np.argsort(T)
    return T[idx], D[idx]

def interp_discount(curve, t, t_col, d_col):
    """Linear interpolation of discount factors D(T)"""
    T, D = _as_arrays(curve, t_col, d_col)
    t = np.asarray(t, dtype=float)
    return np.interp(t, T, D)

### Spot / Zero ###
def spot_rates_from_discount_cc(curve, t_col, d_col, out_col):
    """Continuously compounded spot rates from discount factors"""
    T, D = _as_arrays(curve, t_col, d_col)
    r = np.full_like(T, np.nan, dtype=float)
    pos = T > 0
    r[pos] = -np.log(D[pos]) / T[pos]

    #fill T=0 with first positive rate
    if np.any(~pos):
        if np.any(pos):
            r[~pos] = r[pos][0]
        else:
            r[~pos] = 0.0
    
    out = curve.copy()
    out = out.sort_values(t_col).reset_index(drop=True)
    out[out_col] = r
    return out

def spot_rate_from_discount_simple(curve, t_col, d_col, out_col):
    """Simple annualized zero rates from discount factors"""
    T, D = _as_arrays(curve, t_col, d_col)
    r = np.full_like(T, np.nan, dtype=float)
    pos = T > 0
    r[pos] = D[pos] ** (-1.0 / T[pos]) - 1.0

    #fill T=0 with first positive rate
    if np.any(~pos):
        if np.any(pos):
            r[~pos] = r[pos][0]
        else:
            r[~pos] = 0.0
    
    out = curve.copy()
    out = out.sort_values(t_col).reset_index(drop=True)
    out[out_col] = r
    return out

### Forward Rates ###
def forward_rate_instant_cc(curve, t_col, d_col, out_col):
    """Instantaneous forward rates from discount factors,
    numerically computed using np.gradient on ln D(T)"""
    T, D = _as_arrays(curve, t_col, d_col)
    lnD = np.log(D)

    edge_order = 2 if len(T) >= 3 else 1
    dlnD_dt = np.gradient(lnD, T, edge_order=edge_order)
    f = -dlnD_dt

    out = curve.copy()
    out = out.sort_values(t_col).reset_index(drop=True)
    out[out_col] = f
    return out

def forward_rate_discrete_cc(curve, dt, t_col, d_col, out_col):
    """Discrete forward rate over [T, T+dt], continuously compounded, 
    using interporlation to get D(T+dt)"""
    if dt<=0:
        raise ValueError("dt must be positive.")
    T, D = _as_arrays(curve, t_col, d_col)
    T2 = T + dt
    D2 = np.interp(T2, T,D) #linear interpolation of D(T+dt)
    f = (np.log(D) - np.log(D2)) / dt

    out = curve.copy()
    out = out.sort_values(t_col).reset_index(drop=True)
    out[out_col] = f
    return out

### Wrapper ###
def add_spot_and_forwards(curve, dt, t_col, d_col):
    """Wrapper to add spot_cc, spot_simple, forward_instant_cc, 
    forward_discrete_cc to the curve DataFrame"""
    out = curve.copy()
    out = spot_rates_from_discount_cc(out, t_col, d_col, "spot_cc")
    out = spot_rate_from_discount_simple(out, t_col, d_col, "spot_simple")
    out = forward_rate_instant_cc(out, t_col, d_col, "forward_instant_cc")
    out = forward_rate_discrete_cc(out, dt, t_col, d_col, f"forward_{dt:g}y_cc")
    return out
    