"""
Compute two sets of date-level correlation metrics:
    1. Between replication methods (MCC, Fisher, Waggonner) and GSW.
    2. Between replication methods themselves

This module loads saved curve outputs from `_data`, converts each method to a
common representation, computes correlations on a shared maturity grid, and
writes summary artifacts to `_data`. 
The correlation heatmaps are saved to `_output`.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curve_conversions as cc
import pull_yield_curve_data
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
CHARTS_DIR = Path(config("OUTPUT_DIR"))
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

METHOD_FILE_MAP = {
    "mcc": "mcc_discount_curve.parquet",
    "fisher": "fisher_forward_curve.parquet",
    "waggoner": "waggoner_forward_curve.parquet",
}

METHOD_LABELS = {
    "mcc": "McCulloch",
    "fisher": "Fisher",
    "waggoner": "Waggoner",
}

CURVE_TYPES = ["discount", "spot_cc", "forward_instant_cc"]
METHOD_PAIR_CURVE_TYPES = ["spot_cc", "forward_instant_cc"]


def _gsw_spot(maturities, params):
    """Compute GSW spot rates for given maturities and parameters."""
    tau1, tau2, beta1, beta2, beta3, beta4 = params
    t = np.asarray(maturities, dtype=float)
    t_safe = np.where(t == 0.0, 1e-6, t)

    tau1_exp = (1 - np.exp(-t_safe / tau1)) / (t_safe / tau1)
    tau2_exp = (1 - np.exp(-t_safe / tau2)) / (t_safe / tau2)

    return (
        beta1
        + beta2 * tau1_exp
        + beta3 * (tau1_exp - np.exp(-t_safe / tau1))
        + beta4 * (tau2_exp - np.exp(-t_safe / tau2))
    )


def _forward_curve_to_common(curve_df):
    """Convert a forward curve DataFrame to the common format with discount, spot_cc, and forward_instant_cc."""
    df = curve_df.rename(columns={"t": "T"}).copy()
    df = df[["T", "forward", "date"]].sort_values("T").reset_index(drop=True)

    t = df["T"].to_numpy(dtype=float)
    f = df["forward"].to_numpy(dtype=float)
    dt = np.diff(t)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dt)])
    discount = np.exp(-integral)

    base = pd.DataFrame({"T": t, "discount": discount})
    dt_fwd = float(np.median(dt)) if len(dt) else 0.25
    converted = cc.add_spot_and_forwards(base, dt=dt_fwd, t_col="T", d_col="discount")
    converted["forward_instant_cc"] = f
    converted["date"] = pd.to_datetime(df["date"].iloc[0]).normalize()
    return converted[["date", "T", "discount", "spot_cc", "forward_instant_cc"]]


def _mcc_curve_to_common(curve_df):
    """Convert an MCC curve DataFrame to the common format with discount, spot_cc, and forward_instant_cc."""
    df = curve_df.rename(columns={"t": "T"}).copy()
    df = df[["T", "discount", "date"]].sort_values("T").reset_index(drop=True)
    t = df["T"].to_numpy(dtype=float)
    dt = np.diff(t)
    dt_fwd = float(np.median(dt)) if len(dt) else 0.25
    converted = cc.add_spot_and_forwards(df[["T", "discount"]], dt=dt_fwd, t_col="T", d_col="discount")
    converted["date"] = pd.to_datetime(df["date"].iloc[0]).normalize()
    return converted[["date", "T", "discount", "spot_cc", "forward_instant_cc"]]


def load_method_curve(method, data_dir=DATA_DIR):
    """Load and convert curve data for a given method to the common format."""
    method = method.lower()
    if method not in METHOD_FILE_MAP:
        raise ValueError(f"Unknown method: {method}")

    path = Path(data_dir) / METHOD_FILE_MAP[method]
    if not path.exists():
        raise FileNotFoundError(f"Missing curve artifact for '{method}': {path}")

    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

    converted = []
    for dt, grp in raw.groupby("date", sort=True):
        grp = grp.copy()
        if method == "mcc":
            out = _mcc_curve_to_common(grp)
        else:
            out = _forward_curve_to_common(grp)
        out["date"] = dt
        converted.append(out)

    return pd.concat(converted, ignore_index=True)


def load_all_method_curves(data_dir=DATA_DIR):
    """Load and convert curve data for all methods to the common format."""
    return {m: load_method_curve(m, data_dir=data_dir) for m in METHOD_FILE_MAP}


def build_gsw_curve_for_date(gsw_date, t_grid, dt_fwd=0.25, data_dir=DATA_DIR):
    """Build the GSW curve for a given date and maturity grid using the loaded parameters."""
    gsw_date = pd.to_datetime(gsw_date).normalize()
    params_raw = pull_yield_curve_data.load_fed_yield_curve_all(data_dir=Path(data_dir))

    if not isinstance(params_raw.index, pd.DatetimeIndex):
        if "date" in params_raw.columns:
            params_df = params_raw.copy()
            params_df["date"] = pd.to_datetime(params_df["date"])
            params_df = params_df.set_index("date")
        else:
            raise ValueError("Fed parameter table must have a DatetimeIndex or a 'date' column.")
    else:
        params_df = params_raw.copy()

    params_df.index = pd.to_datetime(params_df.index).normalize()
    params_df = params_df.sort_index()

    required_cols = ["TAU1", "TAU2", "BETA0", "BETA1", "BETA2", "BETA3"]
    valid_params = params_df.dropna(subset=required_cols)
    idx = valid_params.index[valid_params.index <= gsw_date]
    if len(idx) == 0:
        raise ValueError(f"No valid GSW parameters found on/before {gsw_date.date()}")
    actual_date = idx[-1]
    row = valid_params.loc[actual_date]

    tau1 = float(row["TAU1"])
    tau2 = float(row["TAU2"])

    b0 = float(row["BETA0"])
    b1 = float(row["BETA1"])
    b2 = float(row["BETA2"])
    b3 = float(row["BETA3"])

    scale = 0.01 if max(abs(b0), abs(b1), abs(b2), abs(b3)) > 1.0 else 1.0
    params = (tau1, tau2, b0 * scale, b1 * scale, b2 * scale, b3 * scale)

    spot = _gsw_spot(t_grid, params)
    discount = np.exp(-spot * t_grid)
    gsw_curve = pd.DataFrame({"T": t_grid, "discount": discount})
    converted = cc.add_spot_and_forwards(gsw_curve, dt=dt_fwd, t_col="T", d_col="discount")
    converted["date"] = actual_date

    return actual_date, converted[["date", "T", "discount", "spot_cc", "forward_instant_cc"]]


def _interp(y_df, t_grid, col):
    """Interpolate the specified column of a curve DataFrame onto the given maturity grid."""
    work = y_df[["T", col]].dropna().sort_values("T")
    if work.empty:
        return np.full_like(t_grid, np.nan, dtype=float)
    return np.interp(t_grid, work["T"].to_numpy(dtype=float), work[col].to_numpy(dtype=float))


def _safe_corr(x, y):
    """Compute the correlation between two arrays, handling edge cases for insufficient data or zero variance."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x_m = x[mask]
    y_m = y[mask]
    if np.isclose(np.std(x_m), 0.0) or np.isclose(np.std(y_m), 0.0):
        return np.nan
    return float(np.corrcoef(x_m, y_m)[0, 1])


def compute_correlation_metrics(curves_by_method=None, t_grid=None, data_dir=DATA_DIR):
    """Compute date-level correlation metrics between each method and GSW, as well as summary statistics."""
    if curves_by_method is None:
        curves_by_method = load_all_method_curves(data_dir=data_dir)

    if t_grid is None:
        t_grid = np.linspace(0.25, 30.0, 240)

    method_dates = []
    for method in METHOD_FILE_MAP:
        dts = set(pd.to_datetime(curves_by_method[method]["date"]).dt.normalize())
        method_dates.append(dts)

    common_dates = sorted(set.intersection(*method_dates))
    if not common_dates:
        raise ValueError("No common dates across method curve outputs.")

    rows = []
    dt_fwd = float(np.median(np.diff(t_grid)))

    for dt in common_dates:
        gsw_actual_date, gsw_curve = build_gsw_curve_for_date(dt, t_grid=t_grid, dt_fwd=dt_fwd, data_dir=data_dir)

        for method in METHOD_FILE_MAP:
            method_df = curves_by_method[method]
            m_curve = method_df.loc[method_df["date"] == dt]
            if m_curve.empty:
                continue

            for curve_type in CURVE_TYPES:
                c = _safe_corr(
                    _interp(m_curve, t_grid, curve_type),
                    _interp(gsw_curve, t_grid, curve_type),
                )
                rows.append(
                    {
                        "date": pd.to_datetime(dt),
                        "gsw_date": pd.to_datetime(gsw_actual_date),
                        "method": method,
                        "curve_type": curve_type,
                        "correlation": c,
                    }
                )

    detail = pd.DataFrame(rows).sort_values(["date", "method", "curve_type"]).reset_index(drop=True)
    summary = detail.groupby("date", as_index=False)["correlation"].mean().rename(columns={"correlation": "overall_corr"})

    for method in METHOD_FILE_MAP:
        s = (
            detail.loc[detail["method"] == method]
            .groupby("date", as_index=False)["correlation"]
            .mean()
            .rename(columns={"correlation": f"{method}_corr"})
        )
        summary = summary.merge(s, on="date", how="left")

    summary = summary.sort_values("overall_corr").reset_index(drop=True)

    selected = select_representative_dates(summary)
    return detail, summary, selected


def select_representative_dates(summary_by_date):
    """Select representative dates with low, median, and high overall correlation scores."""
    if summary_by_date.empty:
        raise ValueError("No date-level correlation scores available.")

    scores = summary_by_date.sort_values("overall_corr").reset_index(drop=True)
    low_row = scores.iloc[0]
    high_row = scores.iloc[-1]

    mid_val = scores["overall_corr"].median()
    candidates = scores.copy()
    candidates["abs_dev"] = (candidates["overall_corr"] - mid_val).abs()

    used_dates = {pd.to_datetime(low_row["date"]), pd.to_datetime(high_row["date"])}
    mid_candidates = candidates.loc[~pd.to_datetime(candidates["date"]).isin(used_dates)]
    if mid_candidates.empty:
        mid_candidates = candidates
    median_row = mid_candidates.sort_values(["abs_dev", "overall_corr"]).iloc[0]

    selected = pd.DataFrame(
        [
            {"label": "low_corr", "date": low_row["date"], "overall_corr": low_row["overall_corr"]},
            {"label": "median_corr", "date": median_row["date"], "overall_corr": median_row["overall_corr"]},
            {"label": "high_corr", "date": high_row["date"], "overall_corr": high_row["overall_corr"]},
        ]
    )
    return selected


def compute_and_save_correlation_metrics(curves_by_method=None, data_dir=DATA_DIR):
    """Compute correlation metrics and save the results to CSV files or html plots in the specified data directory."""
    detail, summary, selected = compute_correlation_metrics(curves_by_method=curves_by_method, data_dir=data_dir)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    detail_path = data_dir / "correlation_metrics_detail.csv"
    summary_path = data_dir / "correlation_metrics_by_date.csv"
    selected_path = data_dir / "correlation_selected_dates.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    selected.to_csv(selected_path, index=False)

    pairwise_detail, pairwise_overall = compute_method_pairwise_correlations(
        curves_by_method=curves_by_method,
        data_dir=data_dir,
    )

    pairwise_detail_path = data_dir / "method_pairwise_correlation_detail.csv"
    pairwise_spot_path = data_dir / "method_pairwise_correlation_spot_cc.csv"
    pairwise_forward_path = data_dir / "method_pairwise_correlation_forward_instant_cc.csv"

    pairwise_detail.to_csv(pairwise_detail_path, index=False)
    pairwise_overall.loc[pairwise_overall["curve_type"] == "spot_cc"].to_csv(pairwise_spot_path, index=False)
    pairwise_overall.loc[pairwise_overall["curve_type"] == "forward_instant_cc"].to_csv(pairwise_forward_path, index=False)

    _write_method_heatmap_image(
        pairwise_overall=pairwise_overall,
        curve_type="spot_cc",
        out_path=CHARTS_DIR / "method_corr_heatmap_spot_cc.png",
        title="Method Correlation Heatmap: Spot (CC)",
    )
    _write_method_heatmap_image(
        pairwise_overall=pairwise_overall,
        curve_type="forward_instant_cc",
        out_path=CHARTS_DIR / "method_corr_heatmap_forward_instant_cc.png",
        title="Method Correlation Heatmap: Instantaneous Forward (CC)",
    )

    return detail, summary, selected


def compute_method_pairwise_correlations(curves_by_method=None, t_grid=None, data_dir=DATA_DIR):
    """Compute pairwise correlation metrics between replication methods for each curve type, and summary statistics."""
    if curves_by_method is None:
        curves_by_method = load_all_method_curves(data_dir=data_dir)

    if t_grid is None:
        t_grid = np.linspace(0.25, 30.0, 240)

    methods = list(METHOD_FILE_MAP.keys())
    method_dates = []
    for method in methods:
        dts = set(pd.to_datetime(curves_by_method[method]["date"]).dt.normalize())
        method_dates.append(dts)

    common_dates = sorted(set.intersection(*method_dates))
    if not common_dates:
        raise ValueError("No common dates across method curve outputs.")

    rows = []
    for dt in common_dates:
        date_curves = {
            m: curves_by_method[m].loc[pd.to_datetime(curves_by_method[m]["date"]).dt.normalize() == dt]
            for m in methods
        }
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                for curve_type in METHOD_PAIR_CURVE_TYPES:
                    c = _safe_corr(
                        _interp(date_curves[m1], t_grid, curve_type),
                        _interp(date_curves[m2], t_grid, curve_type),
                    )
                    rows.append(
                        {
                            "date": pd.to_datetime(dt),
                            "method_1": m1,
                            "method_2": m2,
                            "curve_type": curve_type,
                            "correlation": c,
                        }
                    )

    detail = pd.DataFrame(rows).sort_values(["date", "curve_type", "method_1", "method_2"]).reset_index(drop=True)
    overall = (
        detail.groupby(["curve_type", "method_1", "method_2"], as_index=False)["correlation"]
        .mean()
        .rename(columns={"correlation": "overall_correlation"})
    )
    return detail, overall


def _pairwise_matrix(pairwise_overall, curve_type):
    """Convert pairwise overall correlation DataFrame into a symmetric matrix format for the specified curve type."""
    methods = list(METHOD_FILE_MAP.keys())
    mat = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods, dtype=float)

    subset = pairwise_overall.loc[pairwise_overall["curve_type"] == curve_type]
    for _, row in subset.iterrows():
        m1 = row["method_1"]
        m2 = row["method_2"]
        val = row["overall_correlation"]
        mat.loc[m1, m2] = val
        mat.loc[m2, m1] = val
    return mat


def _write_method_heatmap_image(pairwise_overall, curve_type, out_path, title):
    """Write a static heatmap image of method pairwise correlations."""
    mat = _pairwise_matrix(pairwise_overall, curve_type=curve_type)
    label_map = {k: v for k, v in METHOD_LABELS.items()}
    x_labels = [label_map[m] for m in mat.columns.tolist()]
    y_labels = [label_map[m] for m in mat.index.tolist()]
    z = mat.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(z, vmin=-1, vmax=1, cmap="RdYlBu_r")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("Method")

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.3f}", ha="center", va="center", color="black")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the module's main workflow."""
    _, summary, selected = compute_and_save_correlation_metrics(data_dir=DATA_DIR)
    print("Wrote correlation metrics to:", DATA_DIR.resolve())
    print("Selected dates:")
    print(selected.to_string(index=False))
    print("Top/bottom overall correlation dates:")
    print(pd.concat([summary.head(3), summary.tail(3)]).to_string(index=False))


if __name__ == "__main__":
    main()
