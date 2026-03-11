"""
Build curve plots for all replication methods (McCulloch, Fisher, Waggoner) and GSW, for selected dates:

Set (1): For three representative dates (low/median/high correlation),
create overlays of all methods plus GSW.

Set (2): For each method, plot discount/spot/forward curves across those
selected dates.

Outputs: plots saved to _output directory

"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import correlation_metrics as cm
from settings import config

CHARTS_DIR = Path(config("OUTPUT_DIR"))
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(config("DATA_DIR"))

METHOD_DISPLAY = {
    "mcc": "McCulloch",
    "fisher": "Fisher",
    "waggoner": "Waggoner",
}

CURVE_SPECS = {
    "discount": {
        "col": "discount",
        "slug": "discount",
        "title": "Discount Curves",
        "yaxis": "Discount Factor",
    },
    "spot": {
        "col": "spot_cc",
        "slug": "spot_cc",
        "title": "Spot Curves (Continuously Compounded)",
        "yaxis": "Spot Rate (CC)",
    },
    "forward": {
        "col": "forward_instant_cc",
        "slug": "fwd_instant_cc",
        "title": "Instantaneous Forward Curves (Continuously Compounded)",
        "yaxis": "Instantaneous Forward Rate (CC)",
    },
}

LABEL_DISPLAY = {
    "low_corr": "Lowest correlation",
    "median_corr": "Median correlation",
    "high_corr": "Highest correlation",
}


def _curve_for_date(curve_df, date):
    """Extract the curve for the specified date from the given DataFrame, matching on the 'date' column."""
    target = pd.to_datetime(date).normalize()
    return curve_df.loc[pd.to_datetime(curve_df["date"]).dt.normalize() == target].copy()


def plot_named_curves(
    named_curves, x_col, y_col, title, y_axis_title, out_image, extra_traces=None
):
    """Plot multiple curves and save as a static PNG image."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, df in named_curves.items():
        ax.plot(df[x_col], df[y_col], label=name, linewidth=2)

    if extra_traces:
        for tr in extra_traces:
            x_vals = tr["df"][tr.get("x_col", x_col)]
            y_vals = tr["df"][tr.get("y_col", y_col)]
            ax.plot(
                x_vals,
                y_vals,
                label=tr["name"],
                linewidth=tr.get("width", 2),
                linestyle="--" if tr.get("dash", "dash") == "dash" else "-",
            )

    ax.set_title(title)
    ax.set_xlabel("Time to Maturity (Years)")
    ax.set_ylabel(y_axis_title)
    ax.grid(alpha=0.25)
    ax.legend(title="Series")

    out_image = Path(out_image)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_image, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_image


def _selected_date_map(selected_df):
    """Convert the selected representative dates DataFrame into a dict mapping label to date."""
    out = {}
    for _, row in selected_df.iterrows():
        out[row["label"]] = pd.to_datetime(row["date"]).normalize()
    return out


def _build_set_one_plots(curves_by_method, selected_dates):
    """For the selected representative dates, plot all method curves plus GSW overlays."""
    generated = []
    t_grid = np.linspace(0.0, 30.0, 30 * 48 + 1)
    dt_fwd = float(np.median(np.diff(t_grid)))

    for label, dt in selected_dates.items():
        actual_gsw_date, gsw_curve = cm.build_gsw_curve_for_date(dt, t_grid=t_grid, dt_fwd=dt_fwd, data_dir=DATA_DIR)

        named_method_curves = {}
        for method in cm.METHOD_FILE_MAP:
            method_curve = _curve_for_date(curves_by_method[method], dt)
            if not method_curve.empty:
                named_method_curves[METHOD_DISPLAY[method]] = method_curve

        if not named_method_curves:
            continue

        label_text = LABEL_DISPLAY.get(label, label)
        dt_txt = pd.to_datetime(dt).date()
        gsw_txt = pd.to_datetime(actual_gsw_date).date()

        for spec in CURVE_SPECS.values():
            out_path = CHARTS_DIR / f"methods_vs_gsw_{label}_{spec['slug']}.png"
            plot_named_curves(
                named_curves=named_method_curves,
                x_col="T",
                y_col=spec["col"],
                title=f"{spec['title']}: Methods vs GSW ({label_text}, date={dt_txt}, GSW={gsw_txt})",
                y_axis_title=spec["yaxis"],
                out_image=out_path,
                extra_traces=[
                    {
                        "name": f"GSW ({gsw_txt})",
                        "df": gsw_curve,
                        "dash": "dash",
                        "width": 3,
                    }
                ],
            )
            generated.append(out_path)

    return generated


def _build_set_two_plots(curves_by_method, selected_dates):
    """For each method, plot curves across the selected dates (without GSW overlay)."""
    generated = []

    for method in cm.METHOD_FILE_MAP:
        method_df = curves_by_method[method]

        for spec in CURVE_SPECS.values():
            named_curves = {}
            for label, dt in selected_dates.items():
                one = _curve_for_date(method_df, dt)
                if one.empty:
                    continue
                lbl = LABEL_DISPLAY.get(label, label)
                named_curves[f"{lbl} ({pd.to_datetime(dt).date()})"] = one

            if not named_curves:
                continue

            out_path = CHARTS_DIR / f"{method}_{spec['slug']}_selected_dates.png"
            plot_named_curves(
                named_curves=named_curves,
                x_col="T",
                y_col=spec["col"],
                title=f"{METHOD_DISPLAY[method]} {spec['title']} (selected dates)",
                y_axis_title=spec["yaxis"],
                out_image=out_path,
            )
            generated.append(out_path)

    return generated


def main():
    curves_by_method = cm.load_all_method_curves(data_dir=DATA_DIR)
    _, _, selected = cm.compute_and_save_correlation_metrics(curves_by_method=curves_by_method, data_dir=DATA_DIR)

    selected_dates = _selected_date_map(selected)

    generated = []
    generated.extend(_build_set_one_plots(curves_by_method, selected_dates))
    generated.extend(_build_set_two_plots(curves_by_method, selected_dates))

    manifest = pd.DataFrame({"file": [str(p) for p in generated]})
    manifest_path = CHARTS_DIR / "curve_plot_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("Wrote curve plots to:", CHARTS_DIR.resolve())
    print("Wrote manifest:", manifest_path.resolve())


if __name__ == "__main__":
    main()
