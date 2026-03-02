"""
Build Table 1a / 1b (in-sample & out-of-sample) for:
- Metrics: WMAE, Hit Rate
- Methods: McCulloch, Fisher, VRP
and extend the paper tables by adding our replication values side-by-side

Outputs:
- ../_output/table_1a_fit.tex
- ../_output/table_1b_fit.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd
import curve_fitting_utils as cfu
import mcc1975_yield_curve as mcc
#import fisher_yield_curve as fisher
#import vrp_yield_curve as vrp

#Categories for the table rows (buckets of maturities)
BUCKETS = ["0-1", "1-3", "3-5", "5-10", ">10", "All"]
METRICS = ["WMAE", "Hit Rate"]
METHODS = ["McCulloch", "Fisher", "VRP"]
SOURCES = ["Paper", "Replication"]

### Formatting Helpers ###
def fmt_wmae(x):
    "Format WMAE values to 3 decimal places"
    return "" if pd.isna(x) else f"{x:.3f}"

def fmt_hit_rate(x):
    "Format Hit Rate values as percentages with 1 decimal place"
    return "" if pd.isna(x) else f"{100.0 * x:.1f}\\%"

def fmt_table_cells(df):
    """
    Converts numeric cells to strings with appropriate 
    formatting based on the metric type
    """
    out = df.copy()
    for method in METHODS:
        #WMAE Row
        if (method, "WMAE") in out.index:
            out.loc[(method, "WMAE"), :] = [
                fmt_wmae(v) for v in out.loc[(method, "WMAE"), :].tolist()
            ]
        #Hit Rate Row
        if (method, "Hit Rate") in out.index:
            out.loc[(method, "Hit Rate"), :] = [
                fmt_hit_rate(v) for v in out.loc[(method, "Hit Rate"), :].tolist()
            ]
    return out

### Build Table ###
def build_extended_table(paper, replication, title=None):
    """
    Builds the extended table by combining the paper's reported values 
    with our replication results:
    -  Rows: Method, Metric
    -  Columns: Buckets, Source (Paper vs Replication)
    """
    row_index = pd.MultiIndex.from_product(
        [METHODS, METRICS], names=["Method", "Metric"]
    )
    col_index = pd.MultiIndex.from_product(
        [BUCKETS, SOURCES], names=["Bucket", "Source"]
    )
    df = pd.DataFrame(index=row_index, columns=col_index, dtype=float)

    for method in METHODS:
        for metric in METRICS:
            for bucket in BUCKETS:
                df.loc[(method, metric), (bucket, "Paper")] = paper.get((method, metric, bucket), np.nan)
                df.loc[(method, metric), (bucket, "Replication")] = replication.get((method, metric, bucket), np.nan)

    df_display = fmt_table_cells(df)
    if title is not None:
        df_display.attrs["title"] = title
    return df_display

### Export to LaTeX ###
def export_to_latex(df, out_path):
    "Export the DataFrame to a LaTeX file in the output directory"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(
        index=True,
        escape=False,
        multicolumn=True,
        multirow=True,
        na_rep="",
        column_format="ll" + "r" * (len(BUCKETS) * len(SOURCES)),
    )
    out_path.write_text(latex, encoding="utf-8")

### Build Table 1a (In-Sample) ###
def table_1a_values():
    """Returns the values for Table 1a (in-sample) as a dictionary"""
    paper = {
        ("McCulloch", "WMAE", "0-1"): 0.019,
        ("McCulloch", "WMAE", "1-3"): 0.108,
        ("McCulloch", "WMAE", "3-5"): 0.179,
        ("McCulloch", "WMAE", "5-10"): 0.375,
        ("McCulloch", "WMAE", ">10"): 0.376,
        ("McCulloch", "WMAE", "All"): 0.055,
        ("McCulloch", "Hit Rate", "0-1"): 0.473,
        ("McCulloch", "Hit Rate", "1-3"): 0.420,
        ("McCulloch", "Hit Rate", "3-5"): 0.342,
        ("McCulloch", "Hit Rate", "5-10"): 0.224,
        ("McCulloch", "Hit Rate", ">10"): 0.278,
        ("McCulloch", "Hit Rate", "All"): 0.370,

        ("Fisher", "WMAE", "0-1"): 0.063,
        ("Fisher", "WMAE", "1-3"): 0.104,
        ("Fisher", "WMAE", "3-5"): 0.177,
        ("Fisher", "WMAE", "5-10"): 0.324,
        ("Fisher", "WMAE", ">10"): 0.265,
        ("Fisher", "WMAE", "All"): 0.085,
        ("Fisher", "Hit Rate", "0-1"): 0.277,
        ("Fisher", "Hit Rate", "1-3"): 0.458,
        ("Fisher", "Hit Rate", "3-5"): 0.347,
        ("Fisher", "Hit Rate", "5-10"): 0.282,
        ("Fisher", "Hit Rate", ">10"): 0.361,
        ("Fisher", "Hit Rate", "All"): 0.353,

        ("VRP", "WMAE", "0-1"): 0.017,
        ("VRP", "WMAE", "1-3"): 0.093,
        ("VRP", "WMAE", "3-5"): 0.170,
        ("VRP", "WMAE", "5-10"): 0.338,
        ("VRP", "WMAE", ">10"): 0.303,
        ("VRP", "WMAE", "All"): 0.049,
        ("VRP", "Hit Rate", "0-1"): 0.512,
        ("VRP", "Hit Rate", "1-3"): 0.494,
        ("VRP", "Hit Rate", "3-5"): 0.367,
        ("VRP", "Hit Rate", "5-10"): 0.266,
        ("VRP", "Hit Rate", ">10"): 0.304,
        ("VRP", "Hit Rate", "All"): 0.416,
    }
    return paper


### Build Table 1b (Out-of-Sample) ###
def table_1b_values():
    """Returns the values for Table 1b (out-of-sample) as a dictionary"""
    paper = {
        ("McCulloch", "WMAE", "0-1"): 0.020,
        ("McCulloch", "WMAE", "1-3"): 0.109,
        ("McCulloch", "WMAE", "3-5"): 0.202,
        ("McCulloch", "WMAE", "5-10"): 0.405,
        ("McCulloch", "WMAE", ">10"): 0.424,
        ("McCulloch", "WMAE", "All"): 0.056,
        ("McCulloch", "Hit Rate", "0-1"): 0.452,
        ("McCulloch", "Hit Rate", "1-3"): 0.419,
        ("McCulloch", "Hit Rate", "3-5"): 0.313,
        ("McCulloch", "Hit Rate", "5-10"): 0.189,
        ("McCulloch", "Hit Rate", ">10"): 0.240,
        ("McCulloch", "Hit Rate", "All"): 0.351,

        ("Fisher", "WMAE", "0-1"): 0.062,
        ("Fisher", "WMAE", "1-3"): 0.105,
        ("Fisher", "WMAE", "3-5"): 0.198,
        ("Fisher", "WMAE", "5-10"): 0.422,
        ("Fisher", "WMAE", ">10"): 0.697,
        ("Fisher", "WMAE", "All"): 0.091,
        ("Fisher", "Hit Rate", "0-1"): 0.264,
        ("Fisher", "Hit Rate", "1-3"): 0.446,
        ("Fisher", "Hit Rate", "3-5"): 0.314,
        ("Fisher", "Hit Rate", "5-10"): 0.192,
        ("Fisher", "Hit Rate", ">10"): 0.283,
        ("Fisher", "Hit Rate", "All"): 0.316,

        ("VRP", "WMAE", "0-1"): 0.019,
        ("VRP", "WMAE", "1-3"): 0.096,
        ("VRP", "WMAE", "3-5"): 0.199,
        ("VRP", "WMAE", "5-10"): 0.413,
        ("VRP", "WMAE", ">10"): 0.356,
        ("VRP", "WMAE", "All"): 0.052,
        ("VRP", "Hit Rate", "0-1"): 0.491,
        ("VRP", "Hit Rate", "1-3"): 0.485,
        ("VRP", "Hit Rate", "3-5"): 0.321,
        ("VRP", "Hit Rate", "5-10"): 0.200,
        ("VRP", "Hit Rate", ">10"): 0.260,
        ("VRP", "Hit Rate", "All"): 0.386,
    }
    return paper

### Compute Replication Values ###

def metrics_df_to_dict(metrics_df, method):
    """
    Helper function that converts the output of get_full_error_metrics()
    into a dictionary format for build_extended_table()
    """
    out = {}
    for idx, row in metrics_df.iterrows():
        b = str(idx)  # Convert bucket index to string
        out[(method, "WMAE", b)] = row["wmae"]
        out[(method, "Hit Rate", b)] = row["hit_rate"]
    return out

def compute_replication_values(sample, sample_label, pre_trained=None):
    """
    Compute replication bucketed metrics for each method using existing scripts
    """
    replication = {}

    #MCC
    if sample_label == "out" and pre_trained is not None:
        mcc_results = mcc.run_mcculloch(sample, pre_trained_results=pre_trained.get("McCulloch"))
    else:
        mcc_results = mcc.run_mcculloch(sample)

    mcc_metrics = mcc.get_full_error_metrics(mcc_results)
    replication.update(metrics_df_to_dict(mcc_metrics, "McCulloch"))

    #Fisher
    if sample_label == "out" and pre_trained is not None:
        fisher_results = fisher.run_fisher(sample, pre_trained_results=pre_trained.get("Fisher"))
    else:
        fisher_results = fisher.run_fisher(sample)
    
    fisher_metrics = fisher.get_full_error_metrics(fisher_results)
    replication.update(metrics_df_to_dict(fisher_metrics, "Fisher"))

    #VRP
    if sample_label == "out" and pre_trained is not None:
        vrp_results = vrp.run_vrp(sample, pre_trained_results=pre_trained.get("VRP"))
    else:
        vrp_results = vrp.run_vrp(sample)

    vrp_metrics = vrp.get_full_error_metrics(vrp_results)
    replication.update(metrics_df_to_dict(vrp_metrics, "VRP"))

    return replication

### Main Execution ###
def main():
    here = Path(__file__).resolve()
    repo_root = here.parents[1] if here.parent.name == "src" else here.parent
    out_dir = repo_root / "_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    #Load IS & OOS Data
    treasury = cfu.load_tidy_CRSP_treasury(output_dir=out_dir)
    treasury_filtered = cfu.filter_waggoner_treasury_data(treasury)
    in_sample, out_sample = cfu.split_in_out_sample_data(treasury_filtered)

    #Table 1a - In-Sample
    paper_1a = table_1a_values()
    repl_1a = compute_replication_values(in_sample, "in")  #TODO: pass in the actual sample data
    table_1a = build_extended_table(paper_1a, repl_1a, title="Table 1a: In-Sample Fit")
    export_to_latex(table_1a, out_dir / "table_1a_fit.tex")

    #Table 1b - Out-of-Sample
    paper_1b = table_1b_values()
    repl_1b = compute_replication_values(out_sample, "out")  #TODO: pass in the actual sample data
    table_1b = build_extended_table(paper_1b, repl_1b, title="Table 1b: Out-of-Sample Fit")
    export_to_latex(table_1b, out_dir / "table_1b_fit.tex")

if __name__ == "__main__":
    main()