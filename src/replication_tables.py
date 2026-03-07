"""
Build Table 1a / 1b (in-sample & out-of-sample) for:
- Metrics: WMAE, Hit Rate
- Methods: McCulloch, Fisher, Waggoner
and extend the paper tables by adding our replication values side-by-side

Outputs:
- ../_output/table_1a_fit.tex
- ../_output/table_1b_fit.tex
"""
from pathlib import Path
import numpy as np
import pandas as pd
from settings import config

DATA_DIR = Path(config("DATA_DIR"))

#Categories for the table rows (buckets of maturities)
BUCKETS = ["0-1", "1-3", "3-5", "5-10", ">10", "All"]
METRICS = ["WMAE", "Hit Rate"]
METHODS = ["McCulloch", "Fisher", "Waggoner"]
SOURCES = ["Paper", "Replication"]
METHOD_TO_FILE_STEM = {
    "McCulloch": "mcc",
    "Fisher": "fisher",
    "Waggoner": "waggoner",
}

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
    # Convert to object dtype before inserting formatted strings.
    out = df.copy().astype(object)
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

    def latex_bucket_label(bucket):
        "Helper function to convert bucket labels to LaTeX-friendly format (e.g., '0-1' stays the same, '>10' becomes '\\textgreater{}10')"
        if bucket == ">10":
            return r"\textgreater{}10"
        return bucket

    df_display = fmt_table_cells(df)
    df_display.index = pd.MultiIndex.from_tuples(
        [(f"\\textbf{{{m}}}", f"\\textbf{{{metric}}}") for m, metric in df_display.index],
        names=["\\textbf{Method}", "\\textbf{Metric}"],
    )
    df_display.columns = pd.MultiIndex.from_tuples(
        [
            (f"\\textbf{{{latex_bucket_label(bucket)}}}", f"\\textbf{{{source}}}")
            for bucket, source in df_display.columns
        ],
        names=["\\textbf{Bucket}", "\\textbf{Source}"],
    )
    if title is not None:
        df_display.attrs["title"] = title
    return df_display

### Export to LaTeX ###
def export_to_latex(df, out_path):
    "Export the DataFrame to a LaTeX file in the output directory"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latex_tabular = df.to_latex(
        index=True,
        escape=False,
        multicolumn=True,
        multirow=True,
        na_rep="",
        multicolumn_format="c",
        column_format="ll" + "r" * (len(BUCKETS) * len(SOURCES)),
    )
    latex = (
        "\\begingroup\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\scriptsize\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"{latex_tabular.rstrip()}\n"
        "}\n"
        "\\endgroup\n"
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

        ("Waggoner", "WMAE", "0-1"): 0.017,
        ("Waggoner", "WMAE", "1-3"): 0.093,
        ("Waggoner", "WMAE", "3-5"): 0.170,
        ("Waggoner", "WMAE", "5-10"): 0.338,
        ("Waggoner", "WMAE", ">10"): 0.303,
        ("Waggoner", "WMAE", "All"): 0.049,
        ("Waggoner", "Hit Rate", "0-1"): 0.512,
        ("Waggoner", "Hit Rate", "1-3"): 0.494,
        ("Waggoner", "Hit Rate", "3-5"): 0.367,
        ("Waggoner", "Hit Rate", "5-10"): 0.266,
        ("Waggoner", "Hit Rate", ">10"): 0.304,
        ("Waggoner", "Hit Rate", "All"): 0.416,
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

        ("Waggoner", "WMAE", "0-1"): 0.019,
        ("Waggoner", "WMAE", "1-3"): 0.096,
        ("Waggoner", "WMAE", "3-5"): 0.199,
        ("Waggoner", "WMAE", "5-10"): 0.413,
        ("Waggoner", "WMAE", ">10"): 0.356,
        ("Waggoner", "WMAE", "All"): 0.052,
        ("Waggoner", "Hit Rate", "0-1"): 0.491,
        ("Waggoner", "Hit Rate", "1-3"): 0.485,
        ("Waggoner", "Hit Rate", "3-5"): 0.321,
        ("Waggoner", "Hit Rate", "5-10"): 0.200,
        ("Waggoner", "Hit Rate", ">10"): 0.260,
        ("Waggoner", "Hit Rate", "All"): 0.386,
    }
    return paper

### Load Replication Values ###

def _standardize_bucket(bucket):
    "Helper function to standardize bucket labels (e.g., '0-1', '1-3', etc.) and handle 'All' case."
    b = str(bucket).strip()
    if b.lower() == "all":
        return "All"
    return b


def metrics_df_to_dict(metrics_df, method):
    """
    Helper function that converts the output of get_full_error_metrics()
    into a dictionary format for build_extended_table()
    """
    out = {}
    for _, row in metrics_df.iterrows():
        b = _standardize_bucket(row["bucket"])
        if b not in BUCKETS:
            continue
        out[(method, "WMAE", b)] = row["wmae"]
        out[(method, "Hit Rate", b)] = row["hit_rate"]
    return out

def _read_metrics_file(path):
    "Read the replication metrics file (csv or parquet) and return a DataFrame with standardized columns."
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if "bucket" not in df.columns:
        if "index" in df.columns:
            df = df.rename(columns={"index": "bucket"})
        else:
            raise ValueError(f"'bucket' column missing in {path}")
    required = {"wmae", "hit_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    return df[["bucket", "wmae", "hit_rate"]].copy()


def _resolve_metrics_path(file_stem, sample_label):
    "Given a method's file stem and sample label, determine the path to the corresponding metrics file (csv or parquet) in the data directory."
    suffix = "_error_metrics" if sample_label == "in" else "_oos_error_metrics"
    candidates = [
        DATA_DIR / f"{file_stem}{suffix}.parquet",
        DATA_DIR / f"{file_stem}{suffix}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    looked_in = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find metrics file for '{file_stem}' ({sample_label}-sample). "
        f"Looked in: {looked_in}"
    )


def compute_replication_values(sample_label):
    """
    Load replication bucketed metrics for each method from run_* outputs
    """
    replication = {}
    for method, file_stem in METHOD_TO_FILE_STEM.items():
        metrics_path = _resolve_metrics_path(file_stem=file_stem, sample_label=sample_label)
        metrics_df = _read_metrics_file(metrics_path)
        replication.update(metrics_df_to_dict(metrics_df, method))
    return replication

### Main Execution ###
def main():
    here = Path(__file__).resolve()
    repo_root = here.parents[1] if here.parent.name == "src" else here.parent
    out_dir = repo_root / "_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    #Table 1a - In-Sample
    paper_1a = table_1a_values()
    repl_1a = compute_replication_values("in")
    table_1a = build_extended_table(paper_1a, repl_1a, title="Table 1a: In-Sample Fit")
    export_to_latex(table_1a, out_dir / "table_1a_fit.tex")

    #Table 1b - Out-of-Sample
    paper_1b = table_1b_values()
    repl_1b = compute_replication_values("out")
    table_1b = build_extended_table(paper_1b, repl_1b, title="Table 1b: Out-of-Sample Fit")
    export_to_latex(table_1b, out_dir / "table_1b_fit.tex")

if __name__ == "__main__":
    main()
