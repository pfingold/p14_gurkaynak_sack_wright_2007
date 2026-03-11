"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import json
import shutil
import uuid
from os import environ, getcwd, path
from pathlib import Path

from colorama import Fore, Style, init

## Custom reporter: Print PyDoit Text in Green
# This is helpful because some tasks write to sterr and pollute the output in
# the console. I don't want to mute this output, because this can sometimes
# cause issues when, for example, LaTeX hangs on an error and requires
# presses on the keyboard before continuing. However, I want to be able
# to easily see the task lines printed by PyDoit. I want them to stand out
# from among all the other lines printed to the console.
from doit.reporter import ConsoleReporter

from settings import config

try:
    in_slurm = environ["SLURM_JOB_ID"] is not None
except:
    in_slurm = False


class GreenReporter(ConsoleReporter):
    def write(self, stuff, **kwargs):
        doit_mark = stuff.split(" ")[0].ljust(2)
        task = " ".join(stuff.split(" ")[1:]).strip() + "\n"
        output = (
            Fore.GREEN
            + doit_mark
            + f" {path.basename(getcwd())}: "
            + task
            + Style.RESET_ALL
        )
        self.outstream.write(output)


if not in_slurm:
    DOIT_CONFIG = {
        "reporter": GreenReporter,
        # other config here...
        # "cleanforget": True, # Doit will forget about tasks that have been cleaned.
        "backend": "sqlite3",
        "dep_file": "./.doit-db.sqlite",
    }
else:
    DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}
init(autoreset=True)


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")
USER = config("USER")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def py_percent_to_notebook(pyfile_path, notebook_path):
    """Convert a percent-format Jupytext .py file into a .ipynb file."""
    pyfile_path = Path(pyfile_path)
    notebook_path = Path(notebook_path)

    lines = pyfile_path.read_text(encoding="utf-8").splitlines()

    cells = []
    current_type = None
    current_lines = []

    def flush_cell():
        nonlocal current_type, current_lines
        if current_type is None:
            return

        if current_type == "markdown":
            md_lines = []
            for line in current_lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.startswith("#"):
                    md_lines.append(line[1:])
                else:
                    md_lines.append(line)
            source = "\n".join(md_lines).rstrip("\n")
            cells.append(
                {
                    "cell_type": "markdown",
                    "id": uuid.uuid4().hex[:8],
                    "metadata": {},
                    "source": source,
                }
            )
        else:
            source = "\n".join(current_lines).rstrip("\n")
            cells.append(
                {
                    "cell_type": "code",
                    "id": uuid.uuid4().hex[:8],
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )

        current_type = None
        current_lines = []

    for line in lines:
        if line.startswith("# %%"):
            flush_cell()
            current_type = "markdown" if "[markdown]" in line else "code"
            current_lines = []
            continue

        if current_type is not None:
            current_lines.append(line)

    flush_cell()

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["ipython ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


def task_test_unit():
    """Run unit tests in src/test_*.py."""
    return {
        "actions": [
            "pytest -q src/test_*.py",
        ],
        "file_dep": [
            "./src/test_curve_fitting_utils.py",
            "./src/test_dodo.py",
            "./src/test_error_metrics.py",
            "./src/test_replication_tables.py",
            "./src/test_correlation_metrics.py",
        ],
        "clean": [],
    }

#Separately Load Data Sources, Can Be Updated Individually
def task_pull_CRSP_treasury():
    """Pull CRSP Treasury data from WRDS"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_CRSP_treasury.py",
        ],
        "targets": [
            DATA_DIR / "TFZ_DAILY.parquet",
            DATA_DIR / "TFZ_INFO.parquet",
            DATA_DIR / "TFZ_consolidated.parquet",
            DATA_DIR / "TFZ_with_runness.parquet",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_CRSP_treasury.py"],
        "clean": [],
    }

def task_pull_fed_yield_curve():
    """Pull Federal Reserve Yield Curve Data, GSW Approach"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/pull_yield_curve_data.py",
        ],
        "targets": [
            DATA_DIR / "fed_yield_curve_all.parquet",
            DATA_DIR / "fed_yield_curve.parquet",
        ],
        "file_dep": ["./src/settings.py", "./src/pull_yield_curve_data.py"],
        "clean": [],
    }

def task_create_diagnostic_charts():
    """Create Diagnostic Charts for Data Sources (Homework 3)"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/generate_chart.py",
        ],
        "targets": [
            OUTPUT_DIR / "crsp_treasury_sample_plot.html",
            OUTPUT_DIR / "fed_yield_curve_sample_plot.html",
        ],
        "file_dep": ["./src/settings.py", 
                     "./src/generate_chart.py",
                     DATA_DIR / "TFZ_consolidated.parquet",
                     DATA_DIR / "fed_yield_curve_all.parquet"],
        "task_dep": ["pull_CRSP_treasury", "pull_fed_yield_curve"],
        "clean": True,
    }

#Tidy the CRSP Treasury Data
def task_tidy_CRSP_treasury():
    """Tidy CRSP Treasury data for curve estimation"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/tidy_CRSP_treasury.py",
        ],
        "targets": [
            DATA_DIR / "tidy_CRSP_treasury.parquet",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/tidy_CRSP_treasury.py",
            DATA_DIR / "TFZ_with_runness.parquet",
        ],
        "task_dep": ["pull_CRSP_treasury"],
        "clean": True,
    }

# Begin Replication Tasks
def task_build_mcc_yield_curve():
    """Run McCulloch (MCC) yield curve replication from CRSP Treasury data"""
    return{
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/run_mcc_yield_curve.py",
        ],
        "targets":[
            DATA_DIR / "mcc_discount_curve.parquet",
            DATA_DIR / "mcc_discount_curve_nodes.csv",
            DATA_DIR / "mcc_bond_fits.parquet",
            DATA_DIR / "mcc_fit_quality_by_date.csv",
            DATA_DIR / "mcc_error_metrics.csv",
            DATA_DIR / "mcc_oos_bond_fits.parquet",
            DATA_DIR / "mcc_oos_fit_quality_by_date.csv",
            DATA_DIR / "mcc_oos_error_metrics.csv",
        ],
        "file_dep":[
            "./src/settings.py",
            "./src/run_mcc_yield_curve.py",
            "./src/mcc1975_yield_curve.py",
            "./src/curve_fitting_utils.py",
            "./src/curve_conversions.py",
            "./src/error_metrics.py",
            DATA_DIR / "tidy_CRSP_treasury.parquet",

        ],
        "task_dep": ["tidy_CRSP_treasury"],
        "clean": True,
    }

def task_build_fisher_yield_curve():
    """Run Fisher (1995) forward-curve replication from CRSP Treasury data"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/run_fisher_yield_curve.py",
        ],
        "targets": [
            DATA_DIR / "fisher_forward_curve.parquet",
            DATA_DIR / "fisher_forward_curve_nodes.csv",
            DATA_DIR / "fisher_bond_fits.parquet",
            DATA_DIR / "fisher_fit_quality_by_date.csv",
            DATA_DIR / "fisher_error_metrics.csv",
            DATA_DIR / "fisher_oos_bond_fits.parquet",
            DATA_DIR / "fisher_oos_fit_quality_by_date.csv",
            DATA_DIR / "fisher_oos_error_metrics.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/run_fisher_yield_curve.py",
            "./src/fisher1995_yield_curve.py",
            "./src/curve_fitting_utils.py",
            "./src/error_metrics.py",
            DATA_DIR / "tidy_CRSP_treasury.parquet",
        ],
        "task_dep": ["tidy_CRSP_treasury"],
        "clean": True,
    }


def task_build_waggoner_yield_curve():
    """Run Waggoner (1997) forward-curve replication from CRSP Treasury data"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/run_waggoner_yield_curve.py",
        ],
        "targets": [
            DATA_DIR / "waggoner_forward_curve.parquet",
            DATA_DIR / "waggoner_forward_curve_nodes.csv",
            DATA_DIR / "waggoner_bond_fits.parquet",
            DATA_DIR / "waggoner_fit_quality_by_date.csv",
            DATA_DIR / "waggoner_error_metrics.csv",
            DATA_DIR / "waggoner_oos_bond_fits.parquet",
            DATA_DIR / "waggoner_oos_fit_quality_by_date.csv",
            DATA_DIR / "waggoner_oos_error_metrics.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/run_waggoner_yield_curve.py",
            "./src/waggoner1997_yield_curve.py",
            "./src/fisher1995_yield_curve.py",
            "./src/curve_fitting_utils.py",
            "./src/error_metrics.py",
            DATA_DIR / "tidy_CRSP_treasury.parquet",
        ],
        "task_dep": ["tidy_CRSP_treasury"],
        "clean": True,
    }

# Modern (rolling 20-year) sample tasks
def task_build_mcc_yield_curve_modern():
    """Run McCulloch yield curve on rolling modern (last 20 years) sample"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/run_mcc_yield_curve_modern.py",
        ],
        "targets": [
            DATA_DIR / "modern_mcc_discount_curve.parquet",
            DATA_DIR / "modern_mcc_discount_curve_nodes.csv",
            DATA_DIR / "modern_mcc_bond_fits.parquet",
            DATA_DIR / "modern_mcc_fit_quality_by_date.csv",
            DATA_DIR / "modern_mcc_error_metrics.csv",
            DATA_DIR / "modern_mcc_oos_bond_fits.parquet",
            DATA_DIR / "modern_mcc_oos_fit_quality_by_date.csv",
            DATA_DIR / "modern_mcc_oos_error_metrics.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/run_mcc_yield_curve_modern.py",
            "./src/run_mcc_yield_curve.py",
            "./src/mcc1975_yield_curve.py",
            "./src/curve_fitting_utils.py",
            "./src/curve_conversions.py",
            "./src/error_metrics.py",
            DATA_DIR / "tidy_CRSP_treasury.parquet",
        ],
        "task_dep": ["tidy_CRSP_treasury"],
        "clean": True,
    }


def task_build_fisher_yield_curve_modern():
    """Run Fisher yield curve on rolling modern (last 20 years) sample"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/run_fisher_yield_curve_modern.py",
        ],
        "targets": [
            DATA_DIR / "modern_fisher_forward_curve.parquet",
            DATA_DIR / "modern_fisher_forward_curve_nodes.csv",
            DATA_DIR / "modern_fisher_bond_fits.parquet",
            DATA_DIR / "modern_fisher_fit_quality_by_date.csv",
            DATA_DIR / "modern_fisher_error_metrics.csv",
            DATA_DIR / "modern_fisher_oos_bond_fits.parquet",
            DATA_DIR / "modern_fisher_oos_fit_quality_by_date.csv",
            DATA_DIR / "modern_fisher_oos_error_metrics.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/run_fisher_yield_curve_modern.py",
            "./src/run_fisher_yield_curve.py",
            "./src/fisher1995_yield_curve.py",
            "./src/curve_fitting_utils.py",
            "./src/error_metrics.py",
            DATA_DIR / "tidy_CRSP_treasury.parquet",
        ],
        "task_dep": ["tidy_CRSP_treasury"],
        "clean": True,
    }


def task_build_waggoner_yield_curve_modern():
    """Run Waggoner yield curve on rolling modern (last 20 years) sample"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/run_waggoner_yield_curve_modern.py",
        ],
        "targets": [
            DATA_DIR / "modern_waggoner_forward_curve.parquet",
            DATA_DIR / "modern_waggoner_forward_curve_nodes.csv",
            DATA_DIR / "modern_waggoner_bond_fits.parquet",
            DATA_DIR / "modern_waggoner_fit_quality_by_date.csv",
            DATA_DIR / "modern_waggoner_error_metrics.csv",
            DATA_DIR / "modern_waggoner_oos_bond_fits.parquet",
            DATA_DIR / "modern_waggoner_oos_fit_quality_by_date.csv",
            DATA_DIR / "modern_waggoner_oos_error_metrics.csv",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/run_waggoner_yield_curve_modern.py",
            "./src/run_waggoner_yield_curve.py",
            "./src/waggoner1997_yield_curve.py",
            "./src/fisher1995_yield_curve.py",
            "./src/curve_fitting_utils.py",
            "./src/error_metrics.py",
            DATA_DIR / "tidy_CRSP_treasury.parquet",
        ],
        "task_dep": ["tidy_CRSP_treasury"],
        "clean": True,
    }

### Build Replication Fit Tables (Table 1a and Table 1b, plus 2a/2b with modern data) ###
def task_build_replication_tables():
    """Build replication fit tables (Table 1a and Table 1b) 
    and replication fit tables with modern data (Table 2a and 2b) 
    from saved method outputs"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/replication_tables.py",
        ],
        "targets": [
            OUTPUT_DIR / "table_1a_fit.tex",
            OUTPUT_DIR / "table_1b_fit.tex",
            OUTPUT_DIR / "table_2a_fit.tex",
            OUTPUT_DIR / "table_2b_fit.tex",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/replication_tables.py",
            "./src/curve_fitting_utils.py",
            "./src/error_metrics.py",
            DATA_DIR / "mcc_error_metrics.csv",
            DATA_DIR / "mcc_oos_error_metrics.csv",
            DATA_DIR / "fisher_error_metrics.csv",
            DATA_DIR / "fisher_oos_error_metrics.csv",
            DATA_DIR / "waggoner_error_metrics.csv",
            DATA_DIR / "waggoner_oos_error_metrics.csv",

            DATA_DIR / "modern_mcc_error_metrics.csv",
            DATA_DIR / "modern_mcc_oos_error_metrics.csv",
            DATA_DIR / "modern_fisher_error_metrics.csv",
            DATA_DIR / "modern_fisher_oos_error_metrics.csv",
            DATA_DIR / "modern_waggoner_error_metrics.csv",
            DATA_DIR / "modern_waggoner_oos_error_metrics.csv",
        ],
        "task_dep": [
            "build_mcc_yield_curve",
            "build_fisher_yield_curve",
            "build_waggoner_yield_curve",
        ],
        "clean": True,
    }


def task_build_fisher_lambda_tables():
    """Build Fisher lambda summary tables (decade and regime) for original and modern samples"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/fisher_lambda_exploration.py",
        ],
        "targets": [
            OUTPUT_DIR / "lambda_decade_table_original.tex",
            OUTPUT_DIR / "lambda_regime_table_original.tex",
            OUTPUT_DIR / "lambda_decade_table_modern.tex",
            OUTPUT_DIR / "lambda_regime_table_modern.tex",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/fisher_lambda_exploration.py",
            DATA_DIR / "fisher_fit_quality_by_date.csv",
            DATA_DIR / "modern_fisher_fit_quality_by_date.csv",
        ],
        "task_dep": [
            "build_fisher_yield_curve",
            "build_fisher_yield_curve_modern",
        ],
        "clean": True,
    }


def task_build_correlation_metrics():
    """Compute date-level correlation metrics across replication methods vs GSW"""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/correlation_metrics.py",
        ],
        "targets": [
            DATA_DIR / "correlation_metrics_detail.csv",
            DATA_DIR / "correlation_metrics_by_date.csv",
            DATA_DIR / "correlation_selected_dates.csv",
            DATA_DIR / "method_pairwise_correlation_detail.csv",
            DATA_DIR / "method_pairwise_correlation_spot_cc.csv",
            DATA_DIR / "method_pairwise_correlation_forward_instant_cc.csv",
            OUTPUT_DIR / "method_corr_heatmap_spot_cc.png",
            OUTPUT_DIR / "method_corr_heatmap_forward_instant_cc.png",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/correlation_metrics.py",
            "./src/curve_conversions.py",
            "./src/pull_yield_curve_data.py",
            DATA_DIR / "mcc_discount_curve.parquet",
            DATA_DIR / "fisher_forward_curve.parquet",
            DATA_DIR / "waggoner_forward_curve.parquet",
            DATA_DIR / "fed_yield_curve_all.parquet",
        ],
        "task_dep": [
            "build_mcc_yield_curve",
            "build_fisher_yield_curve",
            "build_waggoner_yield_curve",
            "pull_fed_yield_curve",
        ],
        "clean": True,
    }


def task_build_curve_plots():
    """Create all method curve plots and method-vs-GSW overlays"""
    chart_targets = [
        "methods_vs_gsw_low_corr_discount.png",
        "methods_vs_gsw_low_corr_spot_cc.png",
        "methods_vs_gsw_low_corr_fwd_instant_cc.png",
        "methods_vs_gsw_median_corr_discount.png",
        "methods_vs_gsw_median_corr_spot_cc.png",
        "methods_vs_gsw_median_corr_fwd_instant_cc.png",
        "methods_vs_gsw_high_corr_discount.png",
        "methods_vs_gsw_high_corr_spot_cc.png",
        "methods_vs_gsw_high_corr_fwd_instant_cc.png",
        "mcc_discount_selected_dates.png",
        "mcc_spot_cc_selected_dates.png",
        "mcc_fwd_instant_cc_selected_dates.png",
        "fisher_discount_selected_dates.png",
        "fisher_spot_cc_selected_dates.png",
        "fisher_fwd_instant_cc_selected_dates.png",
        "waggoner_discount_selected_dates.png",
        "waggoner_spot_cc_selected_dates.png",
        "waggoner_fwd_instant_cc_selected_dates.png",
        "curve_plot_manifest.csv",
    ]
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/plot_curves.py",
        ],
        "targets": [OUTPUT_DIR / f for f in chart_targets],
        "file_dep": [
            "./src/settings.py",
            "./src/plot_curves.py",
            "./src/correlation_metrics.py",
            "./src/curve_conversions.py",
            "./src/pull_yield_curve_data.py",
            DATA_DIR / "mcc_discount_curve.parquet",
            DATA_DIR / "fisher_forward_curve.parquet",
            DATA_DIR / "waggoner_forward_curve.parquet",
            DATA_DIR / "fed_yield_curve_all.parquet",
            DATA_DIR / "correlation_selected_dates.csv",
        ],
        "task_dep": [
            "build_correlation_metrics",
        ],
        "clean": True,
    }


def task_build_fisher_figure7_plot():
    """Create the Fisher Figure 7 style plot for February 28, 1977."""
    return {
        "actions": [
            "ipython ./src/settings.py",
            "ipython ./src/plot_fisher_figure7.py",
        ],
        "targets": [
            BASE_DIR / "docs" / "charts" / "fisher_figure7_1977_02_28.html",
        ],
        "file_dep": [
            "./src/settings.py",
            "./src/plot_fisher_figure7.py",
            DATA_DIR / "fisher_forward_curve.parquet",
            DATA_DIR / "fisher_bond_fits.parquet",
            DATA_DIR / "fisher_oos_bond_fits.parquet",
        ],
        "task_dep": [
            "build_fisher_yield_curve",
        ],
        "clean": True,
    }

#Temporarily Disabling Summary Stats Task for Setup Ease
def DISABLE_task_summary_stats_disabled():
    """Generate summary statistics tables"""
    file_dep = ["./src/example_table.py"]
    file_output = [
        "example_table.tex",
        "pandas_to_latex_simple_table1.tex",
    ]
    targets = [OUTPUT_DIR / file for file in file_output]

    return {
        "actions": [
            "ipython ./src/example_table.py",
            "ipython ./src/pandas_to_latex_demo.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }

###############################################################
## Task below is for Notebook Execution and Conversion to HTML
###############################################################

notebook_tasks = {
    #CRSP Treasury Data Tour Notebook
    "CRSP_treasury_data_tour_ipynb": {
        "path": "./src/CRSP_treasury_data_tour_ipynb.py",
        "file_dep": [],
        "targets": [],
    },
    #Analysis Pipeline Tour Notebook
    "analysis_pipeline_tour_ipynb": {
        "path": "./src/analysis_pipeline_tour_ipynb.py",
        "file_dep": [],
        "targets": [],
    },
}


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix(".ipynb")
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                (py_percent_to_notebook, [pyfile_path, notebook_path]),
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on

###############################################################
## Task below is for LaTeX compilation
###############################################################


def task_compile_latex_docs():
    """Compile the LaTeX documents to PDFs"""
    file_dep = [
        "./reports/report_example.tex",
        "./reports/my_article_header.sty",
        "./reports/slides_example.tex",
        "./reports/my_beamer_header.sty",
        "./reports/my_common_header.sty",
        "./reports/report_simple_example.tex",
        "./reports/slides_simple_example.tex",
        "./src/example_plot.py",
        "./src/example_table.py",
    ]
    targets = [
        "./reports/report_example.pdf",
        "./reports/slides_example.pdf",
        "./reports/report_simple_example.pdf",
        "./reports/slides_simple_example.pdf",
    ]

    return {
        "actions": [
            # My custom LaTeX templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_example.tex",  # Clean
            # Simple templates based on small adjustments to Overleaf templates
            "latexmk -xelatex -halt-on-error -cd ./reports/report_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/report_simple_example.tex",  # Clean
            "latexmk -xelatex -halt-on-error -cd ./reports/slides_simple_example.tex",  # Compile
            "latexmk -xelatex -halt-on-error -c -cd ./reports/slides_simple_example.tex",  # Clean
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
    }

sphinx_targets = [
    "./docs/index.html",
]


def task_build_chartbook_site():
    """Compile Sphinx Docs"""
    notebook_scripts = [
        Path(notebook_tasks[notebook]["path"])
        for notebook in notebook_tasks.keys()
    ]
    file_dep = [
        "./README.md",
        "./chartbook.toml",
        #*notebook_scripts,
    ]

    return {
        "actions": [
            "chartbook build -f",
        ],  # Use docs as build destination
        "targets": sphinx_targets,
        "file_dep": file_dep,
        "task_dep": [
            #"run_notebooks",
            "create_diagnostic_charts",
        ],
        "clean": True,
    }
