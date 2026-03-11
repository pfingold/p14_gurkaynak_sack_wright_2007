"""
Unit tests for data processing and notebook outputs.
Tests include:
    - Existence of data and output directories
    - Existence of key data files (Fed yield curve, CRSP treasury data)
    - Existence of notebook outputs (HTML and notebook copies)
"""
import pytest
from pathlib import Path
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

def test_data_directory_exists():
    """Test that the data directory was created"""
    assert DATA_DIR.exists()
    assert DATA_DIR.is_dir()

def test_output_directory_exists():
    """Test that the output directory was created"""
    assert OUTPUT_DIR.exists()
    assert OUTPUT_DIR.is_dir()

def test_fed_yield_curve_data_exists():
    """Test that the Fed yield curve data file was created"""
    assert (DATA_DIR / "fed_yield_curve.parquet").exists()
    assert (DATA_DIR / "fed_yield_curve_all.parquet").exists()

def test_CRSP_treasury_data_exists():
    """Test that CRSP data files were created"""
    assert (DATA_DIR / "TFZ_consolidated.parquet").exists()
    assert (DATA_DIR / "TFZ_DAILY.parquet").exists()
    assert (DATA_DIR / "TFZ_INFO.parquet").exists()
    assert (DATA_DIR / "TFZ_with_runness.parquet").exists()

def test_new_notebook_outputs_exist():
    """Test that the CRSP treasury tour notebook outputs were created"""
    notebook = "CRSP_treasury_data_tour_ipynb"

    assert (OUTPUT_DIR / f"{notebook}.html").exists(), f"Missing HTML for {notebook}"
    assert (OUTPUT_DIR / f"{notebook}.ipynb").exists(), f"Missing notebook copy for {notebook}"
