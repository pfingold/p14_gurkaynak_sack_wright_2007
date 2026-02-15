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

### REPLACE TESTS ONCE NOTEBOOKS ARE CREATED ###

#def test_notebook_outputs_exist():
    #"""Test that all notebook outputs were created"""
    #notebooks = [
        #"01_CRSP_treasury_overview_ipynb",
        #"02_replicate_GSW2005_ipynb"
    #]

    #for notebook in notebooks:
        # Check for HTML output
        #assert (OUTPUT_DIR / f"{notebook}.html").exists(), f"Missing HTML for {notebook}"
        # Check for notebook copy
        #assert (OUTPUT_DIR / f"{notebook}.ipynb").exists(), f"Missing notebook copy for {notebook}"
