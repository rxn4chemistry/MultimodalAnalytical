"""Tests the data analysis cli pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from mmbart.data_analysis.run_data_analysis import run_data_analysis
from mmbart.data_analysis.utils import (
    exclude_unsupported_files,
    list_files,
)


@pytest.mark.skipif(not os.environ.get('GENAI_KEY') or not os.environ.get('GENAI_API'), reason="GENAI_ settings not set")
def test_data_analysis_cli():
    """Tests the cli for the datavis/data analysis interface."""
    data_path = Path("./tests/test_data")
    files_list = list_files(data_path)
    filtered_list = exclude_unsupported_files(files_list, [".csv"])
    assert len(filtered_list) == 2
    
    runner = CliRunner()
    result = runner.invoke(run_data_analysis, ['--output_dir', './tests/test_data/', '--data_folder', data_path])
    
    assert result.exit_code == 0
