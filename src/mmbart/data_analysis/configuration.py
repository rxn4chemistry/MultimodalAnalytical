"""Data analysis automation configuration."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from typing import List

from pydantic_settings import BaseSettings


class DataAnalysisSettings(BaseSettings):
    """Base data analysis settings object.
    """
    supported_file_types: List[str] = ["csv"]
    metadata_fields: List[str] = ["description", "citation", "homepage", "license"]

# instantiating the objects
DATA_ANALYSIS_SETTINGS = DataAnalysisSettings()
