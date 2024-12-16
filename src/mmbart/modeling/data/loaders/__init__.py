"""Loaders module."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from typing import Dict, Type

from .core import DataLoader
from .hf_data_loader import HFDataLoader

LOADER_REGISTRY: Dict[str, Type[DataLoader]] = {HFDataLoader.__name__: HFDataLoader}
