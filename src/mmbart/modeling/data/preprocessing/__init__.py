"""Preprocessing module."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from typing import Dict, Type

from .core import DataPreprocessor
from .lambda_preprocessor import LambdaPreprocessor
from .tokenizer_preprocessor import TokenizerPreprocessor

PREPROCESSOR_REGISTRY: Dict[str, Type[DataPreprocessor]] = {
    TokenizerPreprocessor.__name__: TokenizerPreprocessor,
    LambdaPreprocessor.__name__: LambdaPreprocessor,
}
