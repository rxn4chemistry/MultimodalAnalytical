"""Tests for lambda preprocessing."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from mmbart.modeling.data.preprocessing.lambda_preprocessor import (
    LambdaPreprocessor,
)


def test_preprocessing_lambda() -> None:
    """Test lambda preprocessor."""
    preprocessor = LambdaPreprocessor(
        function_definition_string="dict(**example, c=example['a'] * 2)"
    )
    assert preprocessor.preprocess({"a": 1, "b": 2}) == {"a": 1, "b": 2, "c": 2}
