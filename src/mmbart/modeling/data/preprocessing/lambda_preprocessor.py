"""Lambda preprocessor."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from typing import Any, Dict, cast

# NOTE: to allow function string eval with common libraries
from .core import DataPreprocessor


class LambdaPreprocessor(DataPreprocessor):
    """Lambda preprocessor."""

    def __init__(self, function_definition_string: str, **kwargs: Any) -> None:
        """Initialize the preprocessor.

        Args:
            function_definition_string: function string to be evaluated for
                defining the preprocessing method. It has to take a dictionary
                called example as input and return a dictionary
        """
        super().__init__(**kwargs)
        self.preprocessing_function = lambda example: eval(function_definition_string)  # noqa

    def preprocess(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preprocessing method.

        Args:
            example: example to preprocess.

        Returns:
            preprocessed example.
        """
        return cast(Dict[str, Any], self.preprocessing_function(example=example))
