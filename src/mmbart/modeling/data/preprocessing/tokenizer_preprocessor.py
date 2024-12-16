"""Tokenizer preprocessor."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from typing import Any, Dict, cast

from transformers import AutoTokenizer

from .core import DataPreprocessor


class TokenizerPreprocessor(DataPreprocessor):
    """Tokenizer preprocessor."""

    def __init__(self, tokenizer: AutoTokenizer, tokenizer_prompt: str, **kwargs: Any) -> None:
        """Initialize the tokenizer preprocessor.

        Args:
            tokenizer: tokenizer to use.
            tokenizer_prompt: function string to be evaluated to format the prompt for the tokenizer.
        """
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.tokenizer_prompt = lambda example: eval(tokenizer_prompt)  # noqa

    def preprocess(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preprocessing method.

        Args:
            example: example to preprocess.

        Returns:
            preprocessed example.
        """
        return cast(Dict[str, Any], self.tokenizer(self.tokenizer_prompt(example=example)))
