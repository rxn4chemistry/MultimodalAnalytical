"""Tokenizers core."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from abc import ABC
from typing import Any

from transformers import AutoTokenizer


class TokenizerLoader(ABC):
    """Tokenizer general loader class"""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Init with model name and other params like setting pad_token

        Args:
            model_name: name of the model related to the tokenizer in HF.
        """
        self.additional_kwargs = kwargs
        self.model_name = model_name

    def load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer and apply configs according to kwargs

        Returns:
            AutoTokenizer: _description_
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Apply config
        if "add_tokens" in self.additional_kwargs:
            tokenizer.add_tokens(self.additional_kwargs["add_tokens"])
        if "add_special_tokens" in self.additional_kwargs:
            tokenizer.add_special_tokens(self.additional_kwargs["add_special_tokens"])
        return tokenizer
