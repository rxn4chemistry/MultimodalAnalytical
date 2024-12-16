"""Collator core object."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from abc import ABC
from typing import Any

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DefaultDataCollator,
)
from transformers.data.data_collator import DataCollator

DATA_COLLATOR_REGISTRY = {
    DefaultDataCollator.__name__: DefaultDataCollator,
    DataCollatorWithPadding.__name__: DataCollatorWithPadding,
    DataCollatorForLanguageModeling.__name__: DataCollatorForLanguageModeling,
}


class DataCollatorLoader(ABC):
    """Data Collator general loader."""

    def __init__(
        self,
        collator_name: str,
        tokenizer: AutoTokenizer,
        **kwargs: Any,
    ) -> None:
        """Init data collator.

        Args:
            collator_name: string of collator name for registry.
            tokenizer: the tokenizer.
        """
        self.tokenizer = tokenizer
        self.collator_name = collator_name
        self.additional_kwargs = kwargs

    def load_collator(self) -> DataCollator:
        """Load collator and return HF collator object"""

        self.data_collator = DATA_COLLATOR_REGISTRY[self.collator_name](
            self.tokenizer, **self.additional_kwargs
        )
        return self.data_collator
