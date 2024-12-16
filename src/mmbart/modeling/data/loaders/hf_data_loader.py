"""HF dataset generic loader."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from typing import Any, Dict

from datasets import Dataset, DatasetDict, load_dataset

from .core import DataLoader


class HFDataLoader(DataLoader):
    """HF dataset generic loader."""

    def __init__(
        self,
        dataset_path: str,
        dataset_loading_args: Dict[str, Any],
        dataset_processing_function_string: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize HFDataLoader.

        Args:
            dataset_path: dataset path (local or remote).
            dataset_loading_args: dataset loading arguments.
            dataset_processing_function_string: string representing a
                dataset processing function that will be processed via eval
                to be used as a callable taking a dataset as argument. Defaults to empty string,
                a.k.a., no processing.
        """
        super().__init__(**kwargs)
        self.dataset_path = dataset_path
        self.dataset_loading_args = dataset_loading_args
        self.dataset_processing_function = (
            lambda dataset: eval(dataset_processing_function_string)
            if dataset_processing_function_string
            else dataset
        )

    def load_dataset(self) -> Dataset | DatasetDict:
        """Load dataset.

        Returns:
            HF dataset or dataset dictionary.
        """
        return self.dataset_processing_function(
            load_dataset(path=self.dataset_path, **self.dataset_loading_args)
        )
