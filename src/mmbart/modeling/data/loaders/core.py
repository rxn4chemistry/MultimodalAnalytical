"""Loader core object."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from abc import ABC, abstractmethod
from typing import Any

from datasets import Dataset, DatasetDict


class DataLoader(ABC):
    """Abstract DataLoader."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize data preprocessor."""
        self.additional_kwargs = kwargs

    @abstractmethod
    def load_dataset(self) -> Dataset | DatasetDict:
        """Load dataset.

        Returns:
            HF dataset or dataset dictionary.
        """

    def load_splits(
        self, test_size: float | int = 0.1, val_size: float | int = 0.1, **kwargs: Any
    ) -> DatasetDict:
        """Load dataset in splits.

        Args:
            test_size: fraction of test sample or their absolute number. Defaults to 0.1.
            val_size: fraction of validation samples or their absolute number. Defaults to 0.1.
            **kwargs: additional arguments to pass to datasets.Dataset.train_test_split.

        Returns:
            HF dataset dict containing splits.
        """
        # clean-up kwargs
        _ = kwargs.pop("test_size", None)
        # load dataset
        dataset = self.load_dataset()
        # split train, validation, test
        split_dataset = DatasetDict()
        # NOTE: here we assume that if the load dataset returns a DatasetDict, then
        # it's already properly splitted
        if isinstance(dataset, Dataset):
            dataset_test_split = dataset.train_test_split(test_size=test_size, **kwargs)
            dataset_train_val_split = dataset_test_split["train"].train_test_split(
                test_size=val_size, **kwargs
            )
            # val , train
            split_dataset["train"] = dataset_train_val_split["train"]
            split_dataset["val"] = dataset_train_val_split["test"]
            split_dataset["test"] = dataset_test_split["test"]
        else:
            split_dataset = dataset
        return split_dataset
