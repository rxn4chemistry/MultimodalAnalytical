"""Preprocessing core object."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class DataPreprocessor(ABC):
    """Abstract DataPreprocessor."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize data preprocessor."""
        self.additional_kwargs = kwargs

    @abstractmethod
    def preprocess(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preprocessing method.

        Args:
            example: example to preprocess.

        Returns:
            preprocessed example.
        """
