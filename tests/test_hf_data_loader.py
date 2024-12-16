"""Tests for HF data loader."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from mmbart.modeling.data.loaders.hf_data_loader import HFDataLoader


def test_hf_data_loader() -> None:
    """Function to test HFDataLoader."""
    loader = HFDataLoader(
        dataset_path="Salesforce/wikitext",
        dataset_loading_args=dict(name="wikitext-103-raw-v1", split="train"),
        dataset_processing_function_string="dataset.take(500)",
    )
    dataset = loader.load_dataset()
    assert len(dataset) == 500
