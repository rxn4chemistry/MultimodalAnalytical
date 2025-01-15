"""Tests for data collators submodule."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from mmbart.modeling.data.collators.core import DataCollatorLoader
from transformers import AutoTokenizer


def test_load_collator() -> None:
    """Test load collator"""
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.pad_token = tokenizer.eos_token
    hello_ids = tokenizer("hello").input_ids
    inputs = [{"input_ids": hello_ids}]

    collator_loader = DataCollatorLoader(
        collator_name="DataCollatorWithPadding", tokenizer=tokenizer
    )
    collator = collator_loader.load_collator()

    output = collator(inputs)

    assert output.input_ids[0] == hello_ids[0]
