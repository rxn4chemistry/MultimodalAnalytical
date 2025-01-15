"""Tests for tokenizers submodule."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from mmbart.modeling.models.tokenizers.core import TokenizerLoader


def test_load_tokenizer_core() -> None:
    """Test load tokenizer"""
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer_loader = TokenizerLoader(model_name=model_name)
    tokenizer = tokenizer_loader.load_tokenizer()
    print(len(tokenizer))


def test_load_tokenizer_add_tokens() -> None:
    """Test load tokenizer"""
    model_name = "HuggingFaceTB/SmolLM-135M"

    add_tokens = ["[HELLO]", "[BYE]"]
    pad_token = "<PAD>"
    add_special_tokens = {"pad_token": pad_token}

    tokenizer_loader = TokenizerLoader(
        model_name=model_name,
        add_tokens=add_tokens,
        add_special_tokens=add_special_tokens,
    )
    tokenizer = tokenizer_loader.load_tokenizer()
    # assert len(tokenizer) == 50265 + len(add_tokens) + 1
    assert tokenizer.pad_token == pad_token
