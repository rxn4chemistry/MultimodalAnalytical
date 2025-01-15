"""Tests for models submodule."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from math import isclose

import torch

from mmbart.modeling.models.core import ModelLoader


def test_load_model_core_pretrained() -> None:
    """Test load tokenizer"""
    model_name = "gpt2"  # a lightweight model for tests
    model_loader = ModelLoader(model_name=model_name, is_pretrained=True)
    model = model_loader.load_model()

    with torch.no_grad():
        params = torch.concat([p.flatten() for p in model.parameters()])
        mean = params.mean().item()
        std = params.std().item()

    assert model is not None
    assert isclose(mean, -0.0005, abs_tol=1e-4)  # pretrained weights
    assert isclose(std, 0.1350, abs_tol=1e-4)  # pretrained weights


def test_load_model_core_config() -> None:
    """Test load tokenizer"""
    model_name = "gpt2"  # a lightweight model for tests
    # AutoConfig's model_type is not always == model_name
    model_loader = ModelLoader(model_name=model_name, is_pretrained=False, model_type="gpt2")
    model = model_loader.load_model()
    with torch.no_grad():
        params = torch.concat([p.flatten() for p in model.parameters()])
        mean = params.mean().item()
        std = params.std().item()

    assert model is not None
    assert isclose(mean, 0.00015, abs_tol=1e-5)  # model is random
    assert isclose(std, 0.0211, abs_tol=1e-4)  # model is random
