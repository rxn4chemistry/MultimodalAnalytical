"""Model core."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from abc import ABC
from typing import Any, Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class ModelLoader(ABC):
    """Abstract Model Loader"""

    def __init__(
        self,
        model_name: str,
        is_pretrained: bool,
        use_model_config_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize model loader.

        Args:
            model_name: name of the model in HF.
            is_pretrained: whether to use a pretrained version or initialize weights randomly.
            use_model_config_cache: whether to use mode config cache. Defaults to False.
        """
        self.model_name = model_name
        self.is_pretrained = is_pretrained
        self.use_model_config_cache = use_model_config_cache
        self.additional_kwargs = kwargs

    def load_model(self, tokenizer: Optional[AutoTokenizer] = None) -> AutoModelForCausalLM:
        """Method to load a model with two options to initialize weights.

        It supports loading either from pretrained HF checkpoint or random init form model config.
        In case a tokenizer
        """
        if self.is_pretrained:
            # pretrained weights
            model = AutoModelForCausalLM.from_pretrained(self.model_name, **self.additional_kwargs)
        else:
            # random weights
            model_config = AutoConfig.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_config(model_config)

        model.config.use_cache = self.use_model_config_cache
        if tokenizer:
            model.resize_token_embeddings(len(tokenizer))  # add embedding vectors for new tokens
            model.config.pad_token = tokenizer.pad_token  # just in case
            model.config.eos_token = tokenizer.eos_token
            model.config.bos_token = tokenizer.bos_token

        return model
