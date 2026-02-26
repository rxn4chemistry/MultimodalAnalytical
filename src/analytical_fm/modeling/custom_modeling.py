from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel

from .utils import CustomLMOutput, Lambda, MultimodalEmbedding, sid

LOSS_FACTORY: Dict[str, Callable] = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "sid": sid}


class AlignConfig:
    """Config for Encoder Alignment."""

    def __init__(
        self,
        align_network: str,
        hidden_dimension: int,
        conv_channels: int,
        kernel_size: int,
        output_dimension: int,
        loss_lambda: float,
        loss_function: str,
    ):
        self.align_network = align_network
        self.hidden_dimension = hidden_dimension
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.output_dimension = output_dimension
        self.loss_lambda = loss_lambda
        self.loss_function = loss_function


class CustomConfig(PretrainedConfig):
    """Config for the Custom Model."""

    def __init__(
        self,
        d_model: int = 512,
        max_position_embeddings: int = 1024,
        encoder_layers: int = 6,
        encoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 2048,
        decoder_layers: int = 6,
        decoder_attention_heads: int = 8,
        decoder_ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation_function: str | Callable = "gelu",
        post_layer_normalisation: bool = True,
        gated_linear: bool = False,
        positional_encoding_type: str = "sin_cos",
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        decoder_start_token_id: int = 2,
        forced_eos_token_id: int = 3,
        guided_generation: bool = False,
        align_config: Optional[AlignConfig] = None,
        **kwargs,
    ) -> None:

        self.d_model = d_model
        self.max_position_embeddings = max_position_embeddings

        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim

        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim

        self.dropout = dropout
        self.activation_function = activation_function
        self.gated_linear = gated_linear
        self.post_layer_normalisation = post_layer_normalisation
        self.positional_encoding_type = positional_encoding_type

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.forced_eos_token_id = forced_eos_token_id

        self.guided_generation = guided_generation

        if align_config and not isinstance(align_config, AlignConfig):
            align_config = AlignConfig(**align_config)

        self.align_config = align_config

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )


class CustomEncoderLayer(nn.TransformerEncoderLayer):
    """Encoder layer with option for gated feedforward."""

    def __init__(
        self,
        d_model: int,
        encoder_attention_heads: int,
        encoder_ffn_dim: int,
        dropout: float,
        activation_function: str | Callable,
        gated_linear: bool = False,
        post_layer_normalisation: bool = True,
    ):

        super().__init__(
            d_model=d_model,
            nhead=encoder_attention_heads,
            dim_feedforward=encoder_ffn_dim,
            dropout=dropout,
            activation=activation_function,
            batch_first=True,
            norm_first=post_layer_normalisation,
        )

        self.gated_linear = gated_linear
        if gated_linear:
            self.gate = nn.Linear(d_model, encoder_ffn_dim, bias=True)
            self._ff_block = self._ff_block_gated  # type: ignore

    def _ff_block_gated(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gated Feedforward block.
        Args:
            hidden_states
        Returns:
            hidden_state with gated linear unit applied.
        """

        hidden_gelu = self.activation(self.linear1(hidden_states))
        hidden_linear = self.gate(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.dropout2(hidden_states)

        return hidden_states


class CustomDecoderLayer(nn.TransformerDecoderLayer):
    """Decoder layer with option for gated feedforward."""

    def __init__(
        self,
        d_model: int,
        decoder_attention_heads: int,
        decoder_ffn_dim: int,
        dropout: float,
        activation_function: str | Callable,
        gated_linear: bool = False,
        post_layer_normalisation: bool = True,
    ) -> None:

        super().__init__(
            d_model=d_model,
            nhead=decoder_attention_heads,
            dim_feedforward=decoder_ffn_dim,
            dropout=dropout,
            activation=activation_function,
            batch_first=True,
            norm_first=post_layer_normalisation,
        )

        self.gated_linear = gated_linear
        if gated_linear:
            self.gate = nn.Linear(d_model, decoder_ffn_dim, bias=True)
            self._ff_block = self._ff_block_glu  # type: ignore

    def _ff_block_glu(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Gated Feedforward block.
        Args:
            hidden_states
        Returns:
            hidden_state with gated linear unit applied.
        """

        hidden_gelu = self.activation(self.linear1(hidden_states))
        hidden_linear = self.gate(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.dropout2(hidden_states)

        return hidden_states


class CustomEncoder(nn.TransformerEncoder):
    """Custom Transformer to ensure compatability with HuggingFace."""

    def __init__(
        self,
        encoder_layer: nn.TransformerEncoderLayer,
        n_layers: int,
        norm: Optional[nn.LayerNorm] = None,
    ) -> None:
        """
        Args:
            encoder_layer: The type of layer to use in the encoder
            n_layers: How many layers to use
            norm: LayerNorm to be used after the encoder
        """
        super().__init__(encoder_layer, n_layers, norm)
        self.main_input_name = "inputs_embeds"

    def forward( # type: ignore
        self,  
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        """Forward. Converts input from HF into a form compatible w. torch Transformer.
        Args:
            inputs_embeds: Input embeddings
            attention_mask: Encoder attention mask
        Returns:
            BaseModelOutput: Output of the transformer containing the last hidden state and attention mask
        """

        if isinstance(attention_mask, torch.Tensor):
            src_key_padding_mask = ~attention_mask.clone().bool()
        else:
            src_key_padding_mask = torch.full((inputs_embeds.shape[:1]), False)

        output = super().forward(inputs_embeds, src_key_padding_mask=src_key_padding_mask)

        output_dict = BaseModelOutput(last_hidden_state=output)
        output_dict["attention_mask"] = attention_mask

        return output_dict


class CustomDecoder(nn.TransformerDecoder):
    """Custom Transformer to ensure compatability with HuggingFace."""

    def __init__(
        self,
        decoder_layer: nn.TransformerDecoderLayer,
        n_layers: int,
        embedding_layer: MultimodalEmbedding,
        norm: Optional[nn.LayerNorm] = None,
        target_modality: Optional[str] = None,
    ) -> None:
        """
        Args:
            decoder_layer: The type of layer to use in the encoder
            n_layers: How many layers to use
            embedding_layer: Embedding layer to embed decoded tokens
            norm: LayerNorm to be used after the encoder
            target_modality: Name of the target modality
        """

        super().__init__(decoder_layer, n_layers, norm)

        self.embedding = embedding_layer
        self.target_modality = target_modality

    def forward( # type: ignore
        self,  
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # noqa: ARG002
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # noqa: ARG002
        inputs_embeds: Optional[torch.FloatTensor] = None,  # noqa: ARG002
        use_cache: Optional[bool] = None,  # noqa: ARG002
        output_attentions: Optional[bool] = None,  # noqa: ARG002
        output_hidden_states: Optional[bool] = None,  # noqa: ARG002
        return_dict: Optional[bool] = None,  # noqa: ARG002
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Forward. Converts input from HF into a form compatible w. torch Transformer.
        Args:
            input_ids: Input IDs for the decoder
            encoder_hidden_states: Encoder output
            encoder_attention_mask: Encoder attention mask
            attention_mask: Decoder Attention mask

            All others are to ensure compatability with HF but are not used.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions: Contains encoder output
        """

        encoder_attention_mask_bool = ~encoder_attention_mask.bool()

        if isinstance(attention_mask, torch.Tensor):
            attention_mask_bool = ~attention_mask.bool()
        else:
            attention_mask_bool = torch.full(input_ids.shape, False, device=input_ids.device)

        decoder_embeds = self.embedding({self.target_modality: input_ids})
        seq_len = input_ids.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=decoder_embeds.device
        )

        decoder_output = super().forward(
            decoder_embeds,
            encoder_hidden_states,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=attention_mask_bool,
            memory_key_padding_mask=encoder_attention_mask_bool.clone(),
        )

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=decoder_output)


class CustomModel(PreTrainedModel, GenerationMixin):
    """Custom Model makes torch Encoder/Decoder compatible with HF Pretrained model."""

    def __init__(
        self,
        target_modality,
        target_tokenizer,
        config: CustomConfig,
        multimodal_embedding_layer: MultimodalEmbedding,
    ):
        """
        Args:
            target_modality: Name of the target modality
            target_tokenizer: Tokenizer of the target modality
            config: Model config
            multimodal_embedding_layer: Embedding layer
        """

        super().__init__(config)

        self.target_modality = target_modality
        self.decoder_vocab_size = target_tokenizer.vocab_size

        # Embedding
        self.embedding = multimodal_embedding_layer

        # Encoder
        enc_norm = nn.LayerNorm(config.d_model)
        enc_layer = CustomEncoderLayer(
            config.d_model,
            config.encoder_attention_heads,
            config.encoder_ffn_dim,
            config.dropout,
            config.activation_function,
            config.gated_linear,
            config.post_layer_normalisation,
        )
        self.encoder = CustomEncoder(enc_layer, config.encoder_layers, norm=enc_norm)

        # align the encoder mixture representation to the target ir
        self.align_network = None
        if config.align_config:
            if config.align_config.align_network == "convolutional":
                self.align_network = nn.Sequential(
                    nn.Linear(config.d_model, config.align_config.hidden_dimension),
                    nn.ReLU(),
                    nn.Linear(
                        config.align_config.hidden_dimension, config.align_config.hidden_dimension
                    ),
                    Lambda(lambda x: x.unsqueeze(-1)),
                    nn.Conv1d(
                        in_channels=config.align_config.hidden_dimension,
                        out_channels=config.align_config.conv_channels,
                        kernel_size=config.align_config.kernel_size,
                        padding=config.align_config.kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=config.align_config.conv_channels,
                        out_channels=config.align_config.output_dimension,
                        kernel_size=1,
                    ),
                    nn.Sigmoid(),
                    Lambda(lambda x: x.squeeze(-1)),
                )
            elif config.align_config.align_network == "mlp":
                self.align_network = nn.Sequential(
                    nn.Linear(config.d_model, config.align_config.hidden_dimension),
                    nn.ReLU(),
                    nn.Linear(
                        config.align_config.hidden_dimension, config.align_config.output_dimension
                    ),
                    nn.Sigmoid(),
                )

        # Decoder
        dec_norm = nn.LayerNorm(config.d_model)
        dec_layer = CustomDecoderLayer(
            config.d_model,
            config.decoder_attention_heads,
            config.decoder_ffn_dim,
            config.dropout,
            config.activation_function,
            config.gated_linear,
            config.post_layer_normalisation,
        )
        self.decoder = CustomDecoder(
            dec_layer,
            config.decoder_layers,
            norm=dec_norm,
            target_modality=self.target_modality,
            embedding_layer=self.embedding,
        )

        # LM Head
        self.token_ff = nn.Linear(config.d_model, self.decoder_vocab_size)

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Dict[str, Any]] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,  # noqa: ARG002
        return_dict: Optional[bool] = False,  # noqa: ARG002
        encoder_align_target: Optional[torch.Tensor] = None,
    ) -> CustomLMOutput:
        """Forward. Converts input from HF into a form compatible w. torch Transformer.
        Args:
            inputs_embeds: Encoder input embeddings
            attention_mask: Encoder attention mask
            encoder_outputs: Encoder output containing encoder last hidden state and encoder attention mask
            decoder_input_ids: Input IDs for the decoder
            decoder_attention_mask: Decoder Attention mask
            labels: Labels for computing loss

            All others are to ensure compatability with HF but are not used.

        Returns:
            CustomLMOutput: Contains total_loss, transformers loss and alignment loss when aligning the encoder, logits and encoder/decoder output
        """

        generating = isinstance(encoder_outputs, dict)
        # Encode
        if not isinstance(encoder_outputs, dict):
            encoder_outputs = self.encoder(inputs_embeds, attention_mask=attention_mask)

        # average all the non padding tokens hidden_state
        align_loss = 0
        align_loss_lambda = 0
        if self.align_network and not generating:
            try:
                align_loss_function = LOSS_FACTORY[self.config.align_config.loss_function]
            except Exception as e:
                raise ValueError(
                    f"Loss function {self.config.align_config.loss_function} not supported for alignment!{e}"
                )
            if isinstance(attention_mask, torch.Tensor):
                mask = attention_mask.unsqueeze(-1)
            else:
                hidden_state = encoder_outputs["last_hidden_state"]
                mask = torch.ones(hidden_state.size()[:-1], device=hidden_state.device).unsqueeze(
                    -1
                )

            num_unmasked = mask.sum(dim=1)
            align_input = (encoder_outputs["last_hidden_state"] * mask).sum(dim=1) / num_unmasked
            pred = self.align_network(align_input)
            target = encoder_align_target
            align_loss = align_loss_function(pred, target)
            align_loss_lambda = self.config.align_config.loss_lambda

        # Decode
        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            encoder_attention_mask=attention_mask,
        )

        # Token classification
        logits = self.token_ff(decoder_output["last_hidden_state"])

        if labels is not None:
            labels = labels.to(logits.device)  # type: ignore
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.decoder_vocab_size), labels.view(-1))  # type: ignore

            total_loss = masked_lm_loss + align_loss_lambda * align_loss
            loss_dict = {
                "model_only_loss": masked_lm_loss,
                "alignment_loss": align_loss if self.align_network else None,
            }
        else:
            total_loss = None
            loss_dict = None

        return CustomLMOutput(
            loss=total_loss,
            logits=logits,
            decoder_hidden_states=decoder_output,
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            loss_dict=loss_dict,
        )
