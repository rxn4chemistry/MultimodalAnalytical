# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# mypy: ignore-errors

"""PyTorch BART model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartDecoder,
    BartDecoderLayer,
    BartEncoder,
    BartEncoderLayer,
    BartForConditionalGeneration,
    BartModel,
)
from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)


class CustomBartConfig(BartConfig):
    """
    Inherits BartConfig. Allows addition of custom fields.
    """

    def __init__(
        self,
        vocab_size: int = 50265,
        max_position_embeddings: int = 1024,
        encoder_layers: int = 12,
        encoder_ffn_dim: int = 4096,
        encoder_attention_heads: int = 16,
        decoder_layers: int = 12,
        decoder_ffn_dim: int = 4096,
        decoder_attention_heads: int = 16,
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        activation_function: str = "gelu",
        d_model: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        init_std: float = 0.02,
        classifier_dropout: float = 0.0,
        scale_embedding: bool = False,
        use_cache: bool = True,
        num_labels: int = 3,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        is_encoder_decoder: bool = True,
        decoder_start_token_id: float = 2,
        forced_eos_token_id: int = 2,
        final_layer_norm: bool = False,
        batch_size: int = 128,
        **kwargs,
    ) -> None:
        """
        Same as HF Bart Config with the addition of the final_layer_norm argument.
        """

        super().__init__(vocab_size,
                         max_position_embeddings,
                         encoder_layers,
                         encoder_ffn_dim,
                         encoder_attention_heads,
                         decoder_layers,
                         decoder_ffn_dim,
                         decoder_attention_heads,
                         encoder_layerdrop,
                         decoder_layerdrop,
                         activation_function,
                         d_model,
                         dropout,
                         attention_dropout,
                         activation_dropout,
                         init_std,
                         classifier_dropout,
                         scale_embedding,
                         use_cache,
                         num_labels,
                         pad_token_id,
                         bos_token_id,
                         eos_token_id,
                         is_encoder_decoder,
                         decoder_start_token_id,
                         forced_eos_token_id,
                         **kwargs,
                         )
        
        self.final_layer_norm = final_layer_norm
        self.batch_size = batch_size


class PreLayerNormBartEncoderLayer(BartEncoderLayer):

    def __init__(self, config: BartConfig):
        super().__init__(config)

        del self.self_attn
        self.self_attn = nn.MultiheadAttention(embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor, # noqa: ARG002
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Changes Normalisation order from Post- to Pre-layer normalisation.

        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        #hidden_states = self.self_attn_layer_norm(hidden_states) # Custom version 1
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states) # Custom version 2

        hidden_states = hidden_states.transpose(1, 0)
        hidden_states, attn_weights = self.self_attn(
            hidden_states, hidden_states, hidden_states, attn_mask=None, key_padding_mask=attention_mask[:, :, 0].squeeze(1)
        )
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        #hidden_states = self.self_attn_layer_norm(hidden_states) # Original code

        #hidden_states = self.final_layer_norm(hidden_states) # Custom version 1
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states) # Custom version 2
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        #hidden_states = self.final_layer_norm(hidden_states) # Original code

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class PreLayerNormBartDecoderLayer(BartDecoderLayer):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Changes Normalisation order from Post- to Pre-layer normalisation.

        Args:
            hidden_states: input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask: attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states:
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask: encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask: mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask: mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value: cached past key and value projection states
            output_attentions:
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        #hidden_states = self.self_attn_layer_norm(hidden_states) # Custom version 1
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states) # Custom version 2
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        #hidden_states = self.self_attn_layer_norm(hidden_states) # Original code

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            #hidden_states = self.encoder_attn_layer_norm(hidden_states) # Custom version 1
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states) # Custom version 2
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            #hidden_states = self.encoder_attn_layer_norm(hidden_states) # Original code

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        #hidden_states = self.final_layer_norm(hidden_states) # Custom version 1
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states) # Custom version 2
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        #hidden_states = self.final_layer_norm(hidden_states) # Original code

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class CustomBartEncoder(BartEncoder):
    """
    Same functionality as standard BartEncoder. Just replaces Encoder layers with PreLayerNormBartEncoderLayers.
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)

        del self.layers
        self.layers = nn.ModuleList([PreLayerNormBartDecoderLayer(config) for _ in range(config.encoder_layers)])
        self.norm = nn.LayerNorm(config.d_model) if config.final_layer_norm else None

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Overwrite forward method of BartEncoder to add layer normalization after layers.
        """
        
        output = super().forward(input_ids,
                                 attention_mask,
                                 head_mask,
                                 inputs_embeds,
                                 output_attentions,
                                 output_hidden_states,
                                 return_dict)
        
        output['last_hidden_state'] = output['last_hidden_state'].transpose(1, 0)
        if self.norm:
            output['last_hidden_state'] = self.norm(output['last_hidden_state'])
        return output


class CustomBartDecoder(BartDecoder):
    """
    Same functionality as standard BartEncoder. Just replaces decoder layers with PreLayerNormBartDecoderLayers.
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)

        del self.layers
        self.layers = nn.ModuleList([PreLayerNormBartDecoderLayer(config) for _ in range(config.encoder_layers)])
        self.norm = nn.LayerNorm(config.d_model) if config.final_layer_norm else None
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Overwrite forward method of BartDecoder to add layer normalization after layers.
        """
        
        output = super().forward(input_ids,
                               attention_mask,
                               encoder_hidden_states,
                               encoder_attention_mask,
                               head_mask,
                               cross_attn_head_mask,
                               past_key_values,
                               inputs_embeds,
                               use_cache,
                               output_attentions,
                               output_hidden_states,
                               return_dict)
        
        output['last_hidden_state'] = output['last_hidden_state'].transpose(1, 0)
        
        if self.norm:
            output['last_hidden_state'] = self.norm(output['last_hidden_state'])
        return output

class CustomBartModel(BartModel):

    def __init__(self, config: BartConfig):
        super().__init__(config)

        del self.encoder
        del self.decoder

        self.encoder = CustomBartEncoder(config, self.shared)
        self.decoder = CustomBartDecoder(config, self.shared)


class CustomBartForConditionalGeneration(BartForConditionalGeneration):

    def __init__(self, config: BartConfig):
        super().__init__(config)

        del self.model
        self.model = CustomBartModel(config)
