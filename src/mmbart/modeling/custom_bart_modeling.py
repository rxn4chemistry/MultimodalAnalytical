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
        

class PreNormEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, config: BartConfig):

        super().__init__(
            d_model=config.d_model,
            nhead=config.encoder_attention_heads,
            dim_feedforward=config.encoder_ffn_dim,
            dropout=config.dropout,
            activation='gelu'
        )
        self.batch_size = config.batch_size

    def forward(self,
                hidden_states: torch.FloatTensor,
                attention_mask: torch.FloatTensor,
                layer_head_mask: torch.FloatTensor, # noqa: ARG002
                output_attentions: Optional[bool] = False):
        
        if hidden_states.shape[0] == self.batch_size:
            hidden_states = hidden_states.transpose(1, 0)
        
        if len(attention_mask.shape) == 4:
            attention_mask = attention_mask[:, :, 0].squeeze(1)

        # Self attention block
        att = self.norm1(hidden_states)
        att = self.self_attn(
            att, att, att, attn_mask=None, key_padding_mask=attention_mask
        )[0]
        att = hidden_states + self.dropout1(att)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        return (out,)
    

class PreNormDecoderLayer(nn.TransformerDecoderLayer):

    def __init__(self, config: BartConfig):

        super().__init__(
            d_model=config.d_model,
            nhead=config.decoder_attention_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            activation='gelu'
        )
        self.batch_size = config.batch_size

    
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
    ):

        if hidden_states.shape[0] == self.batch_size:
            hidden_states = hidden_states.transpose(1, 0)

        if isinstance(attention_mask, torch.Tensor) and len(attention_mask.shape) == 4:
            attention_mask = attention_mask[:, :, -1, :].squeeze(1)

        if encoder_hidden_states.shape[0] == self.batch_size:
            encoder_hidden_states = encoder_hidden_states.transpose(1, 0)

        if isinstance(attention_mask, torch.Tensor) and len(encoder_attention_mask.shape) == 4:
            encoder_attention_mask = encoder_attention_mask[:, :, 0].squeeze(1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(hidden_states.shape[0], device=hidden_states.device)        
        memory_mask = None


        # Self attention block
        query = self.norm1(hidden_states)
        query = self.self_attn(
            query,
            query,
            query,
            attn_mask=tgt_mask,
            key_padding_mask=attention_mask,
        )[0]
        query = hidden_states + self.dropout1(query)

        # Context attention block
        att = self.norm2(query)
        att = self.multihead_attn(
            att,
            encoder_hidden_states,
            encoder_hidden_states,
            attn_mask=memory_mask,
            key_padding_mask=encoder_attention_mask,
        )[0]
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        return (out, )


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
        self.layers = nn.ModuleList([PreNormEncoderLayer(config) for _ in range(config.encoder_layers)])
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
        self.layers = nn.ModuleList([PreNormDecoderLayer(config) for _ in range(config.encoder_layers)])
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
    


from transformers.modeling_utils import PreTrainedModel
from .utils import MultimodalEmbedding
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Dict, Any
from transformers.generation import GenerationMixin


class CustomEncoder(nn.TransformerEncoder):

    def __init__(self,
                 decoder_layer, 
                 n_layers, 
                 norm: Optional[nn.LayerNorm] = None,):
        super().__init__(decoder_layer, n_layers, norm)
        self.main_input_name = "inputs_embeds"

    def forward(self, 
                inputs_embeds: torch.FloatTensor,
                attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        
        if isinstance(attention_mask, torch.Tensor):
            src_key_padding_mask = attention_mask.clone().float()
            src_key_padding_mask[src_key_padding_mask == 0] = float("-Inf")
            src_key_padding_mask[src_key_padding_mask == 1] = 0
        else:
            src_key_padding_mask = torch.full((inputs_embeds.shape[:1]), 0)
        
        output = super().forward(inputs_embeds, src_key_padding_mask=src_key_padding_mask)

        output_dict = BaseModelOutput(last_hidden_state=output)
        output_dict['attention_mask'] = attention_mask

        return output_dict
    
class CustomDecoder(nn.TransformerDecoder):

    def __init__(self,
                 decoder_layer, 
                 n_layers, 
                 norm: Optional[nn.LayerNorm] = None,
                 target_modality: Optional[str] = None,
                 embedding_layer: Optional[MultimodalEmbedding] = None
                 ):
        
        super().__init__(decoder_layer, n_layers, norm)
        
        self.embedding = embedding_layer
        self.target_modality = target_modality


    def forward(self, 
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
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        
        if isinstance(encoder_attention_mask, torch.Tensor):
            encoder_attention_mask = encoder_attention_mask.float()
            encoder_attention_mask[encoder_attention_mask == 0] = float("-Inf")
            encoder_attention_mask[encoder_attention_mask == 1] = 0
        else:
            encoder_attention_mask = torch.full((encoder_hidden_states.shape[:2]), 0.0, device=input_ids.device)
        
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.float()
            attention_mask[attention_mask == 0] = float("-Inf")
            attention_mask[attention_mask == 1] = 0
        else:
            attention_mask = torch.full(input_ids.shape, 0.0, device=input_ids.device)
        
        decoder_embeds = self.embedding({self.target_modality: input_ids}, pos_encoding=True)
        seq_len = input_ids.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=decoder_embeds.device)

        decoder_output = super().forward(
            decoder_embeds,
            encoder_hidden_states,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=attention_mask,
            memory_key_padding_mask=encoder_attention_mask.clone(),
        )

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=decoder_output)



class CustomModel(PreTrainedModel, GenerationMixin):

    def __init__(self,
                 target_modality,
                 target_tokenizer,
                 config: CustomBartConfig,
                 multimodal_embedding_layer: MultimodalEmbedding
                 ):
        
        super().__init__(config)

        self.target_modality = target_modality
        self.decoder_vocab_size = target_tokenizer.vocab_size

        # Embedding
        self.embedding = multimodal_embedding_layer
        
        # Encoder
        enc_norm = nn.LayerNorm(config.d_model)
        enc_layer = nn.TransformerEncoderLayer(config.d_model, 
                                               config.encoder_attention_heads, 
                                               config.encoder_ffn_dim, 
                                               config.dropout, 
                                               config.activation_function, 
                                               norm_first=True,
                                               batch_first=True)
        self.encoder = CustomEncoder(enc_layer, config.encoder_layers, norm=enc_norm)

        # Decoder
        dec_norm = nn.LayerNorm(config.d_model)
        dec_layer = nn.TransformerDecoderLayer(config.d_model, 
                                               config.decoder_attention_heads, 
                                               config.decoder_ffn_dim, 
                                               config.dropout, 
                                               config.activation_function, 
                                               norm_first=True,
                                               batch_first=True)
        self.decoder = CustomDecoder(dec_layer, 
                                     config.decoder_layers, 
                                     norm=dec_norm, 
                                     target_modality=self.target_modality,
                                     embedding_layer=self.embedding)

        # LM Head
        self.token_ff = nn.Linear(config.d_model, self.decoder_vocab_size)

    

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Dict[str, Any] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        return_dict: Optional[bool] = False
    ) -> Seq2SeqLMOutput:
        
        # Encode
        if not isinstance(encoder_outputs, dict):
            encoder_outputs = self.encoder(inputs_embeds, attention_mask=attention_mask, )

        # Decode
        decoder_output = self.decoder(
            input_ids = decoder_input_ids,
            attention_mask = decoder_attention_mask,
            encoder_hidden_states = encoder_outputs['last_hidden_state'],
            encoder_attention_mask = attention_mask,
        )

        # Feedforward
        logits = self.token_ff(decoder_output['last_hidden_state'])

        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.decoder_vocab_size), labels.view(-1))
        else:
            masked_lm_loss = None


        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            decoder_hidden_states=decoder_output,
            encoder_hidden_states=encoder_outputs['last_hidden_state'],
        )




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
