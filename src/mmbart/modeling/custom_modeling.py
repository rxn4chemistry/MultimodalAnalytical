from typing import Any, Dict, List, Optional, Callable

import torch
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.modeling_utils import PreTrainedModel

from .custom_bart_modeling import CustomBartConfig
from .utils import MultimodalEmbedding
from transformers.configuration_utils import PretrainedConfig


class CustomConfig(PretrainedConfig):
    """Config for the Custom Model."""

    def __init__(
        self,
        d_model: int = 512,
        encoder_layers: int = 6,
        encoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 2048,
        decoder_layers: int = 6,
        decoder_attention_heads: int = 8,
        decoder_ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation_function: str | Callable = 'gelu',
        gated_linear: bool = False,
        positional_encoding_type: str = 'sin_cos',
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        decoder_start_token_id: int = 2,
        forced_eos_token_id: int = 3,
        guided_generation: bool = False,
        **kwargs
    ) -> None:

        self.d_model = d_model

        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim

        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim

        self.dropout = dropout
        self.activation_function = activation_function
        self.gated_linear = gated_linear
        self.positional_encoding_type = positional_encoding_type

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.forced_eos_token_id = forced_eos_token_id

        self.guided_generation = guided_generation
        
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

    def __init__(self, 
                 d_model: int, 
                 encoder_attention_heads: int,
                 encoder_ffn_dim: int,
                 dropout: float,
                 activation_function: str,
                 gated_linear: bool = False):
        
        super().__init__(d_model=d_model, 
                       nhead=encoder_attention_heads, 
                       dim_feedforward=encoder_ffn_dim, 
                       dropout=dropout,
                       activation=activation_function,
                       batch_first=True,
                       norm_first=True)

        self.gated_linear = gated_linear
        if gated_linear:
            self.gate = nn.Linear(d_model, encoder_ffn_dim, bias=True)
            self._ff_block = self._ff_block_gated

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

    def __init__(self, 
                 d_model: int, 
                 decoder_attention_heads: int,
                 decoder_ffn_dim: int,
                 dropout: float,
                 activation_function: str,
                 gated_linear: bool = False) -> None:
        
        super().__init__(d_model=d_model, 
                       nhead=decoder_attention_heads, 
                       dim_feedforward=decoder_ffn_dim, 
                       dropout=dropout,
                       activation=activation_function,
                       batch_first=True,
                       norm_first=True)

        self.gated_linear = gated_linear
        if gated_linear:
            self.gate = nn.Linear(d_model, decoder_ffn_dim, bias=True)
            self._ff_block = self._ff_block_glu

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

    def __init__(self,
                 encoder_layer: nn.TransformerEncoderLayer,
                 n_layers: int,
                 norm: Optional[nn.LayerNorm] = None
                 ) -> None:
        """
        Args:
            encoder_layer: The type of layer to use in the encoder
            n_layers: How many layers to use
            norm: LayerNorm to be used after the encoder
        """
        super().__init__(encoder_layer, n_layers, norm)
        self.main_input_name = "inputs_embeds"

    def forward(self, # type: ignore
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
        output_dict['attention_mask'] = attention_mask

        return output_dict
    
class CustomDecoder(nn.TransformerDecoder):
    """Custom Transformer to ensure compatability with HuggingFace."""

    def __init__(self,
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


    def forward(self, # type: ignore
                input_ids: torch.LongTensor,
                encoder_hidden_states: torch.FloatTensor,
                encoder_attention_mask: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None, # noqa: ARG002
                cross_attn_head_mask: Optional[torch.Tensor] = None, # noqa: ARG002
                past_key_values: Optional[List[torch.FloatTensor]] = None, # noqa: ARG002
                inputs_embeds: Optional[torch.FloatTensor] = None, # noqa: ARG002
                use_cache: Optional[bool] = None, # noqa: ARG002
                output_attentions: Optional[bool] = None, # noqa: ARG002
                output_hidden_states: Optional[bool] = None, # noqa: ARG002
                return_dict: Optional[bool] = None, # noqa: ARG002
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
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=decoder_embeds.device)

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

    def __init__(self,
                 target_modality,
                 target_tokenizer,
                 config: CustomConfig,
                 multimodal_embedding_layer: MultimodalEmbedding
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
        enc_layer = CustomEncoderLayer(config.d_model,
                                       config.encoder_attention_heads,
                                       config.encoder_ffn_dim,
                                       config.dropout,
                                       config.activation_function,
                                       config.gated_linear)
        self.encoder = CustomEncoder(enc_layer, config.encoder_layers, norm=enc_norm)

        # Decoder
        dec_norm = nn.LayerNorm(config.d_model)
        dec_layer = CustomDecoderLayer(config.d_model,
                                       config.decoder_attention_heads,
                                       config.decoder_ffn_dim,
                                       config.dropout,
                                       config.activation_function,
                                       config.gated_linear)
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
        encoder_outputs: Optional[Dict[str, Any]] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False, # noqa: ARG002
        return_dict: Optional[bool] = False # noqa: ARG002
    ) -> Seq2SeqLMOutput:
        """ Forward. Converts input from HF into a form compatible w. torch Transformer.
        Args:
            inputs_embeds: Encoder input embeddings
            attention_mask: Encoder attention mask
            encoder_outputs: Encoder output containing encoder last hidden state and encoder attention mask
            decoder_input_ids: Input IDs for the decoder
            decoder_attention_mask: Decoder Attention mask
            labels: Labels for computing loss
            
            All others are to ensure compatability with HF but are not used.
        
        Returns:
            Seq2SeqLMOutput: Contains loss, logits and encoder/decoder output
        """
        
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

        # Token classification
        logits = self.token_ff(decoder_output['last_hidden_state'])

        if labels is not None:
            labels = labels.to(logits.device) # type: ignore
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(logits.view(-1, self.decoder_vocab_size), labels.view(-1)) # type: ignore
        else:
            masked_lm_loss = None


        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            decoder_hidden_states=decoder_output,
            encoder_hidden_states=encoder_outputs['last_hidden_state'],
        )
