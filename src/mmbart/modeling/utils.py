from typing import Any, Dict, Union

import torch
from torch import nn

class MultimodalEmbedding(nn.Module):
    """Multimodal Embedding Layer"""

    def __init__(
        self, data_config: Dict[str, Any], d_model: int, embedding_norm: bool, do_positional_encodings: bool = False, positional_encodings_type: str = "sin_cos"
    ) -> None:
        """Init.
        Args:
            data_config: Config specifying the modalities and their embedding parameters
            d_model: embedding dimension of the model
            layer_norm: Wether or not to apply layernorm
        """
        super().__init__()

        self.data_config = data_config
        self.d_model = d_model
        self.embedding_norm = embedding_norm
        self.do_positional_encodings = do_positional_encodings

        self.embedding_layer_dict = nn.ModuleDict()
        self.embedding_norm_dict = nn.ModuleDict()

        for modality, modality_config in self.data_config.items():
            self.embedding_layer_dict[modality] = self._create_embedding(
                modality_config
            )
            # Add Layer normalisation
            if self.embedding_norm:
                self.embedding_norm_dict[modality] = nn.LayerNorm(self.d_model)

        if self.do_positional_encodings:
            self.positional_encodings = POS_ENC_REGISTRY[positional_encodings_type](d_model)

    def _create_embedding(
        self, modality_config: Dict[str, Any]
    ) -> Union[nn.Embedding, nn.Linear]:
        """Create Embedding layers. Replace with Registry at some point.
        Args:
            modality_config: Config specifying the parameters for the embedding
        Returns:
              Embedding Layer: Either classical embedding or linear
        """

        embedding_layer: Union[nn.Embedding, nn.Linear]
        if modality_config["type"] in [
            "text",
            "text_spectrum",
            "peak_positional_encoding",
            "run_length_encoding",
            "multiplets",
            "carbon",
            "msms_text",
        ]:
            embedding_layer = nn.Embedding(
                modality_config["vocab_size"],
                self.d_model,  # type: ignore[attr-defined]
                padding_idx=modality_config["pad_token_id"],
            )
        elif modality_config["type"] in ["1D_patches", "msms_number"]:

            if modality_config["type"] == "msms_number":
                patch_size = 2
            else:
                patch_size = modality_config["preprocessor_arguments"]["patch_size"]

            encoding_type = (
                modality_config["preprocessor_arguments"]["encoding_type"]
                if "encoding_type" in modality_config["preprocessor_arguments"]
                else "linear"
            )

            if encoding_type == "linear":
                embedding_layer = nn.Linear(patch_size, self.d_model)  # type: ignore[attr-defined]
            elif encoding_type == "linear_2_layer":
                embedding_layer = nn.Sequential(
                    *[  # type: ignore
                        nn.Linear(patch_size, self.d_model // 2),  # type: ignore[attr-defined]
                        nn.ReLU(),
                        nn.Linear(self.d_model // 2, self.d_model),  # type: ignore[attr-defined]
                    ]
                )
            elif encoding_type == "linear_3_layer":
                embedding_layer = nn.Sequential(
                    *[  # type: ignore
                        nn.Linear(patch_size, self.d_model // 3),  # type: ignore[attr-defined]
                        nn.ReLU(),
                        nn.Linear(self.d_model // 3, 2 * (self.d_model // 3)),  # type: ignore[attr-defined]
                        nn.ReLU(),
                        nn.Linear(2 * (self.d_model // 3), self.d_model),  # type: ignore[attr-defined]
                    ]
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(
                f'Unknown modality type: {modality_config["type"]}'
            )

        return embedding_layer

    def forward(self, token_ids: Dict[str, Any]) -> torch.Tensor:
        """Perform Embedding. Handles embedding, layer norm and optionally XVal.
        Args:
            token_ids: Input modalities
        Return:
            embedding: torch.Tensor
        """
        
        embedding = list()
        last_pos_enc_indices: torch.Tensor = None

        for modality, modality_input in token_ids.items():
            # Embed
            if isinstance(modality_input, dict):  # XVal
                modality_embedding = self.embedding_layer_dict[modality](modality_input["tokenized_input"])  # type: ignore[attr-defined]
                modality_embedding = modality_embedding * modality_input[
                    "numerical_values"
                ].unsqueeze(-1)
            else:
                modality_embedding = self.embedding_layer_dict[modality](modality_input)  # type: ignore[attr-defined]

            # Apply positional encodings
            if self.do_positional_encodings:
                batch_size, seq_len = modality_embedding.shape[:2]

                # Get largest pos_enc indice
                if isinstance(last_pos_enc_indices, torch.Tensor):
                    max_pos_enc_indices = torch.max(last_pos_enc_indices, -1)[0].unsqueeze(1)
                else:
                    max_pos_enc_indices = torch.full((batch_size,1), 0, device=modality_embedding.device)
                    
                # Generate tensor of increasing indices and offset by the highest pos_enc indice for each row
                modality_pos_encoding = torch.arange(seq_len, device=modality_embedding.device).repeat(batch_size, 1)
                modality_pos_encoding = modality_pos_encoding + max_pos_enc_indices

                # Mask padding tokens for pos encoding
                if isinstance(self.embedding_layer_dict[modality], nn.Embedding):
                    modality_tensor = modality_input["tokenized_input"] if isinstance(modality_input, dict) else modality_input
                    padding_mask = (modality_tensor == self.embedding_layer_dict[modality].padding_idx)
                    modality_pos_encoding[padding_mask] = -1

                modality_embedding = modality_embedding + self.positional_encodings(modality_pos_encoding.clone())

                last_pos_enc_indices = modality_pos_encoding

            # Normalise
            if self.embedding_norm:  # type: ignore[attr-defined]
                modality_embedding = self.embedding_norm_dict[modality](  # type: ignore[attr-defined]
                    modality_embedding.float()
                )

            # Stack all embeddings
            embedding.append(modality_embedding)

        if embedding is None:
            raise ValueError("At least one modality needs to be in token_ids.")

        embedding_tensor = torch.cat(embedding, dim=1)

        return embedding_tensor


class DummyLayer(nn.Module):
    """Dummy Layer. Does nothing and returns input."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Does nothing.
        Args:
            inputs: tensor
        Returns:
            tensor
        """
        return inputs


class SincCosPositionalEncoding(nn.Module):
    """Given an input returns a sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_seq_len: int = 1024) -> None:
        """Init
        Args:
            d_model: hidden dimension of the model.
            max_seq_len: maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.register_buffer("pos_enc", self._positional_encs())

    def forward(self, indices: torch.Tensor, *args) -> torch.Tensor: # noqa: ARG002
        """Given an input returns a sinusoidal positional encoding.
        Args:
            indices: Batch_size x Seq_len Contains indices of the positional encodings to sample
        Returns:
            torch.Tensor: sinusoidal positional embedding
        """

        return self.pos_enc[indices]

    def _positional_encs(self) -> torch.Tensor:
        """Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """
        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000**encs
        encs = [  # type: ignore
            (torch.sin(pos / encs), torch.cos(pos / encs))
            for pos in range(self.max_seq_len)
        ]
        encs = [torch.stack(enc, dim=1).flatten()[: self.d_model] for enc in encs]  # type: ignore
        encs = torch.stack(encs)  # type: ignore
        return encs


class LearnedPositionalEncoding(nn.Module):
    """Learned Positional Encodings up to max_seq_len."""

    def __init__(self, d_model: int, max_seq_len: int = 1024) -> None:
        """Init
        Args:
            d_model: hidden dimension of the model.
            max_seq_len: maximum sequence length
        """
        super().__init__()

        self.max_seq_len = max_seq_len
        self.pos_encodings = nn.Embedding(self.max_seq_len, d_model)
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Given an input returns a learned positional encoding.
        Args:
            indices: Batch_size x Seq_len Contains indices of the positional encodings to sample
        Returns:
            torch.Tensor: learned positional embedding
        """

        # Set pos enc of masked tokens to self.max_seq_len - 1
        indices[indices == -1] = self.max_seq_len - 1
        return self.pos_encodings(indices)


POS_ENC_REGISTRY = {'sin_cos': SincCosPositionalEncoding,
                    'learned': LearnedPositionalEncoding}