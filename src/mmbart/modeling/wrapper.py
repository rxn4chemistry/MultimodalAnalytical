from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from functools import partial
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from rdkit import Chem
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    GenerationConfig,
    T5ForConditionalGeneration,
    PreTrainedModel
)
from transformers.generation.logits_process import LogitsProcessor
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput

from .utils import DummyLayer, MultimodalEmbedding, PositionalEncoding
from .custom_modeling import CustomBartForConditionalGeneration
from mmbart.modeling.decoder import DecodeSampler

OPTIMISER_REGISTRY = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def load_bart_model(
    model_name: str,
    target_tokenizer: AutoTokenizer,
    target_modality: str,
    data_config: Dict[str, Any],
    multimodal_norm: bool,
    **kwargs,
) -> Tuple[BartForConditionalGeneration, MultimodalEmbedding]:
    """Loads a huggingface bart model.
    Args:
        model_name: Modelname e.g. facebook/bart-large
        target_tokenizer: target tokenizer for the target modality
        target_modality: Key of the target modality
        target_embedding_layer: embedding layer for the target modality
        kwargs: Additional model parameters
    Returns:
        BartForConditionalGeneration: Loaded model
    """

    model_config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=target_tokenizer.vocab_size,
        pad_token_id=target_tokenizer.pad_token_id,
        bos_token_id=target_tokenizer.bos_token_id,
        eos_token_id=target_tokenizer.eos_token_id,
        decoder_start_token_id=target_tokenizer.bos_token_id,
        forced_eos_token_id=target_tokenizer.eos_token_id,
        **kwargs,
    )

    bart_model = BartForConditionalGeneration._from_config(model_config)

    # Replace embedding layer
    multimodal_embedding_layer = MultimodalEmbedding(
        data_config, model_config.d_model, multimodal_norm
    )
    bart_model.model.shared = multimodal_embedding_layer
    bart_model.model.encoder.embed_tokens = multimodal_embedding_layer
    bart_model.model.decoder.embed_tokens = (
        multimodal_embedding_layer.embedding_layer_dict[target_modality]
    )

    # Replace Layer Norm
    if multimodal_norm:
        dummy_layer = DummyLayer()
        bart_model.model.encoder.layernorm_embedding = dummy_layer

    # Replace learned pos embedding
    pos_embeds = PositionalEncoding(model_config.d_model)
    bart_model.model.encoder.embed_positions = pos_embeds
    bart_model.model.decoder.embed_positions = pos_embeds

    return bart_model, multimodal_embedding_layer

def load_custom_bart_model(
    model_name: str,
    target_tokenizer: AutoTokenizer,
    target_modality: str,
    data_config: Dict[str, Any],
    multimodal_norm: bool,
    **kwargs,
) -> Tuple[CustomBartForConditionalGeneration, MultimodalEmbedding]:
    """Loads a huggingface bart model.
    Args:
        model_name: Modelname e.g. facebook/bart-large
        target_tokenizer: target tokenizer for the target modality
        target_modality: Key of the target modality
        target_embedding_layer: embedding layer for the target modality
        kwargs: Additional model parameters
    Returns:
        BartForConditionalGeneration: Loaded model
    """

    model_config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=target_tokenizer.vocab_size,
        pad_token_id=target_tokenizer.pad_token_id,
        bos_token_id=target_tokenizer.bos_token_id,
        eos_token_id=target_tokenizer.eos_token_id,
        decoder_start_token_id=target_tokenizer.bos_token_id,
        forced_eos_token_id=target_tokenizer.eos_token_id,
        **kwargs,
    )

    custom_bart_model = CustomBartForConditionalGeneration._from_config(model_config)

    # Replace embedding layer
    multimodal_embedding_layer = MultimodalEmbedding(
        data_config, model_config.d_model, multimodal_norm
    )
    custom_bart_model.model.shared = multimodal_embedding_layer
    custom_bart_model.model.encoder.embed_tokens = multimodal_embedding_layer
    custom_bart_model.model.decoder.embed_tokens = (
        multimodal_embedding_layer.embedding_layer_dict[target_modality]
    )

    # Replace Layer Norm
    if multimodal_norm:
        dummy_layer = DummyLayer()
        custom_bart_model.model.encoder.layernorm_embedding = dummy_layer

    # Replace learned pos embedding
    pos_embeds = PositionalEncoding(model_config.d_model)
    custom_bart_model.model.encoder.embed_positions = pos_embeds
    custom_bart_model.model.decoder.embed_positions = pos_embeds

    return custom_bart_model, multimodal_embedding_layer

def load_t5_model(
    model_name: str,
    target_tokenizer: AutoTokenizer,
    target_modality: str,
    data_config: Dict[str, Any],
    multimodal_norm: bool,
    **kwargs,
) -> Tuple[T5ForConditionalGeneration, MultimodalEmbedding]:

    model_config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=target_tokenizer.vocab_size,
        pad_token_id=target_tokenizer.pad_token_id,
        eos_token_id=target_tokenizer.eos_token_id,
        **kwargs,
    )

    t5_model = T5ForConditionalGeneration._from_config(model_config)

    # Replace embedding layer
    multimodal_embedding_layer = MultimodalEmbedding(
        data_config, model_config.d_model, multimodal_norm
    )
    t5_model.shared = multimodal_embedding_layer
    t5_model.encoder.set_input_embeddings(multimodal_embedding_layer)

    if multimodal_norm:
        target_embedding = nn.Sequential(
            *[
                multimodal_embedding_layer.embedding_layer_dict[target_modality],
                multimodal_embedding_layer.embedding_norm_dict[target_modality],
            ]
        )
        t5_model.decoder.set_input_embeddings(target_embedding)
    else:
        target_embedding = multimodal_embedding_layer.embedding_layer_dict[
            target_modality
        ]
        t5_model.decoder.set_input_embeddings(target_embedding)

    return t5_model, multimodal_embedding_layer


MODEL_REGISTRY: Dict[str, Callable[..., Tuple[PreTrainedModel, MultimodalEmbedding]]] = {
    T5ForConditionalGeneration.__name__: load_t5_model,
    BartForConditionalGeneration.__name__: load_bart_model,
    CustomBartForConditionalGeneration.__name__: load_custom_bart_model,
}


class HFWrapper(pl.LightningModule):
    """Wrapper for Hugging Face models."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        model_type: str,
        model_name: str,
        target_tokenizer: Union[AutoTokenizer, str],
        optimiser: str = "adam",
        num_steps: int = 1000,
        lr: float = 0.001,
        weight_decay: float = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        multimodal_norm: bool = True,
        **kwargs,
    ) -> None:
        """Init
        Args:
            data_config: Data configuration to set up modalities
            model_type: E.g. T5ForConditionGeneration
            model_name: model config to use e.g. google-t5/t5-small
            target_tokenizer: Either string or AutoTokenizer for the target modality. If string load via HF
            optimiser: Which optimiser to use. adam, adamw
            num_steps: Number of training steps
            lr: Learning rate
            weight_decay: weight decay
            adam_beta1: adam beta 1
            adam_beta2: adam beta 2
            multimodal_norm: Wether to apply layer norm to embedding or not
            kwargs: Additional Model parameters
        """

        super().__init__()

        # Wrapper Arguments
        if isinstance(target_tokenizer, str):
            self.target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer)
        else:
            self.target_tokenizer = target_tokenizer

        self.model_type = model_type
        self.model_name = model_name
        self.data_config = data_config
        self.multimodal_norm = multimodal_norm

        # Extract Target modality
        self.target_modality = ""
        for modality, modality_config in self.data_config.items():
            if modality_config["target"]:
                self.target_modality = modality

        # PL arguments
        self.optimiser = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.num_steps = num_steps

        self.validation_step_outputs: List[Dict[str, Any]] = list()
        self.test_step_outputs: List[Dict[str, Any]] = list()

        self.hf_model, self.multimodal_embedding = MODEL_REGISTRY[self.model_type](
            self.model_name,
            self.target_tokenizer,
            self.target_modality,
            self.data_config,
            self.multimodal_norm,
            **kwargs,
        )

        # Use custom barebones generation config to avoid artifacts from logits_processors
        self.generation_config = GenerationConfig(
            bos_token_id=self.target_tokenizer.bos_token_id,
            decoder_start_token_id=self.target_tokenizer.bos_token_id,
            eos_token_id=self.target_tokenizer.eos_token_id,
            forced_eos_token_id=self.target_tokenizer.eos_token_id,
            max_length=128,  # Make variable at some point
            pad_token_id=self.target_tokenizer.pad_token_id,
        )

        self._init_params()

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        """

        for params in self.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)

    def forward(self, batch: Dict[str, Any]) -> Seq2SeqModelOutput:
        """Forward step of the model.
        Args:
            batch: batch containing input, mask, etc.
        Returns:
            Seq2SeqModelOutput: Output of the model, loss included
        """
        # Reformat for HF:
        # 1. Convert masking from True/False to 0/1
        # 2. Reshape: seq_len, batch_size -> batch_size, seq_len
        input_ids = {
            modality: input_ids.transpose(1, 0)
            for modality, input_ids in batch["encoder_input"].items()
        }
        decoder_input = batch["decoder_input"][self.target_modality].transpose(1, 0)

        attention_mask = (~batch["encoder_pad_mask"]).int().T
        decoder_attention_mask = (~batch["decoder_pad_mask"]).int().T

        labels = batch["target"].T.contiguous()

        # Replace pad_token_id with -100 to conform with HF loss calcs
        labels[labels == self.target_tokenizer.pad_token_id] = -100

        # Make encoder embedding
        inputs_embeds = self.multimodal_embedding(input_ids)

        model_output = self.hf_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        return model_output

    def generate(
        self,
        batch: Dict[str, Any],
        n_beams: int = 1,
        logits_processor: Optional[LogitsProcessor] = None,
    ) -> torch.Tensor:
        """Adapter for HF .generate.
        Args:
            batch: batch containing input, mask, etc.
            n_beams: How many beams are used for beam search
        Returns:
            torch.Tensor: generated sequences
        """

        # Get Encoder output
        model_output = self.forward(batch)

        encoder_output = BaseModelOutput(
            last_hidden_state=model_output.encoder_last_hidden_state
        )

        generated_sequences = self.hf_model.generate(
            num_beams=n_beams,
            num_return_sequences=n_beams,
            generation_config=self.generation_config,
            encoder_outputs=encoder_output,
            logits_processor=logits_processor,
            use_cache=False,
        )

        return generated_sequences
    
    def sample_molecules(
        self,
        batch: Dict[str, Any],
        n_beams: int = 1,
        sampling_alg: str = "beam",
        logits_processor: Optional[LogitsProcessor] = None,
    ) -> torch.Tensor:
        """Use same decoding strategy as BART in Open-source code repo.
        Args:
            batch: batch containing input, mask, etc.
            n_beams: How many beams are used for beam search
        Returns:
            torch.Tensor: generated sequences
        """

        decoder = DecodeSampler(
            self.target_tokenizer,
            125,
            target_modality=self.target_modality,
            molecules=True, # Get it from config file from train_multimodal.py
        )

        self.freeze()

        input_ids = {
            modality: input_ids.transpose(1, 0)
            for modality, input_ids in batch["encoder_input"].items()
        }

        attention_mask = (~batch["encoder_pad_mask"]).int().T

        # Make encoder embedding
        inputs_embeds = self.multimodal_embedding(input_ids)

        # Get Encoder output
        encoder_output = self.hf_model.model.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        #encoder_output = BaseModelOutput(
        #    last_hidden_state=model_output.last_hidden_state
        #)

        decode_fn = partial(self._decode_fn, encoder_output=encoder_output,  attention_mask=attention_mask)
                            
        sorted_mols, sorted_lls = decoder.beam_decode(decode_fn, batch_size=int(inputs_embeds.shape[0]), device=inputs_embeds.device, k=n_beams)

        self.unfreeze()

        return sorted_mols, sorted_lls

    def _decode_fn(self, token_ids, pad_mask, encoder_output, attention_mask):
        
        assert token_ids[self.target_modality].shape == pad_mask.shape

        #model_output = self.hf_model(
        #    inputs_embeds=inputs_embeds,
        #    attention_mask=attention_mask,
        #    decoder_input_ids=token_ids[self.target_modality].transpose(1, 0),
        #    decoder_attention_mask=pad_mask.int().T,
        #)

        model_output = self.hf_model.model.decoder(
            input_ids=token_ids[self.target_modality].transpose(1, 0),
            attention_mask=(~pad_mask).int().T,
            encoder_hidden_states=encoder_output[0],
            encoder_attention_mask=attention_mask
        )

        lm_logits = self.hf_model.lm_head(model_output[0])
        #lm_logits = lm_logits + self.hf_model.final_logits_bias.to(lm_logits.device) Full logits

        token_logits = lm_logits.transpose(0,1)
        log_softmax = nn.LogSoftmax(dim=2)
        token_probs = log_softmax(token_logits)
        

        return token_probs
    
    ### Move to Evaluator ###  # noqa: E266
    def score_sequences(
        self,
        generated_sequences: torch.Tensor,
        targets: List[str],
        n_beams: int = 1,
        smiles: bool = True,
    ) -> Dict[str, float]:
        # Move to evaluator
        """Decodes generated sequences and calculates TopN scores.
        Args:
            generated_sequences: sampled sequences from the model
            target: target sequences
            n_beams: n beams used in generation
        Returns:
            Dict[str, float]: Dictionary containing the TopN scores
        """

        decoded_sequences = self.target_tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )
        decoded_sequences = [
            sequence.replace(" ", "") for sequence in decoded_sequences
        ]

        if smiles:
            decoded_sequences = [
                (
                    Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                    if Chem.MolFromSmiles(smiles)
                    else ""
                )
                for smiles in decoded_sequences
            ]
            targets = [
                Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in targets
            ]

        decoded_sequences = [
            decoded_sequences[i * n_beams : (i + 1) * n_beams]
            for i in range(len(targets))
        ]

        ranks = np.array(
            [
                prediction.index(target) if target in prediction else (n_beams + 1)
                for prediction, target in zip(decoded_sequences, targets)
            ]
        )

        scores = dict()
        for i in range(n_beams):
            scores[f"Top{i+1}"] = (ranks <= i).sum() / len(ranks)

        return scores

    ### Pytorch Lightning Requirements ###  # noqa: E266

    def configure_optimizers(self):
        """Set up optimisers for pytorch lightning"""
        params = self.parameters()

        optim = OPTIMISER_REGISTRY[self.optimiser](
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
        )

        print("Using cyclical LR schedule.")
        cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
        sch = {"scheduler": cycle_sch, "interval": "step"}

        return [optim], [sch]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step implementation for pytorch lightning.
        Args:
            batch: batch containing input, mask, etc.
            batch_idx: Batch number
        Returns:
            torch.Tensor: loss
        """
        self.train()
        model_output = self.forward(batch)
        loss = model_output.loss

        if (batch_idx % 10) == 0:
            self.log(
                "train_loss",
                loss,
                prog_bar=True,
                on_step=True,
                logger=True,
                sync_dist=True,
            )

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Validation step implementation for pytorch lightning.
        Args:
            batch: batch containing input, mask, etc.
            batch_idx: Batch number
        Returns:
            Dict[str, Any]: Dictionary containing loss as well as any metrics calculated during validation step
        """

        self.eval()

        model_output = self.forward(batch)
        loss = model_output.loss

        token_acc = self._calc_token_acc(batch, model_output)

        generated_sequences = self.generate(batch)

        scores = self.score_sequences(
            generated_sequences, batch["target_smiles"], n_beams=1
        )

        val_outputs = {
            "val_loss": loss,
            "val_token_acc": token_acc,
            "val_molecular_accuracy_tensorboard": torch.Tensor([scores["Top1"]]),
            "val_molecular_accuracy": torch.Tensor([scores["Top1"]]),
        }

        self.validation_step_outputs.append(val_outputs)
        return val_outputs

    def _calc_token_acc(self, batch_input, model_output):

        token_ids = batch_input["target"].T
        pred_tokens = torch.argmax(model_output.logits, dim=-1)

        target_mask = token_ids != -100
        correct_ids = torch.eq(token_ids, pred_tokens)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total

        return accuracy

    def on_validation_epoch_end(self):
        avg_outputs = self._avg_dicts(self.validation_step_outputs)
        self._log_dict(avg_outputs)
        self.validation_step_outputs = list()

    def test_step(self, batch, batch_idx):
        """Test step implementation for pytorch lightning.
        Args:
            batch: batch containing input, mask, etc.
            batch_idx: Batch number
        Returns:
            Dict[str, Any]: Dictionary containing loss as well as any metrics calculated during test step
        """

        self.eval()

        model_output = self.forward(batch)
        loss = model_output.loss

        generated_sequences = self.generate(batch, n_beams=10)

        scores = self.score_sequences(
            generated_sequences, batch["target_smiles"], n_beams=10
        )

        test_outputs = {
            "test_loss": loss,
            "test_top1": scores["Top1"],
            "test_top2": scores["Top2"],
            "test_top5": scores["Top5"],
            "test_top10": scores["Top10"],
        }

        self.test_step_outputs.append(test_outputs)
        return test_outputs

    def on_test_epoch_end(self):
        avg_outputs = self._avg_dicts(self.test_step_outputs)
        self._log_dict(avg_outputs)

    def _avg_dicts(self, colls: List[Dict[str, Any]]) -> Dict[str, Any]:
        complete_dict: Dict[str, list] = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            for key in complete_dict.keys():
                complete_dict[key].append(coll[key])

        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            if key == "val_molecular_accuracy":
                self.log(
                    "val_molecular_accuracy",
                    val,
                    prog_bar=True,
                    logger=False,
                    sync_dist=True,
                )
            else:
                self.log(key, val, sync_dist=True)

    def _positional_embs(self):
        # Not used but required for current implementation of Wrapper
        """Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])  # type: ignore
        encs = 10000**encs
        encs = [  # type: ignore
            (torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(1024)
        ]
        encs = [torch.stack(enc, dim=1).flatten()[: self.d_model] for enc in encs]  # type: ignore
        encs = torch.stack(encs)  # type: ignore
        return encs
