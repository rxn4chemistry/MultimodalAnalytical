from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import Dataset, DatasetDict, IterableDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..configuration import DEFAULT_SETTINGS
from .data_utils import IterableDatasetWithLength
from .preprocessors import PatchPreprocessor


@dataclass
class MultiModalDataCollator:
    preprocessors: Dict[str, Any]
    data_config: Dict[str, Any]
    model_type: str

    dataset: InitVar[DatasetDict]
    extra_columns: Optional[List[str]] = None

    padding: bool = True
    max_source_length: Optional[Dict[str, int]] = None
    max_target_length: Optional[int] = None
    return_tensors: str = "pt"

    input_modalities: List[str] = field(init=False)
    target_modality: str = field(init=False)
    alignment_modality: List[str] = field(init=False)

    def __post_init__(self, dataset: DatasetDict):
        """
        Determines the input and target modalities. If not provided computes the max_source and max_target length.
        """
        input_modalities = [
            modality
            for modality, modality_config in self.data_config.items()
            if not modality_config["target"]
        ]
        target_modality_list = [
            modality
            for modality, modality_config in self.data_config.items()
            if modality_config["target"]
            and ("alignment" not in modality_config or not modality_config["alignment"])
        ]
        alignment_modality_list = [
            modality
            for modality, modality_config in self.data_config.items()
            if modality_config["target"]
            and "alignment" in modality_config
            and modality_config["alignment"]
        ]
        if len(alignment_modality_list) > 1:
            raise ValueError("At most 1 target alignment modality can be specified.")
        if len(target_modality_list) != 1:
            raise ValueError("Only 1 target modality can be specified.")

        target_modality = target_modality_list[0]

        self.input_modalities = input_modalities
        self.target_modality = target_modality
        self.alignment_modality = alignment_modality_list

        # Compute max source length if not provided
        if self.max_source_length is None:
            self.max_source_length = self.compute_source_lengths(dataset[list(dataset.keys())[0]])

        # Compute max target length if not provided; Only relevant for Text as output
        if (
            self.max_target_length is None
            and self.data_config[self.target_modality]["type"] == "text"
        ):
            self.max_target_length = self.compute_target_length(dataset[list(dataset.keys())[0]])

    def compute_source_lengths(self, dataset: Dataset) -> Dict[str, int]:
        max_lengths = dict()

        if isinstance(dataset, IterableDatasetWithLength):
            num_samples = min(DEFAULT_SETTINGS.default_samples, dataset._length)
            sampled_dataset = dataset.take(num_samples)
            sampled_dataset = Dataset.from_generator(
                lambda: sampled_dataset.__iter__(), split=dataset.split
            )
        else:
            selected_sample = np.random.randint(
                0, len(dataset), min(DEFAULT_SETTINGS.default_samples, len(dataset))
            )
            sampled_dataset = dataset.select(selected_sample)

        # Compute the max length of each modality
        for modality in self.input_modalities:
            if self.data_config[modality]["type"] == "text":
                for sample in sampled_dataset[modality]:
                    tokenized_sample = self.preprocessors[modality](text=sample, padding=False)[
                        "input_ids"
                    ]
                    if modality not in max_lengths:
                        max_lengths[modality] = len(tokenized_sample) + 5
                    else:
                        if (len(tokenized_sample) + 5) > max_lengths[modality]:
                            max_lengths[modality] = len(tokenized_sample) + 5
            elif self.data_config[modality]["type"] == "1D_patches":
                sample = sampled_dataset.select([0])[modality]
                processed_sample, _ = self.preprocessors[modality](sample)
                max_length_patches = processed_sample.shape[1]
                max_lengths[modality] = max_length_patches

        return max_lengths

    def compute_target_length(self, dataset: Dataset) -> int:
        # Determine the max length of the target modality

        max_target_length = 0

        if isinstance(dataset, IterableDatasetWithLength):
            num_samples = min(DEFAULT_SETTINGS.default_samples, dataset._length)
            sampled_dataset = dataset.take(num_samples)
            sampled_dataset = Dataset.from_generator(
                lambda: sampled_dataset.__iter__(), split=dataset.split
            )
        else:
            selected_sample = np.random.randint(
                0, len(dataset), min(DEFAULT_SETTINGS.default_samples, len(dataset))
            )
            sampled_dataset = dataset.select(selected_sample)

        for sample in sampled_dataset[self.target_modality]:
            tokenized_sample = self.preprocessors[self.target_modality](text=sample, padding=False)[
                "input_ids"
            ]
            if len(tokenized_sample) > max_target_length:
                max_target_length = len(tokenized_sample)

        return max_target_length + 5

    def __call__(self, batch: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        batch_dict = {k: [batch[i][k] for i in range(len(batch))] for k, v in batch[0].items()}

        # Prepare Encoder and target
        input_dict, global_input_attention_mask = self.prepare_encoder_input(
            batch_dict, return_tensors
        )

        alignment_input = None
        if len(self.alignment_modality) == 1:
            if self.alignment_modality[0] not in batch_dict:
                alignment_input = torch.zeros(len(batch_dict[list(batch_dict.keys())[0]]), 1800)
            else:
                alignment_input = torch.tensor(np.array(batch_dict[self.alignment_modality[0]]))

            if alignment_input.shape[1] < 1800:
                alignment_input = torch.nn.functional.pad(
                    alignment_input, (0, 1800 - alignment_input.shape[1]), "constant", 0
                )

            if (
                self.data_config[self.alignment_modality[0]]["type"] == "1D_patches"
                and self.preprocessors[self.alignment_modality[0]].interpolation
            ):
                alignment_input = torch.tensor(
                    self.preprocessors[self.alignment_modality[0]].interpolation(alignment_input),
                    dtype=torch.float32,
                )
        target_tensor = self.prepare_target(batch_dict, return_tensors)

        # Prepare batches for BART or encoder only model
        if self.model_type in [
            "BART",
            "BartForConditionalGeneration",
            "CustomBartForConditionalGeneration",
            "T5ForConditionalGeneration",
            "CustomModel",
        ]:
            tokenized_label_input_ids = target_tensor["input_ids"].transpose(0, 1)

            # Construct decoder input as dict to conform with model wrapper embedding logic
            decoder_input = {self.target_modality: tokenized_label_input_ids[:-1, :]}

            decoder_pad_mask = ~target_tensor["attention_mask"].transpose(0, 1).type(torch.bool)

            if self.data_config[self.target_modality]["type"] == "carbon":
                target = self.preprocessors[self.target_modality].process_carbon(
                    batch_dict[self.target_modality]
                )
            elif self.data_config[self.target_modality]["type"] == "multiplets":
                target = self.preprocessors[self.target_modality].process_multiplets(
                    batch_dict[self.target_modality],
                    encoding=self.preprocessors[self.target_modality].encoding,
                    j_values=self.preprocessors[self.target_modality].j_values,
                )[0]
            else:
                target = batch_dict[self.target_modality]

            return_dict = {
                "encoder_input": input_dict,
                "encoder_pad_mask": global_input_attention_mask,
                "decoder_input": decoder_input,
                "decoder_pad_mask": decoder_pad_mask[:-1, :],
                "target": tokenized_label_input_ids.clone()[1:, :],
                "target_mask": decoder_pad_mask.clone()[1:, :],
                "target_smiles": target,
            }

            if alignment_input is not None:
                return_dict["encoder_alignment_input"] = alignment_input

            if self.extra_columns and self.extra_columns != [None]:
                for col in self.extra_columns:
                    if col not in return_dict:
                        return_dict[col] = batch_dict[col]
            return return_dict

        elif self.model_type == "encoder":
            return {
                "encoder_input": input_dict,
                "encoder_pad_mask": global_input_attention_mask,
                "target": target_tensor,
            }

        else:
            raise ValueError(f"Unknown model type {self.model_type}")

    def prepare_encoder_input(
        self, batch_dict: Dict[str, Any], return_tensors: str
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        input_dict = dict()

        # Irregular attention mask
        global_input_attention_mask = None
        for modality in self.input_modalities:
            if self.data_config[modality]["type"] == "text":
                tokenized_modality = self.preprocessors[modality](
                    batch_dict[modality],
                    padding="max_length",
                    max_length=self.max_source_length[modality],  # type: ignore
                    truncation=True,
                    return_tensors=return_tensors,
                )

                tokenized_input = tokenized_modality["input_ids"].transpose(0, 1)
                attention_mask = ~tokenized_modality["attention_mask"].transpose(0, 1).type(
                    torch.bool
                )

                input_dict[modality] = tokenized_input

            elif self.data_config[modality]["type"] in [
                "multiplets",
                "carbon",
                "msms_text",
                "msms_number",
            ]:
                tokenized_modality = self.preprocessors[modality](batch_dict[modality])

                tokenized_input_ids = tokenized_modality["input_ids"].transpose(0, 1)
                attention_mask = ~tokenized_modality["attention_mask"].transpose(0, 1).type(
                    torch.bool
                )

                if "numerical_values" in tokenized_modality:
                    tokenized_input = {
                        "tokenized_input": tokenized_input_ids,
                        "numerical_values": tokenized_modality["numerical_values"].transpose(0, 1),
                    }
                else:
                    tokenized_input = tokenized_input_ids

                input_dict[modality] = tokenized_input

            elif self.data_config[modality]["type"] in "text_spectrum":
                # Text spectrum requires formula and spectra column as keys in the config

                spectra = batch_dict[self.data_config[modality]["spectra_column"]]

                if self.data_config[modality]["spectra_only"]:
                    formulae = None
                else:
                    formulae = batch_dict[self.data_config[modality]["formula_column"]]

                tokenized_modality = self.preprocessors[modality](
                    formulae=formulae, spectra=spectra
                )

                tokenized_input_ids = tokenized_modality["input_ids"].transpose(0, 1)
                attention_mask = ~tokenized_modality["attention_mask"].transpose(0, 1).type(
                    torch.bool
                )

                if "numerical_values" in tokenized_modality:
                    tokenized_input = {
                        "tokenized_input": tokenized_input_ids,
                        "numerical_values": tokenized_modality["numerical_values"].transpose(0, 1),
                    }
                else:
                    tokenized_input = tokenized_input_ids

                input_dict[modality] = tokenized_input

            elif self.data_config[modality]["type"] == "peak_positional_encoding":
                spectra = batch_dict[modality]
                tokenized_spectra = self.preprocessors[modality](spectra=spectra)

                tokenized_input = tokenized_spectra["input_ids"].transpose(0, 1)
                token_indices = tokenized_spectra["indices"]
                input_dict[modality] = {
                    "tokenized_input": tokenized_input,
                    "token_indices": token_indices,
                }

                attention_mask = ~tokenized_spectra["attention_mask"].transpose(0, 1).type(
                    torch.bool
                )

            elif self.data_config[modality]["type"] == "run_length_encoding":
                # Text spectrum requires formula and spectra column as keys in the config
                spectra = batch_dict[modality]

                tokenized_modality = self.preprocessors[modality](spectra=spectra)

                tokenized_input_ids = tokenized_modality["input_ids"].transpose(0, 1)
                attention_mask = ~tokenized_modality["attention_mask"].transpose(0, 1).type(
                    torch.bool
                )

                input_dict[modality] = tokenized_input_ids

            elif self.data_config[modality]["type"] == "1D_patches":
                processed_input, attention_mask = self.preprocessors[modality](batch_dict[modality])
                processed_input = processed_input.transpose(0, 1)

                # All spectra have the same length => No masking; Chemformer uses False to indicate when a token is not masked
                attention_mask = attention_mask.transpose(0, 1)

                input_dict[modality] = processed_input

            if global_input_attention_mask is None:
                global_input_attention_mask = attention_mask
            else:
                global_input_attention_mask = torch.cat((
                    global_input_attention_mask,
                    attention_mask,
                ))

        return input_dict, global_input_attention_mask

    def prepare_target(self, batch_dict: Dict[str, Any], return_tensors: str) -> Any:
        if self.data_config[self.target_modality]["type"] == "text":
            target_tensor = self.preprocessors[self.target_modality](
                text=batch_dict[self.target_modality],
                max_length=self.max_target_length,
                padding=self.padding,
                return_tensors=return_tensors,
                truncation=True,
            )
        elif self.data_config[self.target_modality]["type"] in ["carbon", "multiplets"]:
            target_tensor = self.preprocessors[self.target_modality](
                batch_dict[self.target_modality]
            )

        elif self.data_config[self.target_modality]["type"] in [
            "functional_group",
            "class_one_hot",
        ]:
            target = self.preprocessors[self.target_modality](batch_dict[self.target_modality])
            target_tensor = torch.Tensor(target)
        elif self.data_config[self.target_modality]["type"] == "no_action":
            target_tensor = torch.Tensor(batch_dict[self.target_modality])

        elif self.data_config[self.target_modality]["type"] == "normalise":
            processed_data = self.preprocessors[self.target_modality](
                np.array(batch_dict[self.target_modality])
            )
            target_tensor = torch.Tensor(processed_data)

        else:
            raise ValueError(f"Unknown Target type: {self.target_modality}")

        return target_tensor


class MultiModalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetDict,
        preprocessors: Dict[str, Union[AutoTokenizer, PatchPreprocessor]],
        data_config: Dict[str, Union[str, bool, int]],
        model_type: str,
        batch_size: int = 128,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        extra_columns: Optional[List[str]] = None,
        num_workers: int = 8,
    ):
        super().__init__()

        self.dataset = dataset
        self.preprocessors = preprocessors
        self.data_config = data_config
        self.model_type = model_type
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_workers = num_workers
        self.extra_columns = extra_columns

        self.collator = self.get_multimodal_data_collator()

    # Abstract functions that we dont use
    def setup(self, *args, **kwargs):
        pass

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self) -> DataLoader:

        train_loader = DataLoader(
            self.dataset["train"],
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=True
            if not isinstance(self.dataset["train"], (IterableDataset, IterableDatasetWithLength))
            else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader

    def val_dataloader(
        self,
    ) -> DataLoader:

        if isinstance(self.dataset["validation"], IterableDatasetWithLength):
            selected_validation_set = self.dataset["validation"].take(
                min(10000, self.dataset["validation"]._length)
            )
            selected_validation_set = Dataset.from_generator(
                lambda: selected_validation_set.__iter__(), split="validation"
            )
        else:
            selected_sample = np.random.choice(
                len(self.dataset["validation"]),
                min(10000, len(self.dataset["validation"])),
                replace=False,
            )
            selected_validation_set = self.dataset["validation"].select(selected_sample)

        val_loader = DataLoader(
            selected_validation_set,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False
            if not isinstance(
                self.dataset["validation"], (IterableDataset, IterableDatasetWithLength)
            )
            else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return val_loader

    def predict_dataloader(
        self,
        test_idx: Optional[Path] = None,
    ) -> DataLoader:

        if isinstance(self.dataset["test"], IterableDatasetWithLength):
            selected_test_set = self.dataset["test"].take(min(10000, self.dataset["test"]._length))
            selected_test_set = Dataset.from_generator(
                lambda: selected_test_set.__iter__(), split="test"
            )
        else:
            if test_idx is None:
                # Sample random 10k samples
                selected_sample = np.random.choice(
                    len(self.dataset["test"]), min(10000, len(self.dataset["test"])), replace=False
                )
            else:
                with test_idx.open("rb") as f:
                    selected_sample = np.load(f)

            selected_test_set = self.dataset["test"].select(selected_sample)

        test_loader = DataLoader(
            selected_test_set,
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False
            if not isinstance(self.dataset["test"], (IterableDataset, IterableDatasetWithLength))
            else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return test_loader

    def get_multimodal_data_collator(self) -> MultiModalDataCollator:
        data_collator = MultiModalDataCollator(
            preprocessors=self.preprocessors,
            data_config=self.data_config,
            dataset=self.dataset,
            model_type=self.model_type,
            extra_columns=self.extra_columns,
        )
        return data_collator
