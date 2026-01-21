import json
import random
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss  # type:ignore
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from activeft.sift import Retriever  # type:ignore
from datasets import Dataset, DatasetDict, IterableDataset
from loguru import logger
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
            if modality_config["target"] and ("alignment" not in modality_config or not modality_config["alignment"])
        ]
        alignment_modality_list = [
            modality
            for modality, modality_config in self.data_config.items()
            if modality_config["target"] and "alignment" in modality_config and modality_config["alignment"]
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
            self.max_source_length = self.compute_source_lengths(
                dataset[list(dataset.keys())[0]]
            )

        # Compute max target length if not provided; Only relevant for Text as output
        if (
            self.max_target_length is None
            and self.data_config[self.target_modality]["type"] == "text"
        ):
            self.max_target_length = self.compute_target_length(
                dataset[list(dataset.keys())[0]]
            )

    def compute_source_lengths(self, dataset: Dataset) -> Dict[str, int]:
        max_lengths = dict()

        if isinstance(dataset, IterableDatasetWithLength):
            num_samples = min(DEFAULT_SETTINGS.default_samples, dataset._length)
            sampled_dataset = dataset.take(num_samples)
            sampled_dataset = Dataset.from_generator(lambda: sampled_dataset.__iter__(), split=dataset.split)
        else:
            selected_sample = np.random.randint(
                0, len(dataset), min(DEFAULT_SETTINGS.default_samples, len(dataset))
            )
            sampled_dataset = dataset.select(selected_sample)

        # Compute the max length of each modality
        for modality in self.input_modalities:
            if self.data_config[modality]["type"] == "text":
                for sample in sampled_dataset[modality]:
                    tokenized_sample = self.preprocessors[modality](
                        text=sample, padding=False
                    )["input_ids"]
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
            sampled_dataset = Dataset.from_generator(lambda: sampled_dataset.__iter__(), split=dataset.split)
        else:
            selected_sample = np.random.randint(
                0, len(dataset), min(DEFAULT_SETTINGS.default_samples, len(dataset))
            )
            sampled_dataset = dataset.select(selected_sample)

        for sample in sampled_dataset[self.target_modality]:
            tokenized_sample = self.preprocessors[self.target_modality](
                text=sample, padding=False
            )["input_ids"]
            if len(tokenized_sample) > max_target_length:
                max_target_length = len(tokenized_sample)

        return max_target_length + 5
    
    def __call__(
        self, batch: List[Dict[str, Any]], return_tensors=None
    ) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors

        batch_dict = {
            k: [batch[i][k] for i in range(len(batch))] for k, v in batch[0].items()
        }


        # Prepare Encoder and target
        input_dict, global_input_attention_mask = self.prepare_encoder_input(
            batch_dict, return_tensors
        )


        alignment_input = None
        if len(self.alignment_modality) == 1:
            alignment_input = torch.tensor(np.array(batch_dict[self.alignment_modality[0]]))
            if isinstance(self.preprocessors[self.alignment_modality[0]], PatchPreprocessor) and alignment_input.shape[1] < 1800:
                alignment_input = torch.nn.functional.pad(alignment_input, (0, 1800 - alignment_input.shape[1]), "constant", 0)

            if self.data_config[self.alignment_modality[0]]["type"] == "1D_patches" and self.preprocessors[self.alignment_modality[0]].interplation_merck:
                alignment_input = torch.tensor(self.preprocessors[self.alignment_modality[0]].interpolation_merck(alignment_input), dtype=torch.float32)
        target_tensor = self.prepare_target(batch_dict, return_tensors)

        # Prepare batches for BART or encoder only model
        if self.model_type in [
            "BART",
            "BartForConditionalGeneration",
            "CustomBartForConditionalGeneration",
            "T5ForConditionalGeneration",
            "CustomModel"
        ]:
            tokenized_label_input_ids = target_tensor["input_ids"].transpose(0, 1)

            # Construct decoder input as dict to conform with model wrapper embedding logic
            decoder_input = {self.target_modality: tokenized_label_input_ids[:-1, :]}

            decoder_pad_mask = (
                ~target_tensor["attention_mask"].transpose(0, 1).type(torch.bool)
            )

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

            return_dict =  {
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
                attention_mask = (
                    ~tokenized_modality["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
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
                attention_mask = (
                    ~tokenized_modality["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
                )

                if "numerical_values" in tokenized_modality:
                    tokenized_input = {
                        "tokenized_input": tokenized_input_ids,
                        "numerical_values": tokenized_modality[
                            "numerical_values"
                        ].transpose(0, 1),
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
                attention_mask = (
                    ~tokenized_modality["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
                )

                if "numerical_values" in tokenized_modality:
                    tokenized_input = {
                        "tokenized_input": tokenized_input_ids,
                        "numerical_values": tokenized_modality[
                            "numerical_values"
                        ].transpose(0, 1),
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

                attention_mask = (
                    ~tokenized_spectra["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
                )

            elif self.data_config[modality]["type"] == "run_length_encoding":
                # Text spectrum requires formula and spectra column as keys in the config
                spectra = batch_dict[modality]

                tokenized_modality = self.preprocessors[modality](spectra=spectra)

                tokenized_input_ids = tokenized_modality["input_ids"].transpose(0, 1)
                attention_mask = (
                    ~tokenized_modality["attention_mask"]
                    .transpose(0, 1)
                    .type(torch.bool)
                )

                input_dict[modality] = tokenized_input_ids

            elif self.data_config[modality]["type"] == "1D_patches":
                processed_input, attention_mask = self.preprocessors[modality](
                    batch_dict[modality]
                )
                processed_input = processed_input.transpose(0, 1)

                # All spectra have the same length => No masking; Chemformer uses False to indicate when a token is not masked
                attention_mask = attention_mask.transpose(0, 1)

                input_dict[modality] = processed_input

            if global_input_attention_mask is None:
                global_input_attention_mask = attention_mask
            else:
                global_input_attention_mask = torch.cat(
                    (global_input_attention_mask, attention_mask)
                )

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
            target = self.preprocessors[self.target_modality](
                batch_dict[self.target_modality]
            )
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
            shuffle = False, #True if not isinstance(self.dataset["train"], (IterableDataset, IterableDatasetWithLength)) else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader

    def val_dataloader(
        self,
    ) -> DataLoader:
        
        val_loader = DataLoader(
            self.dataset["validation"],
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle=False if not isinstance(self.dataset["validation"], (IterableDataset, IterableDatasetWithLength)) else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return val_loader

    def predict_dataloader(
        self,
        test_idx: Optional[Path] = None,
    ) -> DataLoader:

        if "eval_sample" in self.data_config and self.data_config["eval_sample"]:
            if test_idx is None:
                #Sample random 10k samples
                selected_sample = np.random.choice(
                    len(self.dataset["test"]),
                    min(10000, len(self.dataset["test"])),
                    replace=False
                )
            else:
                with test_idx.open("rb") as f:
                    selected_sample = np.load(f)
            selected_test_set = self.dataset["test"].select(selected_sample)
        else:
            selected_test_set = self.dataset["test"]

        test_loader = DataLoader(
            selected_test_set,
            collate_fn=self.collator,
            batch_size=self.batch_size if self.batch_size <= 64 else 64, # to make optional for other modalities
            shuffle=False, # if not isinstance(selected_test_set, (IterableDataset, IterableDatasetWithLength)) else None,
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
            extra_columns=self.extra_columns
        )
        return data_collator

class TTTMultiModalDataModule(MultiModalDataModule):
    """Class to perform test-time tuning on the given dataset.
    """
    def __init__(
        self,
        model,
        dataset: DatasetDict,
        preprocessors: Dict[str, Union[AutoTokenizer, PatchPreprocessor]],
        data_config: Dict[str, Union[str, bool, int]],
        model_type: str,
        batch_size: int = 128,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        extra_columns: Optional[List[str]] = None,
        num_workers: int = 8,
        device: str = 'cpu',
        reduced_val: bool = False,
        only_faiss: bool = True,
        path_selection: str = None,
        similarity_criterion: str = 'embeddings',
        nearest_neighbors: int = None,
    ):
        # inherit all the methods from MultiModalDataModule
        super().__init__(dataset, preprocessors, data_config, model_type, batch_size, max_source_length, max_target_length, extra_columns)

        self.model = model
        self.device = device
        self.num_workers = num_workers
        self.only_faiss = only_faiss
        self.path_selection = path_selection
        self.similarity_criterion = similarity_criterion
        self.nearest_neighbors = nearest_neighbors

        if reduced_val:
            if len(self.dataset["validation"]) > self.batch_size:
                indices_val = random.sample(range(len(self.dataset["validation"])), self.batch_size) # using 1 batch
                self.dataset["validation"] = self.dataset["validation"].select(indices_val)
                logger.info(f'Using only {self.batch_size} samples from validation set')

        # implement anyways the standard datamodule, needed to access the processed data
        self.datamodule = MultiModalDataModule(
            dataset=self.dataset,
            preprocessors=self.preprocessors,
            data_config=self.data_config,
            model_type=self.model_type,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            extra_columns=self.extra_columns
        )
        
        self.epochs = 0
        self.d = self.model.hf_model.encoder.norm.normalized_shape[0] # Dimension of the vectors
        self.d_fp = self.model.hf_model.align_network[2].out_features if self.similarity_criterion == "fingerprints" else self.d


    def make_embeddings(self, dim, batches, device, save=None):
        """Make embeddings for vectors in the test set for compression mode mean
        """
        embeddings = torch.empty((0, dim)).to(device)
        for batch in iter(batches):
            embeddings_chunk, att_mask = self.make_embeddings_datamodule(batch, save)
            embeddings_chunk = torch.stack([torch.mean(vec[att==1], dim=0) for vec, att in zip(embeddings_chunk, att_mask)])
            embeddings = torch.cat((embeddings, embeddings_chunk), 0)
        
        return embeddings.detach().cpu()

    def make_embeddings_datamodule(self, batch, save=None):
        """Make the embeddings of the data given the model, using the dataloader (takes care of the batches automatically)
        """
        with torch.no_grad():
            input_ids = {modality: input_ids.transpose(1, 0).to(self.device) for modality, input_ids in batch["encoder_input"].items()}
            input_embeds = self.model.multimodal_embedding(input_ids)
            attention_mask = (~batch["encoder_pad_mask"]).int().T.to(self.device)
            embeddings = self.model.hf_model.encode(input_embeds, attention_mask)
        
        if save and isinstance(save, str):
            data = pd.DataFrame()
            data['last_hidden_state'] = embeddings.last_hidden_state.cpu().numpy().tolist()
            data['attention_mask'] = embeddings.attention_mask.cpu().numpy().tolist()
            with Path.open(Path(f'{self.path_selection}/{save}_embeddings_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(data.to_json(), f)
            f.close()

        return embeddings.last_hidden_state, embeddings.attention_mask

    def get_fingerprints(self, loader):

        embeddings = self.make_embeddings(self.d, loader, self.device)
        fingerprints = self.model.hf_model.predict_fingerprint(embeddings=embeddings)
        
        del embeddings

        return fingerprints

class KmeansTTTMultiModalDataModule(TTTMultiModalDataModule):
    """Class to perform K-means clustering on the test set of the given dataset,
    to then use only one point per cluster to perform test-time tuning.
    """
    def __init__(
            self,
            model,
            dataset,
            preprocessors,
            data_config,
            model_type,
            batch_size = 128,
            max_source_length = None,
            max_target_length = None,
            extra_columns = None,
            num_workers = 8,
            device = 'cpu',
            reduced_val = False,
            only_faiss = True,
            path_selection = None,
            similarity_criterion = 'fingerprints',
            nearest_neighbors = None,
            n_clusters = None,
            n_test_points = None,
            n_train_points = None,
            update_embeds = 10, # False or int
            seed = 3247
        ):
        
        super().__init__(model, dataset, preprocessors, data_config, model_type, batch_size, max_source_length, max_target_length, extra_columns, num_workers, device, reduced_val, only_faiss, path_selection, similarity_criterion, nearest_neighbors)

        # make embeddings (fps) train set
        logger.info('Making embeddings train set')
        fingerprints_train = self.get_fingerprints(self.datamodule.train_dataloader())
        tensor = torch.nn.functional.normalize(fingerprints_train, p=2, dim=1)
        tensor = tensor.detach().cpu()
        self.train_tensor = tensor
        
        # make embeddings (fps) test points
        logger.info('Making embeddings test set')
        fingerprints_test = self.get_fingerprints(self.datamodule.predict_dataloader())
        tensor = torch.nn.functional.normalize(fingerprints_test, p=2, dim=1)
        tensor = tensor.detach().cpu()
        self.test_tensor = tensor

        # save predicted fps
        if self.path_selection:
            df = self.dataset['train'].to_pandas()
            df['predicted_fingerprints'] = self.train_tensor.numpy().tolist()
            df['predicted_fingerprints_before_norm'] = fingerprints_train.detach().cpu().numpy().tolist()
            with Path.open(Path(f'{self.path_selection}/train_tensor_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(df.to_json(), f)
            f.close()
            del df

            df = self.dataset['test'].to_pandas()
            df['predicted_fingerprints'] = self.test_tensor.numpy().tolist()
            df['predicted_fingerprints_before_norm'] = fingerprints_test.detach().cpu().numpy().tolist()
            with Path.open(Path(f'{self.path_selection}/test_tensor_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(df.to_json(), f)
            f.close()
            del df
        
        del tensor, fingerprints_test, fingerprints_train
        
        # Perform k-means clustering with ncentroids and save the centroids
        logger.info(f'Clustering the test points in {n_clusters} clusters')
        kmeans = faiss.Kmeans(self.d_fp, n_clusters, niter=200, verbose=True, gpu=True, seed=seed)
        kmeans.train(self.test_tensor)

        # Calculate here the indices of the respective cluster for every test point
        D, I = kmeans.index.search(self.test_tensor, 1)
        self.I = I.reshape((len(I)))
        self.D = D

        # Get the clusters
        dict_clusters = dict.fromkeys(self.I)
        # Populate the dictionary
        for i, centroid in enumerate(kmeans.centroids):
            dict_clusters[i] = dict.fromkeys(['centroid','indices','ind_closest'], [])
            dict_clusters[i]['centroid'] = centroid
            # Save the index of the test points we want to use for selection
            index = faiss.IndexFlatIP(self.d_fp)
            index.add(self.test_tensor)
            _, indices = index.search(np.array([centroid]), n_test_points)
            dict_clusters[i]['ind_closest'] = indices[0]
        # Save the indices of the test points belonging to each cluster (sequential)
        for idx, cluster_id in enumerate(self.I):
            dict_clusters[cluster_id]['indices'].append(idx)
        self.clusters = dict_clusters

        self.n_test_points = n_test_points
        self.n_train_points = n_train_points if n_train_points else int(self.batch_size/n_test_points)
        self.update_embeds = update_embeds

        del kmeans
    
    def get_train_tensor(self):
        return self.train_tensor
    
    def get_test_tensor(self):
        return self.test_tensor

    def get_centroids(self):
        return self.centroids
    
    def get_clusters(self):
        return self.dict_clusters
    
    def get_clusters_info(self):
        clusters = self.get_clusters()

        for cluster_id in sorted(clusters.keys()):
            d_avg = np.mean(self.D[clusters[cluster_id]['indices']])
            d_max = np.max(self.D[clusters[cluster_id]['indices']])
            d_min = np.min(self.D[clusters[cluster_id]['indices']])
            logger.info(f"Cluster {cluster_id}: {len(clusters[cluster_id]['indices'])} points\nAvg distance = {d_avg}\n Max distance = {d_max}\n Min distance = {d_min}")

    def get_cluster_distances(self):
        return self.D

    def get_closest_points(self, centroid, n_points):
        index = faiss.IndexFlatIP(self.d_fp)
        index.add(self.test_tensor)
        _, indices = index.search(centroid, n_points)

        return self.test_tensor[indices[0]]

    # overwrite only the train dataloader
    def train_dataloader(self) -> DataLoader:

        # Update embeddings
        if self.update_embeds and self.update_embeds > 0:
            if self.epochs > 0 and self.epochs % self.update_embeds == 0:

                logger.info("Recomputing embeddings/fingerprints for the training set")
                fingerprints_train = self.get_fingerprints(self.datamodule.train_dataloader())
                tensor = torch.nn.functional.normalize(fingerprints_train, p=2, dim=1)
                tensor = tensor.detach().cpu()
                self.train_tensor = tensor
                
                logger.info("Recomputing embeddings/fingerprints for the test set")
                fingerprints_test = self.get_fingerprints(self.datamodule.predict_dataloader())
                tensor = torch.nn.functional.normalize(fingerprints_test, p=2, dim=1)
                tensor = tensor.detach().cpu()
                self.test_tensor = tensor

                # save predicted fps
                if self.path_selection:
                    df = self.dataset['train'].to_pandas()
                    df['predicted_fingerprints'] = self.train_tensor.numpy().tolist()
                    df['predicted_fingerprints_before_norm'] = fingerprints_train.detach().cpu().numpy().tolist()
                    with Path.open(Path(f'{self.path_selection}/train_tensor_iteration_{self.epochs}.json'), 'w') as f:
                        json.dump(df.to_json(), f)
                    f.close()
                    del df

                    df = self.dataset['test'].to_pandas()
                    df['predicted_fingerprints'] = self.test_tensor.numpy().tolist()
                    df['predicted_fingerprints_before_norm'] = fingerprints_test.detach().cpu().numpy().tolist()
                    with Path.open(Path(f'{self.path_selection}/test_tensor_iteration_{self.epochs}.json'), 'w') as f:
                        json.dump(df.to_json(), f)
                    f.close()
                    del df

        id_cluster = self.epochs % len(self.clusters)
        logger.info(f"Training model for cluster id={id_cluster}")
        # Select the test point we want to use for selection in this cluster (we are already storing the index)
        ind_test_points = self.clusters[id_cluster]['ind_closest']
        test_points = self.test_tensor[ind_test_points] # in this way we update the representation consistently
                
        if self.similarity_criterion == 'fingerprints':
            indices = []
            cpu_index = faiss.IndexFlatIP(self.d_fp)  # inner product for cosine similarity
            cpu_index.add(self.train_tensor)

            # retrieval of datapoints using activeft code
            retriever = Retriever(cpu_index, fast=False, only_faiss=self.only_faiss, also_query_opposite=True) # we're using inner product index, so negative sim values should also be considered
            indices = []
            time_tot_faiss = 0
            time_tot_sift = 0
            for test_point in test_points:
                _, ind, _, time_retrieval = retriever.search(np.array([test_point]), N=self.n_train_points, K=self.nearest_neighbors, threads=self.num_workers)
                indices = np.concatenate((indices,ind), axis=None)
                time_tot_faiss += time_retrieval.faiss
                time_tot_sift += time_retrieval.sift
            indices_nopad = indices[indices >= 0] # type:ignore
            logger.info(f"Time taken for faiss selection = {time_tot_faiss}")
            logger.info(f"Additional time taken for sift selection = {time_tot_sift}")
        else:
            raise ValueError(f"Selection with similarity criterion {self.similarity_criterion} not implemented.")

        # memorize selected indices
        for ind in indices:
            self.model.set_sel_indices.add(ind)

        # Save selection
        if self.path_selection:
            df = self.dataset['train'].select(indices_nopad).to_pandas()
            df['predicted_fingerprints'] = self.train_tensor[indices_nopad].numpy().tolist()
            df['index'] = indices.astype(int)
            
            df_test = pd.DataFrame()
            df_test['index'] = int(ind_test_points[0])
            df_test['predicted_fingerprints'] = test_points.numpy().tolist()
            for k in self.data_config.keys():
                df_test[k] = self.dataset['test'].select(ind_test_points)[k]
            
            with Path.open(Path(f'{self.path_selection}/sel_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(df.to_json(), f)
            f.close()
            del df

            with Path.open(Path(f'{self.path_selection}/test_iteration_{self.epochs}.json'), 'w') as f:
                json.dump(df_test.to_json(), f)
            f.close()
            del df_test

        train_loader = DataLoader(
            self.dataset["train"].select(indices_nopad),
            collate_fn=self.collator,
            batch_size=self.batch_size,
            shuffle = True if not isinstance(self.dataset["train"], (IterableDataset, IterableDatasetWithLength)) else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.epochs += 1

        del cpu_index, retriever

        return train_loader
    
    def predict_dataloader(
        self,
    ) -> DataLoader:
        """It selects the points in the test set that are only part of the cluster in question."""

        logger.info(f'Predicting for {len(self.dataset["test"])} test points')

        test_loader = DataLoader(
            self.dataset["test"],
            collate_fn=self.collator,
            batch_size=self.batch_size if self.batch_size <= 64 else 64,
            shuffle=False if not isinstance(self.dataset["test"], (IterableDataset, IterableDatasetWithLength)) else None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return test_loader
