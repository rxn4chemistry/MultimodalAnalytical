import logging
import math
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from ..configuration import DEFAULT_SETTINGS
from .augmentations import augment
from .data_utils import IterableDatasetWithLength

logger=logging.getLogger(__name__)

valid_modalities = ["text", "vector"]

def identity(x):
    return x

def multi_config_mix(dataset: Dataset, mixture_config: DictConfig, split: str, seed: int = DEFAULT_SETTINGS.default_seed) -> Generator[Dict[str, Any], None, None]:
    mix_generators = [mix_spectra(dataset = dataset,  mix_config=mixture_config[mode], split=split, seed=seed) for mode in mixture_config]
    for samples in zip_longest(*mix_generators, fillvalue=None): # alternate generators to have pseudo-randomicity
        for sample in samples:
            if sample is not None:
                yield sample

def normalize_spectrum(spectrum: List[float]) -> List[float]:
    min_val = min(spectrum)
    max_val = max(spectrum)
    spectrum = [max(0, x) for x in spectrum]
    if max_val - min_val == 0:
        return [0] * len(spectrum)
    return [(x - min_val) / (max_val - min_val) for x in spectrum]

def mix_spectra(dataset: Dataset, mix_config: DictConfig, split: str, seed: int = DEFAULT_SETTINGS.default_seed) -> Generator[Dict[str, Any], None, None]:
    np.random.seed(seed)

    n_compounds = mix_config["n_compounds"]
    compounds_ratio = mix_config["compounds_ratio"]
    parallel_samples = mix_config["parallel_samples"]
    max_n_samples = mix_config[f"{split}_max_n_samples"]
    normalize = mix_config["normalize"]
    mixed = mix_config.get("mixed", False)

    if max_n_samples // parallel_samples < 1:
        parallel_samples = max_n_samples

    if compounds_ratio is None:
        compounds_ratio = [1 / n_compounds] * n_compounds


    if len(compounds_ratio) != n_compounds or sum(compounds_ratio) != 1:
        raise ValueError(
            f"Invalid compound ratios: expected {n_compounds} compounds with ratios summing to 1. "
            f"Found {len(compounds_ratio)} compounds, and their sum is {sum(compounds_ratio):.2f}. "
            "Please ensure the number of compounds and the sum of the ratios are correct."
        )
    
    num_expected = math.perm(len(dataset), n_compounds)
    
    a = list(range(len(dataset)))
    dataset_df = dataset.to_pandas()
    dataset_df = dataset_df[["Smiles", "Formula", "IR"]]
    data = dataset_df.values.tolist()

    if mixed:
        if compounds_ratio != [1 / n_compounds] * n_compounds:
            raise ValueError(
                "Mixed mode is only supported with equal compound ratios at the moment."
            )
        mock = [0] * len(data[0][2])
        for i in range(len(data)):
            yield {
                "Smiles": data[i][0],
                "Formula": data[i][1],
                "IR": normalize_spectrum(data[i][2]) if normalize else data[i][2],
                "Additional_smiles": "mock",
                "Percentage": f"{1 / n_compounds}",
                "IR_target": mock
            }
    else:
        for n in range(max_n_samples // parallel_samples):
            random_indices = np.random.choice(a, size=(parallel_samples, n_compounds))
            random_indices = np.unique(random_indices, axis=0)

            valid_mask = np.array([len(set(row)) == len(row) for row in random_indices])

            random_indices = random_indices[valid_mask]

            if n * parallel_samples +  parallel_samples >= num_expected:
                break

            for idx in random_indices:
                smiles_formula = [data[s][0] for s in idx]
                molecular_formula = [data[s][1] for s in idx]
                spectra = [data[s][2] for s in idx]
                combined_spectrum = np.average(
                    spectra, weights=compounds_ratio, axis=0
                ).tolist()

                if normalize:
                    combined_spectrum = normalize_spectrum(combined_spectrum)

                if len(combined_spectrum) != 1800: # pad the real data
                    combined_spectrum = combined_spectrum + [0] * (1800-len(combined_spectrum))
                for i in range(n_compounds):
                    if compounds_ratio[i] == 0:
                        continue
                    
                    yield {"Smiles": smiles_formula[i], "Formula": molecular_formula[i], "IR": combined_spectrum, "Additional_smiles": ",".join([smiles_formula[idx] for idx in range(n_compounds) if idx != i]), "Percentage": f"{compounds_ratio[i]}", "IR_target": spectra[i]}

def split(dataset: Dataset, cv_split: int = 0, seed: int = 3245) -> DatasetDict:
    """
    Split a dataset into train, test, and validation sets. Allows selection of cv_split.

    Args:
        dataset: The dataset to split.
        cv_split: The index of the cross-validation split to use. Defaults to 0.
        seed: The random seed for the split. Defaults to 3245.

    Returns:
        DatasetDict: A dictionary containing the train, test, and validation sets.
    """
    k_folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(k_folds.split(X=dataset))
    train_indices, test_indices = splits[cv_split][0], splits[cv_split][1]

    test_set = dataset.select(test_indices)
    train_set = dataset.select(train_indices)

    split_data_val = train_set.train_test_split(
        test_size=min(int(0.1 * len(train_set)), DEFAULT_SETTINGS.default_val_set_size),
        shuffle=True,
        seed=seed,
    )

    return DatasetDict(
        {
            "train": split_data_val["train"],
            "test": test_set,
            "validation": split_data_val["test"],
        }
    )


def func_split(data_path, cv_split: int = 0, seed: int = 3453) -> DatasetDict:

    data_path = Path(data_path)
    parquet_paths = data_path.glob("*.parquet")

    data_chunks = list()
    for parquet_path in parquet_paths:
        chunk = pd.read_parquet(parquet_path)
        data_chunks.append(chunk)
    data = pd.concat(data_chunks)

    data["functional_group_names"] = data["functional_group_names"].apply(
        lambda x: ".".join(sorted(x))
    )

    counts: Dict[str, int] = {}
    for sample in data["functional_group_names"]:
        if sample in counts.keys():
            counts[sample] += 1
        else:
            counts[sample] = 1

    counts_df = pd.DataFrame(counts.items(), columns=["functional_groups", "counts"])
    single_counts = counts_df[counts_df["counts"] == 1].copy()
    multi_counts = counts_df[counts_df["counts"] > 1].copy()

    single_counts_df = data[
        data["functional_group_names"].isin(single_counts["functional_groups"])
    ]
    multi_counts_df = data[
        data["functional_group_names"].isin(multi_counts["functional_groups"])
    ]

    if cv_split == -1:
        train_set, test_set = train_test_split(
            multi_counts_df,
            stratify=multi_counts_df["functional_group_names"],
            test_size=0.1,
            random_state=3453,
            shuffle=True,
        )
    else:
        k_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        splits = list(
            k_folds.split(
                X=multi_counts_df, y=multi_counts_df["functional_group_names"]
            )
        )
        train_indices, test_indices = splits[cv_split][0], splits[cv_split][1]
        train_set, test_set = (
            multi_counts_df.iloc[train_indices],
            multi_counts_df.iloc[test_indices],
        )

    train_set, val_set = train_test_split(
        train_set,
        test_size=min(int(0.05 * len(train_set)), DEFAULT_SETTINGS.default_val_set_size),
        random_state=seed,
        shuffle=True,
    )
    train_set = pd.concat([train_set, single_counts_df])

    train_set = Dataset.from_pandas(train_set)
    val_set = Dataset.from_pandas(val_set)
    test_set = Dataset.from_pandas(test_set)

    return DatasetDict({"train": train_set, "test": test_set, "validation": val_set})

def filter_dataset_on_targets(dataset: Dataset, all_targets: List[Any], selected_targets: Set[Any]) -> List[int]:
    """
    Filter a dataset based on the targets.

    Args:
        dataset: The dataset to be filtered.
        all_targets: A list of all targets in the dataset.
        selected_targets: A set of targets to be selected.

    Returns:
        List[int]: A list of indices of the selected targets in the dataset.
    """
    idx = [i for i, target in enumerate(all_targets) if target in selected_targets]
    return dataset.select(idx)


def target_split(dataset: Dataset, target_column: str, cv_split: int = 0, seed: int = 3453) -> DatasetDict:
    """
    Split the dataset based on unique values in the target column.

    Args:
        dataset: The dataset to be split.
        target_column: The name of the target column.
        cv_split: The index of the cross-validation split to use. Defaults to 0.
        seed: The random seed to use for splitting. Defaults to 3453.
    Returns:
        DatasetDict: A dictionary containing the train, test, and validation datasets.
    """

    all_targets = dataset[target_column]
    unique_targets = np.unique(dataset[target_column])

    k_folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(k_folds.split(X=unique_targets))
    train_indices, test_indices = splits[cv_split][0], splits[cv_split][1]

    train_targets, test_targets = (
        unique_targets[train_indices],
        set(unique_targets[test_indices])
    )

    train_targets, val_targets = train_test_split(train_targets, test_size=min(int(0.05 * len(train_targets)), DEFAULT_SETTINGS.default_val_set_size), random_state=seed, shuffle=True)
    train_targets, val_targets = set(train_targets), set(val_targets)

    train_set = filter_dataset_on_targets(dataset, all_targets, train_targets)
    val_set = filter_dataset_on_targets(dataset, all_targets, val_targets)
    test_set = filter_dataset_on_targets(dataset, all_targets, test_targets)

    return DatasetDict({"train": train_set, "test": test_set, "validation": val_set})



def build_dataset_multimodal(
    data_config: Dict[str, Any],
    data_path: str,
    splitting: str,
    cv_split: int,
    augment_config: Optional[DictConfig] = None,
    num_cpu: int = 7,
    mixture_config: Optional[DictConfig] = None,
) -> Tuple[Dict[str, Union[str, int, bool]], DatasetDict]:

    
    if not Path(data_path).is_dir():
        raise ValueError(
            "Data path must specify path to directory containing the dataset files as parqet."
        )
    
    relevant_columns = set()
    for modality in data_config.keys():
        if isinstance(data_config[modality]["column"], str):
            if data_config[modality]["column"] not in ["percentage"] and ("alignment" not in data_config[modality] or not data_config[modality]["alignment"]):
                relevant_columns.add(data_config[modality]["column"])
        elif isinstance(data_config[modality]["column"], list):
            relevant_columns.update(data_config[modality]["column"])
        else:
            raise ValueError(
                f"Expected column to be either list or str for modality: {modality}"
            )
    
    logger.info(f"Loading dataset from {data_path}")
    dataset_dict = load_dataset("parquet", data_dir=data_path, num_proc=num_cpu, columns=list(relevant_columns))
    logger.info("Dataset Loaded")
    # Concatenates all datasets into a single test set
    if splitting == "test_only":
        datasets = list(dataset_dict.values())
        combined_dataset = concatenate_datasets(datasets)
        dataset_dict = DatasetDict({"test": combined_dataset, 'train': combined_dataset, 'validation': combined_dataset})

    # Split based on functinal group occurence. Only relevant for Merck
    elif splitting == "func_group_split":
        dataset_dict = func_split(data_path, cv_split=cv_split, seed=DEFAULT_SETTINGS.default_seed)
    
    # Split based on unique values in the target column
    elif splitting == "unique_target":
        # Get Target column
        target_column = ""
        for modality_config in data_config.values():
            if modality_config["target"] and ("alignment" not in modality_config or not modality_config["alignment"]):
                target_column = modality_config["column"]
                break
        
        # Combine dataset
        datasets = list(dataset_dict.values())
        combined_dataset = concatenate_datasets(datasets)

        # Split Dataset
        dataset_dict = target_split(combined_dataset, target_column, cv_split=cv_split, seed=DEFAULT_SETTINGS.default_seed)

    # Random Split
    elif splitting == "random":
        datasets = list(dataset_dict.values())
        combined_dataset = concatenate_datasets(datasets)
        dataset_dict = split(combined_dataset, cv_split)

    # Sanity check for loading a dataset already split into train/test/val
    elif splitting == "given_splits" and len(dataset_dict) == 3:

        if set(dataset_dict.keys()) != {"train", "validation", "test"}:
            raise ValueError(
                f"Expected ['train', 'validation', 'test'] in dataset but found {list(dataset_dict.keys())}."
            )

    # Raise Error for all edge cases
    else:
        raise ValueError(
            f"Unknown split {splitting}."
        )
    
    # Augment
    dataset_dict['train'] = augment(dataset_dict['train'], augment_config, num_cpu)

    # Rename columns
    rename_columns = dict()
    for modality in data_config.keys():
        if isinstance(data_config[modality]["column"], str):
            if data_config[modality]["column"] not in ["percentage"] and ("alignment" not in data_config[modality] or not data_config[modality]["alignment"]):
                rename_columns[data_config[modality]["column"]] = modality

    processed_dataset_dict = DatasetDict()
    for dataset_key in dataset_dict.keys():
        selected_dataset = dataset_dict[dataset_key]
        processed_dataset = selected_dataset.rename_columns(rename_columns)
        processed_dataset_dict[dataset_key] = processed_dataset

    if isinstance(mixture_config, DictConfig):
        logger.info("Creating mixture dataset")
        for dataset_key in processed_dataset_dict.keys():
            max_samples = sum([mixture_config[conf][f"{dataset_key}_max_n_samples"] for conf in mixture_config])

            processed_dataset_dict[dataset_key] = IterableDatasetWithLength(generator_fn=multi_config_mix, generator_args = {"dataset": processed_dataset_dict[dataset_key], "mixture_config": mixture_config, "split": dataset_key, "seed": DEFAULT_SETTINGS.default_seed}, length=max_samples, split=dataset_key)
            logger.info(f"Max len for {dataset_key}: {processed_dataset_dict[dataset_key]._length}")
        logger.info("IR spectra Iterable Dataset created!")

    return data_config, processed_dataset_dict
