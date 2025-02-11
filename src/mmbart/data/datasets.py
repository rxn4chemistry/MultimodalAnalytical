from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from rdkit import Chem  # RDLogger
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from mmbart.defaults import (
    DEFAULT_SEED,
    DEFAULT_TEST_SET_SIZE,
    DEFAULT_VAL_SET_SIZE,
)

valid_modalities = ["text", "vector"]


def split(dataset: Dataset, cv_split: int = 0) -> DatasetDict:
    all_indices_shuffled = np.arange(0, len(dataset), 1)
    np.random.shuffle(all_indices_shuffled)

    splits = np.array_split(all_indices_shuffled, int(1 / DEFAULT_TEST_SET_SIZE))
    test_indices = splits[cv_split]
    train_indices = all_indices_shuffled[~np.isin(all_indices_shuffled, test_indices)]

    test_set = dataset.select(test_indices)
    train_set = dataset.select(train_indices)

    split_data_val = train_set.train_test_split(
        test_size=min(int(0.1 * len(train_set)), DEFAULT_VAL_SET_SIZE),
        shuffle=True,
        seed=DEFAULT_SEED,
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
        test_size=min(int(0.05 * len(train_set)), DEFAULT_VAL_SET_SIZE),
        random_state=seed,
        shuffle=True,
    )
    train_set = pd.concat([train_set, single_counts_df])

    train_set = Dataset.from_pandas(train_set)
    val_set = Dataset.from_pandas(val_set)
    test_set = Dataset.from_pandas(test_set)

    return DatasetDict({"train": train_set, "test": test_set, "validation": val_set})


def smiles_split_fn(
    data_path: str, target_column: str, cv_split: int = 0, seed: int = 3453
):
    datapath = Path(data_path)
    parquet_paths = datapath.glob("*.parquet")

    data_chunks = list()
    for parquet_path in parquet_paths:
        chunk = pd.read_parquet(parquet_path)
        data_chunks.append(chunk)
    data = pd.concat(data_chunks)

    unique_smiles = pd.unique(data[target_column])

    k_folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(k_folds.split(X=unique_smiles))
    train_indices, test_indices = splits[cv_split][0], splits[cv_split][1]
    train_smiles, test_smiles = (
        unique_smiles[train_indices],
        unique_smiles[test_indices],
    )

    train_set, test_set = (
        data[data[target_column].isin(train_smiles)],
        data[data[target_column].isin(test_smiles)],
    )
    train_set, val_set = train_test_split(
        train_set,
        test_size=min(int(0.05 * len(train_set)), DEFAULT_VAL_SET_SIZE),
        random_state=seed,
        shuffle=True,
    )

    train_set = Dataset.from_pandas(train_set)
    val_set = Dataset.from_pandas(val_set)
    test_set = Dataset.from_pandas(test_set)

    return DatasetDict({"train": train_set, "test": test_set, "validation": val_set})



def build_dataset_multimodal(
    data_config: Dict[str, Any],
    data_path: str,
    splitting_procedure: str,
    cv_split: int,
    augment_names: Optional[str] = None,
    augment_path: Optional[str] = None,
    augment_fraction: float = 0.0,
    augment_model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Union[str, int, bool]], DatasetDict]:
    
    if not Path(data_path).is_dir():
        raise ValueError(
            "Data path must specify path to directory containing the dataset files as parqet."
        )

    dataset_dict = load_dataset("parquet", data_dir=data_path)

    # Concatenates all datasets into a single test set
    if splitting_procedure == "test_only":
        datasets = list(dataset_dict.values())
        combined_dataset = concatenate_datasets(datasets)
        dataset_dict = DatasetDict({"test": combined_dataset})

    # Split based on functinal group occurence. Only relevant for Merck
    elif splitting_procedure == "func_group_split":
        dataset_dict = func_split(data_path, cv_split=cv_split, seed=DEFAULT_SEED)
    
    # Split based on unique values in the target column
    elif splitting_procedure == "unique_target":
        target_column = ""
        for modality_config in data_config.values():
            if modality_config["target"]:
                target_column = modality_config["column"]
                break
        dataset_dict = smiles_split_fn(
            data_path, target_column, cv_split=cv_split, seed=DEFAULT_SEED
        )

    # Random Split if dataset is not already split into train/test/val
    elif len(dataset_dict) == 1:
        dataset = list(dataset_dict.values())[0]
        dataset_dict = split(dataset, cv_split)

    # Sanity check for loading a dataset already split into train/test/val
    elif len(dataset_dict) == 3: 

        if set(dataset_dict.keys()) != {"train", "validation", "test"}:
            raise ValueError(
                f"Expected ['train', 'validation', 'test'] in dataset but found {list(dataset_dict.keys())}."
            )

    # Raise Error for all edge cases
    else:
        raise ValueError(
            f"Failsed to load Dataset. Excpected to find three datasets with name ['train', 'validation', 'test'] but found {len(dataset_dict)} with names {list(dataset_dict.keys())}."
        )
    

    
    if augment_model_config is not None and augment_model_config["apply"]:
            dataset = augment_model(dataset, augment_model_config)

    if augment_path is not None:
        augment_data = load_from_disk(augment_path)
        sample_indices = np.random.choice(
            range(len(augment_data)), int(len(augment_data) * augment_fraction)
        )
        sampled_augment_data = augment_data.select(sample_indices)

        dataset_dict["train"] = concatenate_datasets(
            [dataset_dict["train"], sampled_augment_data]
        )
        dataset_dict["train"] = dataset_dict["train"].shuffle()

    relevant_columns = set()
    rename_columns = dict()

    extracted_config = dict()
    for modality in data_config.keys():
        if isinstance(data_config[modality]["column"], str):
            relevant_columns.add(data_config[modality]["column"])
            rename_columns[data_config[modality]["column"]] = modality

            extracted_config[data_config[modality]["column"]] = data_config[modality]
            extracted_config[data_config[modality]["column"]].pop("column")

        elif isinstance(data_config[modality]["column"], list):
            relevant_columns.update(data_config[modality]["column"])
            extracted_config[modality] = data_config[modality]

        else:
            raise ValueError(
                f"Expected column to be either list or str for modality: {modality}"
            )

    existing_columns = set(dataset_dict[list(dataset_dict.keys())[0]].column_names)
    columns_to_drop = existing_columns.difference(relevant_columns)

    processed_dataset_dict = DatasetDict()
    for dataset_key in dataset_dict.keys():
        selected_dataset = dataset_dict[dataset_key]
        processed_dataset = selected_dataset.remove_columns(columns_to_drop)
        processed_dataset = selected_dataset.rename_columns(rename_columns)

        processed_dataset_dict[dataset_key] = processed_dataset

    if augment_names is not None and not test_only:
        augment_names = augment_names.split("_")  # type: ignore
        augment_fn = partial(augment, augment_names=augment_names)
        augmented_train_set = processed_dataset_dict["train"].map(
            augment_fn,
            batched=True,
            batch_size=1,
            remove_columns=processed_dataset_dict["train"].column_names,
            num_proc=7,
        )
        processed_dataset_dict["train"] = augmented_train_set

    return data_config, processed_dataset_dict
