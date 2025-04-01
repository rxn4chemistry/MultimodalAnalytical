from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)
from omegaconf.dictconfig import DictConfig
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from analytical_fm.data.augmentations import augment
from analytical_fm.defaults import (
    DEFAULT_SEED,
    DEFAULT_VAL_SET_SIZE,
)

valid_modalities = ["text", "vector"]


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
        test_size=min(int(0.1 * len(train_set)), DEFAULT_VAL_SET_SIZE),
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
        test_size=min(int(0.05 * len(train_set)), DEFAULT_VAL_SET_SIZE),
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
    unique_targets = pd.unique(dataset[target_column])

    k_folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(k_folds.split(X=unique_targets))
    train_indices, test_indices = splits[cv_split][0], splits[cv_split][1]

    train_targets, test_targets = (
        unique_targets[train_indices],
        set(unique_targets[test_indices])
    )

    train_targets, val_targets = train_test_split(train_targets, test_size=min(int(0.05 * len(train_targets)), DEFAULT_VAL_SET_SIZE), random_state=seed, shuffle=True)
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
    augment_config: Optional[DictConfig] = None
) -> Tuple[Dict[str, Union[str, int, bool]], DatasetDict]:
    
    if not Path(data_path).is_dir():
        raise ValueError(
            "Data path must specify path to directory containing the dataset files as parqet."
        )

    dataset_dict = load_dataset("parquet", data_dir=data_path)

    # Concatenates all datasets into a single test set
    if splitting == "test_only":
        datasets = list(dataset_dict.values())
        combined_dataset = concatenate_datasets(datasets)
        dataset_dict = DatasetDict({"test": combined_dataset, "train": combined_dataset, "val": combined_dataset})

    # Split based on functinal group occurence.
    elif splitting == "func_group_split":
        dataset_dict = func_split(data_path, cv_split=cv_split, seed=DEFAULT_SEED)
    
    # Split based on unique values in the target column
    elif splitting == "unique_target":
        # Get Target column
        target_column = ""
        for modality_config in data_config.values():
            if modality_config["target"]:
                target_column = modality_config["column"]
                break
        
        # Combine dataset
        datasets = list(dataset_dict.values())
        combined_dataset = concatenate_datasets(datasets)

        # Split Dataset
        dataset_dict = target_split(combined_dataset, target_column, cv_split=cv_split, seed=DEFAULT_SEED)

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
    dataset_dict['train'] = augment(dataset_dict['train'], augment_config)

    # Rename columns and drop uncessecary
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

    return data_config, processed_dataset_dict
