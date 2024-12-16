"""Data utils for automating exploratory dataset analysis"""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset

from mmbart.data_analysis.configuration import DATA_ANALYSIS_SETTINGS


def list_files(data_path: Path = Path()) -> List[Path]:
    """Lists all the files in the data_path folder.

    Args:
        data_path: string of the path. Defaults to "".

    Returns:
        A list of paths.
    """
    sub_files = []

    if not data_path.is_dir():
        return [data_path]

    for folder_path in data_path.iterdir():
        sub_files += list_files(folder_path)

    return sub_files


def exclude_unsupported_files(files_list: List[Path], supported_types: List[str]) -> List[Path]:
    """Removes files that are not in the supported extensions from a list of paths.

    Args:
        files_list: list of paths.
        supported_types: list of supported extensions.

    Returns:
        The filtered list of paths.
    """
    for file_path in files_list:
        if file_path.suffix not in supported_types:
            files_list.remove(file_path)
    return files_list


def file_to_pandas_df(csv_file: Path) -> pd.DataFrame:
    """Converts the csv file to a pandas dataframe from the csv path.

    Args:
        csv_file: path to csv file.

    Returns:
        pandas dataframe.
    """
    try:
        return pd.read_csv(csv_file)
    except Exception:
        return pd.DataFrame()


def df_to_hf_dataset(pd_dataframe: pd.DataFrame, metadata: Optional[Dict[str, str]] = {}) -> DatasetDict:
    """Takes a pandas dataframe and a dictionary of metadata and creates a HF dataset.

    Args:
        pd_dataframe: pandas dataframe.
        metadata: dataset metadata. Defaults to "".

    Returns:
        HF dataset dict containing the data in the pandas df.
    """
    hf_dataset = Dataset.from_pandas(pd_dataframe)
    dataset_dict = DatasetDict({"train": hf_dataset})
    if metadata:
        for field_name in DATA_ANALYSIS_SETTINGS.metadata_fields:
            if field_name in metadata:
                setattr(dataset_dict["train"].info, field_name, metadata[field_name])
    return dataset_dict


def file_to_hf_dataset(path_to_file: Path) -> DatasetDict:
    """Load data in path_to_file.

    Args:
        path_to_file: path to the file to load.

    Returns:
        HF dataset dict containing the data in the file.
    """

    return load_dataset(path_to_file.suffix.strip("."), data_files=str(path_to_file))


def get_dataset_feature_names(dataset: DatasetDict) -> str:
    """Returns a string of all the feature names of the dataset.

    Args:
        dataset: a HF dataset object.

    Returns:
        the feature names formatted as a string for interence to a language model.
    """
    features_list = map(
        str,
        [
            feature_name
            for feature_name in dataset["train"].features.keys()
            if not feature_name.startswith("__")
        ],
    )
    delimiter = ", "
    string_of_features_list = delimiter.join(features_list)
    return string_of_features_list


def get_dataset_label_names(dataset: DatasetDict) -> str:
    """Gets a string of labels of the datast.

    Args:
        dataset: HF dataset.

    Returns:
        list of dataset labels as a string.
    """
    if "label" in dataset["train"].features.keys():
        labels_list = dataset["train"].features["label"].names
        delimiter = ", "
        string_of_labels = delimiter.join(labels_list)
        return string_of_labels.strip()
    return ""


def parse_metadata(hf_metadata: DatasetInfo) -> Dict[str, str]:
    """Parses HF metadata into a dictionary of strings.

    Args:
        hf_metadata: HF dataset.

    Returns:
        Parsed metadata.
    """
    dataset_metadata_to_return = {}
    metadata_fields = [
        field
        for field in dir(hf_metadata)
        if not field.startswith("__") and not callable(getattr(hf_metadata, field))
    ]
    for field in metadata_fields:
        if type(hf_metadata.__getattribute__(str(field))) is str:
            dataset_metadata_to_return[str(field)] = hf_metadata.__getattribute__(str(field))
    return dataset_metadata_to_return


def clean_and_format_string(input_string: str) -> str:
    """Cleans a string from single spaces and special characters.

    Args:
        input_string: _description_.

    Returns:
        _description_.
    """
    cleaned_string = input_string.strip()
    cleaned_string = re.sub(r"\s+", " ", cleaned_string)
    cleaned_string = re.sub(r"[^\w\s]", "", cleaned_string)

    return cleaned_string


def get_metadata_string(known_metadata: Dict[str, str] = {}) -> str:
    """Converts the metadata into a string.

    Args:
        known_metadata: datast metadata. Defaults to {}.

    Returns:
        Converted metadata into a string.
    """
    key_delimiter = ": "
    item_delimiter = ", "
    string = ""
    for key in known_metadata:
        string += key.join(key_delimiter)
        string += known_metadata[key]
        string += item_delimiter
    return string
