"""Runs a basic data-analysis pipeline augmented with large language models."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from pathlib import Path
from typing import Dict, Optional

import click
import pandas as pd
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from loguru import logger

from mmbart.data_analysis.bam_inference import bam_description_generation, bam_name_generation
from mmbart.data_analysis.configuration import DATA_ANALYSIS_SETTINGS
from mmbart.data_analysis.utils import (
    df_to_hf_dataset,
    exclude_unsupported_files,
    file_to_pandas_df,
    get_dataset_label_names,
    list_files,
    parse_metadata,
)

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>
load_dotenv()


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"


def get_name(hf_dataset: DatasetDict, dataset_name: str, client: Optional[Client] = None, model_id: str = ""):
    """Given a HF dataset, generates a name with inference to a Large Language Model.

    Args:
        hf_dataset: HF dataset dict .
        dataset_name: name of the dataset.
        client: inference api client. Defaults to None.
        model_id: model name. Defaults to "".

    Returns:
        Descriptive name for the dataset.
    """
    if not dataset_name:
        dataset_name = bam_name_generation(hf_dataset, client=client, model_id=model_id)
        logger.info(f"Inferred dataset name: {dataset_name}")
    return dataset_name


def get_description(
    hf_dataset, client: Optional[Client] = None, model_id: str = "", known_metadata: Dict[str, str] = {}
):
    """Given a HF dataset, generates a name with inference to a Large Language Model.

    Args:
        hf_dataset: HF dataset dict .
        dataset_name: name of the dataset.
        client: inference api client. Defaults to None.
        model_id: model name. Defaults to "".

    Returns:
        A generated description of the dataset.
    """
    dataset_description = bam_description_generation(
        hf_dataset, client=client, model_id=model_id, known_metadata=known_metadata
    )
    logger.info(f"Inferred dataset description: {dataset_description}")
    return dataset_description


@click.command()
@click.option(
    "--dataset_name",
    help="HF dataset name",
    required=False,
    type=str,
)
@click.option(
    "--config_name",
    help="HF dataset config name",
    required=False,
    type=str,
)
@click.option(
    "--data_folder",
    help="path to main folder with all the data files. Note: only CSVs are currently supported",
    required=False,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--output_dir",
    help="output directory for the report",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
def run_data_analysis(
    output_dir: Path,
    dataset_name: str = "",
    config_name: str = "",
    data_folder: Optional[Path] = None,
    known_metadata: Optional[Dict[str, str]] = {},
) -> None:
    """Runs a basic data analysis script that generates a dataset report with metadata.

    Args:
        dataset name: name of the HF dataset to analyze, if there is one. Defaults to an empty string
        data_folder: path to the folder where the data is located. Defaults to None.
    """

    dataset_report = {}

    if not data_folder and not dataset_name:
        logger.error("Add at least a dataset path or a HF dataset name")
        return

    if data_folder:
        files_list = list_files(data_folder)
        if len(files_list) < 1:
            logger.error("No files in folder")
            return
        filtered_file_list = exclude_unsupported_files(files_list, DATA_ANALYSIS_SETTINGS.supported_file_types)
        if len(filtered_file_list) < 1:
            logger.error("No supported files in folder")
            return

        data_frames = []
        for file_path in filtered_file_list[:]:
            data_frames.append(file_to_pandas_df(file_path))

        hf_dataset = df_to_hf_dataset(pd.concat(data_frames), metadata=known_metadata)

    else:
        if dataset_name.strip() == "":
            if "dataset_name" not in known_metadata:
                logger.error("Empty dataset name.")
                return
            else:
                dataset_name = known_metadata["dataset_name"]

        hf_dataset = load_dataset(dataset_name, config_name)

    dataset_metadata = hf_dataset["train"].info
    parsed_dataset_metadata = parse_metadata(dataset_metadata)
    label_names = get_dataset_label_names(hf_dataset)
    if label_names:
        known_metadata["label_names"] = label_names
    for key in parsed_dataset_metadata:
        known_metadata[key] = parsed_dataset_metadata[key]

    client = Client(credentials=Credentials.from_env())
    model_id = "mistralai/mixtral-8x7b-instruct-v01"

    dataset_report["name"] = get_name(hf_dataset, dataset_name, client=client, model_id=model_id)
    known_metadata["dataset_name"] = dataset_report["name"]

    generated_description = get_description(
        hf_dataset, client=client, model_id=model_id, known_metadata=known_metadata
    )
    dataset_report["description"] = (
        generated_description if not dataset_metadata.description else dataset_metadata.description
    )
    dataset_report["generated_description"] = generated_description

    # TODO: see list below
    # get date of creation
    # get dataset version
    # get dataset license
    # get dataset keywords
    # get dataset domain
    # get data type
    # get dataset file format stats
    # get dataset size
    # get number of features
    # get number of instances
    # get features description
    # get possible target variables
    # get missing values stats
    # get data collection method
    # get quality/ accuracy/ completeness

    dataset_report_df = pd.DataFrame.from_records([dataset_report], index=[0])

    output_file = Path.joinpath(output_dir, f"{dataset_report['name']}-report.csv")
    dataset_report_df.to_csv(output_file)

    return


if __name__ == "__main__":
    run_data_analysis()
