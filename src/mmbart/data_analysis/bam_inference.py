# mypy: ignore-errors
"""Set of functions to perform dataset metadata generation augmented with the language models in BAM. """

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from typing import Dict, Optional, Tuple

from datasets import DatasetDict
from genai.client import Client
from genai.schema import (
    TextGenerationParameters,
    TextGenerationReturnOptions,
)

from mmbart.data_analysis.utils import (
    clean_and_format_string,
    get_dataset_feature_names,
    get_metadata_string,
)


def bam_description_generation(
    dataset: DatasetDict,
    client: Optional[Client] = None,
    model_id: str = "",
    known_metadata: Dict[str, str] = {},
) -> str:
    """Call to BAM to generate a description for the dataset.

    Args:
        dataset: Hugging face dataset.
        client: Bam inference client opbject. Defaults to None.
        model_id: model name. Defaults to "".
        known_metadata: dictionary of known dataset information. Defaults to the empty dictionary.

    Returns:
        A string with the description for the dataset.
    """

    if not (client and model_id):
        description = "No description available. "
    else:
        description_gen_prompt, description_gen_params = get_description_prompt(
            dataset, known_metadata=known_metadata
        )
        request = client.text.generation.create(
            model_id=model_id,
            inputs=[description_gen_prompt],
            parameters=description_gen_params,
        )
        for response in request:
            result = response.results[0]
            description = get_description_from_response(result.generated_text)

    if "creation_date" in known_metadata:
        description += f"Created on {known_metadata['creation_date']}"

    return description


def get_description_prompt(
    dataset: DatasetDict, known_metadata: Dict[str, str] = {}
) -> Tuple[str, TextGenerationParameters]:
    """Generates the prompt for BAM to get the dataset description.

    Args:
        dataset: a HF dataset object.
        known_metadata: dictionary of known dataset information. Defaults to the empty dictionary.

    Returns:
        the prompt for the model and the parameters object.
    """

    feature_names = get_dataset_feature_names(dataset)
    metadata_string = get_metadata_string(known_metadata)
    prompt = f"Write a short description for the dataset {known_metadata['dataset_name']} with features named: {feature_names}. Consider the following additional information {metadata_string}. If possible from the labels, say what the data content is. Do not add information about the license."
    # logger.info(f"Promt: {prompt}")
    params = TextGenerationParameters(
        max_new_tokens=150,
        min_new_tokens=10,
        return_options=TextGenerationReturnOptions(
            input_text=False,
        ),
    )
    return prompt, params


def bam_name_generation(dataset: DatasetDict, client: Client = None, model_id: str = "") -> str:
    """Call to BAM to generate a descriptive name for the dataset.

    Args:
        dataset: Hugging face dataset.
        client: Bam inference client opbject. Defaults to None.
        model_id: model name. Defaults to "".

    Returns:
        A string representing the name suggested for the dataset.
    """
    if not (client and model_id):
        return "Unnamed dataset"
    else:
        name_gen_prompt, name_gen_params = get_name_prompt(dataset)
        request = client.text.generation.create(
            model_id=model_id,
            inputs=[name_gen_prompt],
            parameters=name_gen_params,
        )
        for response in request:
            result = response.results[0]
            dataset_name = get_name_from_response(result.generated_text)
            return dataset_name
    return "Unnamed dataset"

def get_name_prompt(dataset: DatasetDict) -> Tuple[str, TextGenerationParameters]:
    """Generates the prompt for BAM to get the dataset name.

    Args:
        dataset: a HF dataset object.

    Returns:
        the prompt for the model and the parameters object.
    """
    feature_names = get_dataset_feature_names(dataset)
    prompt = f"Propose a one word title for a dataset with features named: {feature_names}. Answer with a single word."

    params = TextGenerationParameters(
        max_new_tokens=5,
        min_new_tokens=1,
        return_options=TextGenerationReturnOptions(
            input_text=False,
        ),
    )
    return prompt, params


def get_name_from_response(model_reponse: str = ""):
    """Gets a model response for the inference of the dataset name and returns a two-words formatted name.

    Args:
        model_reponse: Model response formatted as a string. Defaults to "".

    Returns:
        Max two words string describing the dataset name.
    """
    if model_reponse == "":
        return "Unnamed dataset"
    words = clean_and_format_string(model_reponse).split(" ")
    words = words[:2]
    return "".join(words)


def get_description_from_response(model_reponse: str = ""):
    """Gets a model response for the inference of the dataset description.

    Args:
        model_reponse: Model response formatted as a string. Defaults to "".

    Returns:
        A string describing the dataset.
    """
    if model_reponse == "":
        return "No description available."
    return clean_and_format_string(model_reponse)
