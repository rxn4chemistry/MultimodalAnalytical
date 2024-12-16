import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import Dataset
from rdkit import Chem
from transformers import AutoTokenizer

from mmbart.data.preprocessors import PREPROCESSORS, return_type
from mmbart.data.tokenizer import build_regex_tokenizer
from mmbart.defaults import DEFAULT_SAMPLES

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_preprocessors(
    data_set: Dataset,
    config: Dict[str, Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, return_type],
]:
    preprocessors = dict()

    selected_sample = np.random.randint(
        0, len(data_set), min(DEFAULT_SAMPLES, len(data_set))
    )
    sampled_dataset = data_set.select(selected_sample)
    for modality, modality_config in config.items():
        # Tokenizer only relevant for text

        if modality_config["type"] == "text":
            if "tokenizer_path" in modality_config["preprocessor_arguments"]:
                tokenizer = AutoTokenizer.from_pretrained(
                    modality_config["preprocessor_arguments"]
                )

            elif "tokenizer_regex" in modality_config["preprocessor_arguments"]:
                logger.info(f"Building Tokenizer from scratch for modality: {modality}")

                behaviour = (
                    "isolated"
                    if "tokenizer_behaviour"
                    not in modality_config["preprocessor_arguments"]
                    else modality_config["preprocessor_arguments"][
                        "tokenizer_behaviour"
                    ]
                )

                tokenizer = build_regex_tokenizer(
                    sampled_dataset[modality],
                    modality_config["preprocessor_arguments"]["tokenizer_regex"],
                    tokenizer_behaviour=behaviour,
                )

                logger.info(
                    f"Modality {modality} has Vocab Size: {tokenizer.vocab_size}"
                )

            else:
                raise ValueError(
                    "One of tokenizer_path or tokenizer_regex has to be defined for datatype text."
                )

            preprocessors[modality] = tokenizer

            # Add parameters for embedding
            config[modality]["vocab_size"] = tokenizer.vocab_size
            config[modality]["pad_token_id"] = tokenizer.pad_token_id

        elif modality_config["type"] in PREPROCESSORS:

            logger.info(f"Building {modality_config['type']} preprocessor")

            if modality_config["preprocessor_arguments"]:
                preprocessor = PREPROCESSORS[modality_config["type"]](
                    **modality_config["preprocessor_arguments"]
                )
            else:
                preprocessor = PREPROCESSORS[modality_config["type"]]()

            logger.info(f"Initialising {modality_config['type']} preprocessor")

            preprocessor.initialise(sampled_dataset, modality)

            preprocessors[modality] = preprocessor

            if hasattr(preprocessor, "tokenizer"):
                f"Modality {modality} has Vocab Size: {preprocessor.tokenizer.vocab_size}"
                config[modality]["vocab_size"] = preprocessor.tokenizer.vocab_size
                config[modality]["pad_token_id"] = preprocessor.tokenizer.pad_token_id
            elif hasattr(preprocessor, "n_features"):
                config[modality]["n_features"] = preprocessor.n_features

        elif modality_config["type"] == "no_action":
            logger.info(f"No action for modality {modality}.")
            sample = sampled_dataset.select([1])
            try:
                config[modality]["n_features"] = len(sample[modality][0])
            except TypeError:
                config[modality]["n_features"] = 1

        else:
            raise ValueError(
                f"Modality with type {modality_config['type']} is not implemented."
            )

    return config, preprocessors


def calculate_functional_group(
    smiles: str, functional_groups: Dict[str, Any]
) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)

    groups = np.zeros(len(functional_groups))

    for i, func_group in enumerate(functional_groups.values()):
        n = len(mol.GetSubstructMatches(func_group))
        groups[i] = 0 if n == 0 else 1

    return groups


def get_functional_groups(
    smiles_list: List[str], functional_groups: Dict[str, Any]
) -> np.ndarray:
    batch_func_groups = list()
    for smiles in smiles_list:
        func_groups = calculate_functional_group(smiles, functional_groups)
        batch_func_groups.append(func_groups)

    return np.vstack(batch_func_groups)
