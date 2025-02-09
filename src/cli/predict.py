"""Run the prediction pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import logging
import pickle
from pathlib import Path

import hydra
import pandas as pd
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

from mmbart.data.data_utils import load_preprocessors
from mmbart.data.datamodules import MultiModalDataModule
from mmbart.data.datasets import (  # noqa: F401
    build_dataset_multimodal,
)
from mmbart.modeling.wrapper import HFWrapper
from mmbart.utils import calc_sampling_metrics, calculate_training_steps, seed_everything

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@hydra.main(version_base=None, config_path="../configs", config_name="config_predict")
def main(config: DictConfig):

    seed_everything()
    logger.info(config)

    if config.model.model_checkpoint_path is None:
        raise ValueError(
            "Please supply model_checkpoint_path with config.model_checkpoint_path=..."
        )


    # Load dataset
    data_config = config["data"].copy()
    print(data_config)
    data_config = OmegaConf.to_container(data_config, resolve=True)

    data_config, dataset = build_dataset_multimodal(
        data_config,
        data_path=config["data_path"],
        cv_split=config["cv_split"],
        func_group_split=config["func_group_split"],
        smiles_split=config["smiles_split"],
        augment_path=config["augment_path"],
        augment_fraction=config["augment_fraction"],
        augment_names=config["augment_names"],
        augment_model_config=config["augment_model"],
    )
    logging.info("Build dataset")


    # Load/build tokenizers and preprocessors
    if config["preprocessor_path"] is None:
        preprocessor_path = (
            Path(config["working_dir"]) / config["job_name"] / "preprocessor.pkl"
        )
    else:
        preprocessor_path = Path(config["preprocessor_path"])

    if preprocessor_path.is_file():
        data_config, preprocessors = pd.read_pickle(preprocessor_path)
    else:
        data_config, preprocessors = load_preprocessors(dataset["train"], data_config)
        with preprocessor_path.open("wb") as f:
            pickle.dump((data_config, preprocessors), f)
    logging.info("Build preprocessors")


    # Load datamodule
    model_type = config["model"]["model_type"]
    batch_size = config["model"]["batch_size"]
    mixture = config["mixture"]

    data_module = MultiModalDataModule(
        dataset=dataset,
        preprocessors=preprocessors,
        data_config=data_config,
        model_type=model_type,
        batch_size=batch_size,
        mixture=mixture
    )
    target_modality = data_module.collator.target_modality
    target_tokenizer = preprocessors[target_modality]
    logging.info("Build Datamodule")


    # Load Model
    train_steps = calculate_training_steps(data_module, config)
    checkpoint_path = config["model"]["model_checkpoint_path"]

    model = HFWrapper(
        data_config=data_config,
        target_tokenizer=target_tokenizer,
        num_steps=train_steps,
        **config["model"],
    )

    checkpoint_path = config["model"]["model_checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    # Evaluate model
    test_loader = data_module.test_dataloader()

    predictions = list()
    ground_truth = list()

    n_beams =10
    decode_method = "custom"

    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        batch["encoder_input"] = {
            modality: modality_input.to(device)
            for modality, modality_input in batch["encoder_input"].items()
        }
        batch["decoder_input"] = {
            modality: modality_input.to(device)
            for modality, modality_input in batch["decoder_input"].items()
        }

        for key in ["encoder_pad_mask", "decoder_pad_mask", "target_mask", "target"]:
            batch[key] = batch[key].to(device)

        
        if decode_method == "custom":

            with torch.no_grad():
                predictions_batch, log_lhs_batch = model.sample_molecules(
                    batch, n_beams=n_beams, sampling_alg="beam"
                )

            if isinstance(predictions_batch, torch.Tensor):
                predictions_batch = predictions_batch.cpu().tolist()
            if isinstance(log_lhs_batch, torch.Tensor):
                log_lhs_batch = log_lhs_batch.cpu().tolist()

            if "target_smiles" in batch:
                if isinstance(batch["target_smiles"], torch.Tensor):
                    ground_truth.extend(batch["target_smiles"].cpu().tolist())
                else:
                    ground_truth.extend(batch["target_smiles"])
            else:
                if isinstance(batch["target"], torch.Tensor):
                    ground_truth.extend(batch["target"].cpu().tolist())
                else:
                    ground_truth.extend(batch["target"])

            predictions.extend(predictions_batch)
        
        else:

            generated_sequences = model.generate(batch, n_beams=n_beams)

            detokenized_sequences = preprocessors[target_modality].batch_decode(
                generated_sequences, skip_special_tokens=True
            )
            detokenized_sequences = [
                detokenized_sequences[j * n_beams : (j + 1) * n_beams]
                for j in range(len(detokenized_sequences) // n_beams)
            ]

            predictions.extend(detokenized_sequences)
            ground_truth.extend(batch["target_smiles"])

    metrics = calc_sampling_metrics(predictions, ground_truth)
    logger.info(metrics)
    
    save_path = (
        Path(config["working_dir"])
        / config["job_name"]
        / f"test_data_logits_beam_{n_beams}.pkl"
    )
    with (save_path).open("wb") as save_file:
        pickle.dump(
            {"predictions": predictions, "ground_truth": ground_truth},
            save_file,
        )  # To do: Move away from pickle

if __name__ == "__main__":
    main()
