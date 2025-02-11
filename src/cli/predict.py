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
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from mmbart.data.data_utils import load_preprocessors
from mmbart.data.datamodules import MultiModalDataModule
from mmbart.data.datasets import (  # noqa: F401
    build_dataset_multimodal,
)
from mmbart.modeling.wrapper import HFWrapper
from mmbart.trainer.trainer import build_trainer
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

    #  Evaluate Model
    n_beams = 10
    trainer = build_trainer(model_type, **config["trainer"])
    batch_predictions: List[Dict[str, Any]] = trainer.predict(model, datamodule=data_module) # type:ignore
    
    # Concatenate Predictions
    predictions = {'avg_loss': np.mean([batch['loss'] for batch in batch_predictions]),
                   'predictions': [batch['predictions'][i * n_beams : (i+1) * n_beams] for batch in batch_predictions for i in range(len(batch['predictions']) // n_beams)],
                   'targets': [target for batch in batch_predictions for target in batch['targets']]}
    
    calc_sampling_metrics(predictions['predictions'], predictions['targets'], molecules=config['molecules'], logging=True)
    
    save_path = (
        Path(config["working_dir"])
        / config["job_name"]
        / f"test_data_logits_beam_{n_beams}.pkl"
    )
    with (save_path).open("wb") as save_file:
        pickle.dump(
            predictions,
            save_file,
        )

if __name__ == "__main__":
    main()
