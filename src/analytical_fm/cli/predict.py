"""Run the prediction pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from analytical_fm.configuration import DEFAULT_SETTINGS
from analytical_fm.data.data_utils import load_preprocessors
from analytical_fm.data.datamodules import MultiModalDataModule
from analytical_fm.data.datasets import (  # noqa: F401
    build_dataset_multimodal,
)
from analytical_fm.modeling.wrapper import HFWrapper
from analytical_fm.trainer.trainer import build_trainer
from analytical_fm.utils import calc_sampling_metrics, reject_sample, seed_everything

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@hydra.main(version_base=None, config_path=DEFAULT_SETTINGS.configs_path, config_name="config_predict")
def main(config: DictConfig):

    seed_everything()

    if config.model.model_checkpoint_path is None:
        raise ValueError(
            "Please supply model_checkpoint_path with config.model_checkpoint_path=..."
        )

    # Load dataset
    data_config = config["data"].copy()
    print(data_config)
    data_config = OmegaConf.to_container(data_config, resolve=True)
    model_config: Dict[str, Any] = OmegaConf.to_container(config["model"].copy(), resolve=True) # type: ignore


    data_config, dataset = build_dataset_multimodal(
        data_config,
        data_path=config["data_path"],
        cv_split=config["cv_split"],
        splitting=config["splitting"],
        augment_config=config["augment"],
        num_cpu=config["num_cpu"],
        mixture_config=config["mixture"],
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
        data_config_loaded, preprocessors = pd.read_pickle(preprocessor_path)
        data_config_model = data_config_loaded.copy()
        modalities_not_included = set(data_config_loaded.keys()).difference(data_config.keys())
        for modality in modalities_not_included:
            data_config_loaded.pop(modality)
        data_config_datamodule = data_config_loaded
        
    else:
        data_config, preprocessors = load_preprocessors(dataset["train"], data_config)
        with preprocessor_path.open("wb") as f:
            pickle.dump((data_config, preprocessors), f)
    logging.info("Build preprocessors")


    # Load datamodule
    model_type = config["model"]["model_type"]
    batch_size = config["model"]["batch_size"]
    predict_class = config["predict_class"]


    data_module = MultiModalDataModule(
        dataset=dataset,
        preprocessors=preprocessors,
        data_config=data_config_datamodule,
        model_type=model_type,
        batch_size=batch_size,
        num_workers=config["num_cpu"],
        extra_columns=[predict_class]
    )
    target_modality = data_module.collator.target_modality
    target_tokenizer = preprocessors[target_modality]
    logging.info("Build Datamodule")


    # Load Model
    checkpoint_path = config["model"]["model_checkpoint_path"]

    model = HFWrapper(
        data_config=data_config_model,
        target_tokenizer=target_tokenizer,
        num_steps=100,
        **model_config,
    )

    checkpoint_path = config["model"]["model_checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    #  Evaluate Model
    n_beams = config["model"]["n_beams"] if "n_beams" in config["model"] else 10
    rejection_sampling = config["model"]["rejection_sampling"] if "rejection_sampling" in config["model"] else False
    trainer = build_trainer(model_type, **config["trainer"])

    classes = None

    logger.info(f"Calculating metrics for class: {predict_class}")
    if predict_class and predict_class in data_config.keys():
        logger.info("Class is present in the dataset.")
        classes = []
        for batch in data_module.predict_dataloader():
            classes.extend(batch[predict_class])

        if isinstance(classes[0],list):
            classes = [cl[0] for cl in classes]

        logger.info(f"Classes: {set(classes)}")
        logger.info(f"Len of classes array: {len(classes)}")

    batch_predictions: List[Dict[str, Any]] = trainer.predict(model, datamodule=data_module) # type:ignore
    
    # Concatenate Predictions
    predictions = {'avg_loss': np.mean([batch['loss'] for batch in batch_predictions]),
                   'predictions': [batch['predictions'][i * n_beams : (i+1) * n_beams] for batch in batch_predictions for i in range(len(batch['predictions']) // n_beams)],
                   'targets': [target for batch in batch_predictions for target in batch['targets']]}
    
    if rejection_sampling:
        predictions = reject_sample(predictions, molecules=config['molecules'])

    
    metrics = calc_sampling_metrics(predictions['predictions'], predictions['targets'], classes=classes, molecules=config['molecules'], logging=True)
    
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

    metrics_path = (
        Path(config["working_dir"])
        / config["job_name"]
        / f"metrics_beam_{n_beams}.json"
    )
    with (metrics_path).open("w") as metrics_file:
        json.dump(metrics, metrics_file)
        
    logger.info(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
