"""Run the prediction pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
import contextlib
import logging
import os
import pickle
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, TextIO, cast

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from utils import StreamToLogger  # type: ignore

from analytical_fm.configuration import DEFAULT_SETTINGS
from analytical_fm.data.data_utils import load_preprocessors
from analytical_fm.data.datamodules import MultiModalDataModule
from analytical_fm.data.datasets import (  # noqa: F401
    build_dataset_multimodal,
)
from analytical_fm.modeling.wrapper import HFWrapper
from analytical_fm.trainer.trainer import build_trainer
from analytical_fm.utils import (
    calc_sampling_metrics,
    evaluate,
    fail_safe_conditional_distributed_barrier,
    reject_sample,
    save_to_files,
    seed_everything,
)


@hydra.main(version_base=None, config_path=DEFAULT_SETTINGS.configs_path, config_name="config_predict")
def main(config: DictConfig):

    if not torch.distributed.is_initialized():
        try:
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                timeout=timedelta(
                    minutes=float(
                        os.getenv("TORCH_PROCESS_GROUP_TIMEOUT_IN_MINUTES", 30)
                    )
                ),
            )
            logger.info("Process group has been initialized successfully")
        except ValueError:
            logger.warning(
                "Initializing the process group from the environment was not possible!"
            )

    logger.remove()
    logger.add(cast(TextIO, sys.__stderr__), enqueue=True)
    
    logger.add(
        Path(config["working_dir"]) / config["job_name"] / "loguru-training.log",
        enqueue=True,  # NOTE: added to ensure log file is written withd no lock during distributed training
    )

    stream = StreamToLogger(level="INFO")

    with contextlib.redirect_stderr(stream):  # type:ignore
        try:
            if config["seed"]:
                seed_everything(seed=config["seed"])
            else:
                seed_everything()
    

            if config.model.model_checkpoint_path is None:
                raise ValueError(
                    "Please supply model_checkpoint_path with config.model_checkpoint_path=..."
                )

            # Load dataset
            data_config = config["data"].copy()
            logger.info(data_config)
            data_config = OmegaConf.to_container(data_config, resolve=True)
            model_config: Dict[str, Any] = OmegaConf.to_container(config["model"].copy(), resolve=True) # type: ignore

            # Only preprocess on main thread
            fail_safe_conditional_distributed_barrier(
                lambda: torch.distributed.get_rank() > 0
            )

            data_config, dataset = build_dataset_multimodal(
                data_config, # type: ignore
                data_path=config["data_path"],
                cv_split=config["cv_split"],
                splitting=config["splitting"],
                augment_config=config["augment"],
                num_cpu=config["num_cpu"],
                mixture_config=config["mixture"],
            )
            # dataset = dataset.shuffle()
            if config['eval_points_test']:
                dataset['test'] = dataset['test'].select(np.arange(config['eval_points_test']))
            logging.info("Built dataset")

            # Load/build tokenizers and preprocessors
            if config["preprocessor_path"] is None:
                preprocessor_path = (
                    Path(config["working_dir"]) / config["job_name"] / "preprocessor.pkl"
                )
            else:
                preprocessor_path = Path(config["preprocessor_path"])

            if preprocessor_path.is_file():
                logging.info(f"Loading existing preprocessor from: {str(preprocessor_path)}")
                data_config, preprocessors = pd.read_pickle(preprocessor_path)
            else:
                logging.info(f"No existing preprocessor found at: {str(preprocessor_path)}")
                data_config, preprocessors = load_preprocessors(dataset["train"], data_config)
                with preprocessor_path.open("wb") as f:
                    pickle.dump((data_config, preprocessors), f)
            logging.info("Built preprocessors")

            # Load datamodule
            model_type = config["model"]["model_type"]
            batch_size = config["model"]["batch_size"]
            modality_dropout = config["modality_dropout"]
            predict_class = config["predict_class"]

            data_module = MultiModalDataModule(
                dataset=dataset,
                preprocessors=preprocessors,
                data_config=data_config,
                model_type=model_type,
                batch_size=batch_size,
                num_workers=config["num_cpu"],
                extra_columns=[predict_class]
            )
            target_modality = data_module.collator.target_modality
            logging.info("Built Datamodule")

            # Lift barrier data loading/preprocessing is finished
            fail_safe_conditional_distributed_barrier(
                lambda: torch.distributed.get_rank() == 0 and torch.cuda.is_available()
            )

            classes = None
            rejection_sampling = model_config["rejection_sampling"] if "rejection_sampling" in model_config else False
            if not model_config["n_beams"]:
                model_config["n_beams"] = 50 if rejection_sampling else 10
            n_beams = model_config["n_beams"]

            # Load Model
            model = HFWrapper(
                data_config=data_config,
                target_tokenizer=preprocessors[target_modality],
                num_steps=100,
                modality_dropout=modality_dropout,
                **model_config,
            )

            checkpoint_path = config["model"]["model_checkpoint_path"]
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["state_dict"])

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.eval()
            model.to(device)

            trainer = build_trainer(model_type, **config["trainer"])

            #  Evaluate Model
            predictions = evaluate(predict_class, data_config, config, data_module, trainer, model, n_beams)

            # Rejection sampling
            if rejection_sampling:
                predictions = reject_sample(predictions, molecules=config['molecules'])

            predictions_path = save_to_files(predictions=predictions, metrics=None, config=config, n_beams=n_beams, name_file=f"after_training-")
            logger.info(f"Predictions saved to: {predictions_path}")

            metrics = calc_sampling_metrics(predictions['predictions'], predictions['targets'], classes=predictions['classes'], molecules=config['molecules'], logging=True)                
            metrics_path = save_to_files(predictions=None, metrics=metrics, config=config, n_beams=n_beams, name_file=f"after_training-")    
            logger.info(f"Metrics saved to: {metrics_path}")

        except Exception:
            logger.exception("Pipeline execution failed!")

if __name__ == "__main__":
    main()
