"""Run the training pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
import contextlib
import json
import logging
import os
import pickle
import shutil
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, TextIO, cast

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
    calculate_training_steps,
    fail_safe_conditional_distributed_barrier,
    seed_everything,
)


@hydra.main(version_base=None, config_path=DEFAULT_SETTINGS.configs_path, config_name="config_train")
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
            seed_everything()

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
                mixture_config=config["mixture"]
            )
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

            if config["finetuning"] and 'MSMS' in preprocessors:
                preprocessors['MSMS'].max_sequence_length = model_config["max_position_embeddings"] - 100
                logging.info(f"Arbitrarily set max_len_seq to {preprocessors['MSMS'].max_sequence_length}")

            # Load datamodule
            model_type = model_config["model_type"]
            batch_size = model_config["batch_size"]
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
            # Load Model
            train_steps = calculate_training_steps(dataset['train'], config)
            model = HFWrapper(
                data_config=data_config,
                target_tokenizer=preprocessors[target_modality],
                num_steps=train_steps,
                modality_dropout = modality_dropout,
                **model_config,
            )

            #  Create Trainer
            trainer = build_trainer(model_type, **config["trainer"])

            # Train
            checkpoint_path = model_config["model_checkpoint_path"]
            if config["finetuning"]:
                checkpoint = torch.load(model_config["model_checkpoint_path"])

                keys_align = [k for k in checkpoint["state_dict"].keys() if "align_network" in k]

                if len(keys_align) != 0 and model_config["align_config"] is None:
                    for k in keys_align:
                        del checkpoint["state_dict"][k]

                model.load_state_dict(checkpoint["state_dict"])
                logger.info(f"Loaded checkpoint from {checkpoint_path}.")
                trainer.fit(model, datamodule=data_module)
            else:
                trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)


            # Load best model
            best_model_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            logger.info(f"Loading best Model from: {best_model_path}")

            shutil.copy(Path(best_model_path), f"{Path(best_model_path).parent}/best.ckpt")

            best_checkpoint = torch.load(best_model_path)

            model = HFWrapper(
                data_config=data_config,
                target_tokenizer=preprocessors[target_modality],
                num_steps=train_steps,
                modality_dropout = modality_dropout,
                **model_config,
            )

            model.load_state_dict(best_checkpoint["state_dict"])

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.eval()
            model.to(device)

            # Evaluate best model
            n_beams = config["model"]["n_beams"] if "n_beams" in config["model"] else 10
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
            predictions = {
                'avg_loss': np.mean([batch['loss'] for batch in batch_predictions]),
                'predictions': [batch['predictions'][i * n_beams : (i+1) * n_beams] for batch in batch_predictions for i in range(len(batch['predictions']) // n_beams)],
                'targets': [target for batch in batch_predictions for target in batch['targets']],
                'classes': [cl for batch in batch_predictions for cl in batch[predict_class]] if predict_class else None,
            }
            
            metrics = calc_sampling_metrics(predictions['predictions'], predictions['targets'], classes=predictions['classes'], molecules=config['molecules'], logging=True)

    
            rank = torch.distributed.get_rank() if torch.cuda.is_available() else 0
            save_path = (
                Path(config["working_dir"])
                / config["job_name"]
                / f"test_data_logits_beam_{n_beams}_{rank}.pkl"
            )
            with (save_path).open("wb") as save_file:
                pickle.dump(
                    predictions,
                    save_file,
                )

            #save metrics
            metrics_path = (
                Path(config["working_dir"])
                / config["job_name"]
                / f"metrics_beam_{n_beams}_{rank}.json"
            )
            with (metrics_path).open("w") as metrics_file:
                json.dump(metrics, metrics_file)
                
            logger.info(f"Metrics saved to: {metrics_path}")
            
        except Exception:
            logger.exception("Pipeline execution failed!")

if __name__ == "__main__":
    main()
