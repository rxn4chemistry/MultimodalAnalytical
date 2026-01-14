"""Run the test-time tuning pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
import contextlib
import os
import pickle
import shutil
import sys
import warnings
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, TextIO, cast

import hydra  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
import torch_incremental_pca as tip  # type: ignore
from loguru import logger  # type: ignore
from omegaconf import DictConfig, OmegaConf  # type: ignore
from utils import StreamToLogger  # type: ignore

from analytical_fm.configuration import DEFAULT_SETTINGS
from analytical_fm.data.data_utils import load_preprocessors
from analytical_fm.data.datamodules import KmeansTTTMultiModalDataModule
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

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
            seed_everything(seed=config['seed'])

            # PARALLEL - Only preprocess on main thread
            fail_safe_conditional_distributed_barrier(
                lambda: torch.distributed.get_rank() > 0
            )

            # Load dataset
            data_config = config["data"].copy()
            logger.info(data_config)
            data_config = OmegaConf.to_container(data_config, resolve=True)
            model_config: Dict[str, Any] = OmegaConf.to_container(config["model"].copy(), resolve=True) # type: ignore
            activeft_config = config["activeft"]


            # Load parameters
            model_type = model_config["model_type"]
            batch_size = model_config["batch_size"]
            modality_dropout = config["modality_dropout"]
            predict_class = config["predict_class"]

            classes = None
            rejection_sampling = model_config["rejection_sampling"] if "rejection_sampling" in model_config else False
            if "n_beams" not in model_config or model_config['n_beams'] is None:
                model_config["n_beams"] = 50 if rejection_sampling else 10
            n_beams = model_config["n_beams"]

            # Build dataset
            data_config, dataset = build_dataset_multimodal(
                data_config, # type: ignore
                data_path=config["data_path"],
                cv_split=config["cv_split"],
                splitting=config["splitting"],
                augment_config=config["augment"],
                num_cpu=config["num_cpu"],
                mixture_config=config["mixture"],
            )
            dataset['train'] = dataset['train'].shuffle()
            dataset['validation'] = dataset['validation'].shuffle()
            # no shuffle of the testset to then compare predictions
            logger.info(f"Info dataset:\nTraining on {len(dataset['train'])}\nValidation on {len(dataset['validation'])}\nTesting on {len(dataset['test'])}\n")
            assert activeft_config["n_clusters"] <= len(dataset["test"]), "Number of clusters cannot be larger than the number of test points!"

            # Load/build tokenizers and preprocessors
            if config["preprocessor_path"] is None:
                preprocessor_path = (
                    Path(config["working_dir"]) / config["job_name"] / "preprocessor.pkl"
                )
            else:
                preprocessor_path = Path(config["preprocessor_path"])

            if preprocessor_path.is_file():
                logger.info(f"Loading existing preprocessor from: {str(preprocessor_path)}")
                data_config, preprocessors = pd.read_pickle(preprocessor_path)
            else:
                logger.info(f"No existing preprocessor found at: {str(preprocessor_path)}")
                data_config, preprocessors = load_preprocessors(dataset["train"], data_config)
                with preprocessor_path.open("wb") as f:
                    pickle.dump((data_config, preprocessors), f)
            logger.info("Built preprocessors")

            target_modality = 'Smiles'
            if 'MSMS' in preprocessors.keys() and preprocessors['MSMS'].max_sequence_length > 924:
                preprocessors['MSMS'].max_sequence_length = model_config["max_position_embeddings"] - 100
                logger.info(f"Changed max_len_seq to {preprocessors['MSMS'].max_sequence_length}")

            # only for activeft
            model_config['lr_gamma'] = 0.995

            # Load Model
            model = HFWrapper(
                data_config=data_config,
                target_tokenizer=preprocessors[target_modality],
                num_steps=int(config["trainer"]["epochs"]),
                modality_dropout=modality_dropout,
                **model_config,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            # Load model from checkpoint
            if model_config["model_checkpoint_path"]:
                checkpoint = torch.load(model_config["model_checkpoint_path"], map_location=device, weights_only=True)
                keys_align = [k for k in checkpoint["state_dict"].keys() if "align_network" in k]
                if len(keys_align) != 0 and model_config["align_config"] is None:
                    for k in keys_align:
                        del checkpoint["state_dict"][k]
                model.load_state_dict(checkpoint["state_dict"])
                logger.info(f'Loaded checkpoint from {model_config["model_checkpoint_path"]}')
            # model_initial = model.state_dict()


            # Path to save selection
            path_selection = f'{config["working_dir"]}/{config["job_name"]}/selection'
            if not Path.exists(Path(path_selection)):
                Path.mkdir(Path(path_selection))
            
            # adjust trainer settings for activeft
            config["trainer"]["acc_batches"] = 1
            config["trainer"]["update_dataloaders"] = True
            config["trainer"]["epochs"] = activeft_config["n_clusters"]*activeft_config["repeat_training"]
            config["trainer"]["early_stopping_patience"] = config["trainer"]["epochs"] + 1 # to make sure the training doesn't stop before all the clusters have been used

            # Initialization of the trainer
            trainer = build_trainer(model_type, **config["trainer"])
            logger.info('Trainer built')

            # PARALLEL - Lift barrier for main thread data loading/preprocessing is finished
            fail_safe_conditional_distributed_barrier(
                lambda: torch.distributed.get_rank() == 0 and torch.cuda.is_available()
            )

            # Build data module
            data_module = KmeansTTTMultiModalDataModule(
                dataset=dataset,
                model=model,
                preprocessors=preprocessors,
                data_config=data_config,
                model_type=model_type,
                batch_size=batch_size,
                num_workers=config["num_cpu"],
                extra_columns=[predict_class],
                device=device,
                reduced_val=activeft_config["reduced_val"],
                only_faiss=activeft_config["only_faiss"],
                path_selection=path_selection,
                similarity_criterion=activeft_config["similarity_criterion"],
                nearest_neighbors=activeft_config["nearest_neighbors"],
                n_clusters=activeft_config["n_clusters"],
                n_test_points=activeft_config["n_test_points"],
                n_train_points=activeft_config["n_train_points"],
                update_embeds=activeft_config["update_embeddings"],
            )

            # Training
            model.train()
            model.to(device)
            trainer.fit(model, datamodule=data_module)

            # PARALLEL - Lift barrier for all threads when training is finished
            fail_safe_conditional_distributed_barrier(
                lambda: torch.distributed.get_rank() >= 0 and torch.cuda.is_available()
            )

            # Evaluate model here on all testset
            if trainer.checkpoint_callback is not None:
                if config["eval_ckpt"] == 'best':
                    best_model_path = trainer.checkpoint_callback.best_model_path # type: ignore
                    logger.info(f"Loading best Model from: {best_model_path}")
                    shutil.copy(Path(best_model_path), f"{Path(best_model_path).parent}/best.ckpt")
                    best_checkpoint = torch.load(best_model_path)
                    model.load_state_dict(best_checkpoint["state_dict"])
                elif config["eval_ckpt"] == 'last':
                    last_model_path = trainer.checkpoint_callback.last_model_path # type: ignore
                    logger.info(f"Using last ckpt: {last_model_path}")
                    last_checkpoint = torch.load(last_model_path)
                    model.load_state_dict(last_checkpoint["state_dict"])
                else:
                    raise ValueError("Unknown evaluation checkpoint method.")

                model.eval()
                model.to(device)

                predictions = evaluate(predict_class, data_config, config, data_module, trainer, model, n_beams) # data_module_test contains all the testset

                # Rejection sampling
                if rejection_sampling:
                    predictions = reject_sample(predictions, molecules=config['molecules'])

                predictions_path = save_to_files(predictions=predictions, metrics=None, config=config, n_beams=n_beams, name_file=f"after_training-")
                logger.info(f"Predictions saved to: {predictions_path}")

                metrics = calc_sampling_metrics(predictions['predictions'], predictions['targets'], classes=None, molecules=config['molecules'], logging=True)                
                metrics_path = save_to_files(predictions=None, metrics=metrics, config=config, n_beams=n_beams, name_file=f"after_training-")
                logger.info(f"Metrics saved to: {metrics_path}")

            else:
                logger.warning("No checkpoints saved in the trainer.")
            
        except Exception:
            logger.exception("Pipeline execution failed!")

if __name__ == "__main__":
    main()
