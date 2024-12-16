# mypy: ignore-errors

import logging
import math
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

from multimodal.data.datamodules import MultiModalDataModule
from multimodal.defaults import DEFAULT_SEED

# Default model hyperparams
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 6
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_ACTIVATION = "gelu"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_DROPOUT = 0.1

DEFAULT_DEEPSPEED_CONFIG_PATH = "ds_config.json"
DEFAULT_LOG_DIR = "tb_logs"
DEFAULT_VOCAB_PATH = "bart_vocab.txt"
DEFAULT_CHEM_TOKEN_START = 272
REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

DEFAULT_GPUS = 1
DEFAULT_NUM_NODES = 1

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def calculate_training_steps(
    data_module: MultiModalDataModule, config: DictConfig
) -> int:
    batches_per_gpu = math.ceil(
        len(data_module.train_dataloader())
        / float(1)  # Number of gpus, for now hardcoded to 1
    )
    train_steps = (
        math.ceil(batches_per_gpu / config["trainer"]["acc_batches"])
        * config["trainer"]["epochs"]
    )

    return train_steps


class OptLRMonitor(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, trainer, *args, **kwargs):
        # Only support one optimizer
        opt = trainer.optimizers[0]

        # Only support one param group
        stats = {"lr-Adam": opt.param_groups[0]["lr"]}
        trainer.logger.log_metrics(stats, step=trainer.global_step)


def seed_everything(seed: Optional[int] = None) -> None:
    if seed is None:
        seed = DEFAULT_SEED

    pl.seed_everything(seed)


def print_results(args, results):
    print(f"Results for model: {args.model_path}")
    print(f"{'Item':<25}Result")
    for key, val in results.items():
        print(f"{key:<25} {val:.4f}")
