from typing import Optional, Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def build_trainer(
    model_type: str,
    log_dir: str,
    task: str,
    epochs: int,
    acc_batches: int = 8,
    clip_grad: float = 1.0,
    limit_val_batches: float = 5.0,
    checkpoint_monitor: str = "val_molecular_accuracy",
    val_check_interval: Optional[Union[int, float]] = None,
    early_stopping_patience: Optional[int] = None,
) -> Trainer:
    logger = TensorBoardLogger(log_dir, name=task)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if model_type in [
        "BART",
        "BartForConditionalGeneration",
        "CustomBartForConditionalGeneration",
        "T5ForConditionalGeneration",
        "CustomModel",
    ]:
        checkpoint_callback = ModelCheckpoint(
            monitor=checkpoint_monitor,
            save_last=True,
            save_top_k=5,
            mode="max" if "loss" not in checkpoint_monitor else "min",
        )
        checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "_"
    elif model_type == "encoder":
        if "weather" in task:
            mode = "min"
        else:
            mode = "max"
        print(mode)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_f1_score", save_last=True, save_top_k=5, mode=mode
        )

    callbacks = [lr_monitor, checkpoint_callback]

    if early_stopping_patience:
        callbacks.append(
            EarlyStopping(
                monitor=checkpoint_monitor,
                patience=early_stopping_patience,
                mode="max" if "loss" not in checkpoint_monitor else "min",
            )
        )
    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"

    trainer = Trainer(
        devices=-1 if torch.cuda.is_available() else 1,
        logger=logger,
        max_epochs=epochs,
        accumulate_grad_batches=acc_batches,
        gradient_clip_val=clip_grad,
        limit_val_batches=limit_val_batches,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        strategy=strategy,
        val_check_interval=val_check_interval,
    )
    return trainer
