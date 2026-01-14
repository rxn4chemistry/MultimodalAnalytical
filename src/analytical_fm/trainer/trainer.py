from typing import Optional, Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, Callback
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
    save_checkpoints: str = "best_5",
    early_stopping_delta: Optional[float] = None,
    early_stopping_len_set_sel: Optional[bool] = False,
    update_dataloaders: Optional[bool] = False
) -> Trainer:
    logger = TensorBoardLogger(log_dir, name=task)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if model_type in [
        "BART",
        "BartForConditionalGeneration",
        "CustomBartForConditionalGeneration",
        "T5ForConditionalGeneration",
        "CustomModel"
    ]:
        if save_checkpoints == "best_5":
            checkpoint_callback = ModelCheckpoint(
                monitor=checkpoint_monitor, save_last=True, save_top_k=5, mode="max" if "loss" not in checkpoint_monitor else "min"
            )
        elif save_checkpoints == "every_5_epochs":
            checkpoint_callback = ModelCheckpoint(
                monitor=checkpoint_monitor, save_last=True, save_top_k=-1, every_n_epochs=5, mode="max" if "loss" not in checkpoint_monitor else "min"
            )
        elif save_checkpoints == "all":
            checkpoint_callback = ModelCheckpoint(
                monitor=checkpoint_monitor, save_last=True, save_top_k=-1, mode="max" if "loss" not in checkpoint_monitor else "min"
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
                monitor="val_token_acc",
                min_delta=early_stopping_delta,
                patience=early_stopping_patience,
                mode="max" if "loss" not in checkpoint_monitor else "min",
            )
        )

    if early_stopping_len_set_sel:
        callbacks.append(
            EarlyStopping(
                monitor="len_set_sel",
                min_delta=0.,
                patience=3,
                mode="max",
            )
        )

    strategy = "ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"

    trainer = Trainer(
        devices = -1 if torch.cuda.is_available() else 1,
        logger = logger,
        max_epochs = epochs,
        accumulate_grad_batches = acc_batches,
        gradient_clip_val = clip_grad,
        limit_val_batches = limit_val_batches,
        callbacks = callbacks,
        check_val_every_n_epoch = 1,
        precision = "16-mixed" if torch.cuda.is_available() else "32-true" ,
        strategy = strategy,
        val_check_interval=val_check_interval,
        deterministic=True,
        log_every_n_steps=1,
        reload_dataloaders_every_n_epochs=1 if update_dataloaders else 0
    )
    return trainer
