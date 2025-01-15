from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def build_trainer(
    model_type: str,
    log_dir: str,
    task: str,
    epochs: int,
    acc_batches: int = 8,
    clip_grad: float = 1.0,
    limit_val_batches: float = 5.0,
) -> Trainer:
    logger = TensorBoardLogger(log_dir, name=task)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if model_type in [
        "BART",
        "BartForConditionalGeneration",
        "CustomBartForConditionalGeneration",
        "T5ForConditionalGeneration",
    ]:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_molecular_accuracy", save_last=True, save_top_k=5, mode="max"
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

    trainer = Trainer(
        logger=logger,
        min_epochs=epochs,
        max_epochs=epochs,
        accumulate_grad_batches=acc_batches,
        gradient_clip_val=clip_grad,
        limit_val_batches=limit_val_batches,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        precision="16-mixed",
    )
    return trainer
