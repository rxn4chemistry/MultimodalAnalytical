"""Training pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import logging
import os
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import yaml
from datasets import DatasetDict
from pydantic import BaseModel, Field
from transformers import Trainer, TrainingArguments

from ..data.collators.core import DataCollatorLoader
from ..data.loaders import LOADER_REGISTRY
from ..data.preprocessing import PREPROCESSOR_REGISTRY
from ..models.core import ModelLoader
from ..models.tokenizers.core import TokenizerLoader
from .utils import seed_everything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.cuda.is_bf16_supported()


class TrainPipelinePhase(str, Enum):
    """Training pipeline phases."""

    load_data = "load_data"
    process_data = "process_data"
    trainer = "trainer"
    modeling = "modeling"


class TrainPipelineConfigurationMetadata(BaseModel):
    """TrainPipeline configuration metadata."""

    task_name: str
    seed: int = 42


class GenericTrainPipelinePhaseArguments(BaseModel):
    """Generic pipeline phase arguments."""

    phase_name: str = Field(
        ...,
        description="phase name, should match with one of the registries.",
    )
    phase_arguments: Dict[str, Any]


class ModelingTrainPipelinePhaseArguments(BaseModel):
    """Modeling pipeline phase arguments."""

    tokenizer: Dict[str, Any]
    collator: Dict[str, Any]
    model: Dict[str, Any]


class TrainerTrainPipelinePhaseArguments(BaseModel):
    """Training pipeline phase arguments."""

    args: Dict[str, Any]
    train_args: Dict[str, Any]
    initialization_args: Dict[str, Any] = Field(..., default_factory=dict)
    save_model_args: Dict[str, Any] = Field(..., default_factory=dict)


class TrainPipelinePhases(BaseModel):
    """TrainPipeline phases object."""

    load_data: GenericTrainPipelinePhaseArguments
    process_data: List[GenericTrainPipelinePhaseArguments]
    trainer: TrainerTrainPipelinePhaseArguments
    modeling: ModelingTrainPipelinePhaseArguments


class TrainPipelineConfiguration(BaseModel):
    """TrainPipeline configuration."""

    metadata: TrainPipelineConfigurationMetadata
    pipeline_phases: TrainPipelinePhases


class TrainPipelineDataProcessingStep:
    """Processing pipeline step definition."""

    def __init__(
        self, pipeline_phase_arguments: GenericTrainPipelinePhaseArguments, **kwargs: Any
    ) -> None:
        """Initialize the pipeline steps.

        Args:
            pipeline_phase_arguments: pipeline phase arguments.
        """
        self.logic_to_be_applied_on_dataset_with_map = PREPROCESSOR_REGISTRY[
            pipeline_phase_arguments.phase_name
        ](**pipeline_phase_arguments.phase_arguments)
        self.additional_arguments = kwargs
        if "load_from_cache_file" not in self.additional_arguments:
            self.additional_arguments["load_from_cache_file"] = True

    def __call__(self, in_process_dataset: DatasetDict) -> DatasetDict:
        """Apply the step on a dataset.

        Args:
            in_process_dataset: dataset to process.

        Returns:
            processed dataset.
        """
        return in_process_dataset.map(
            function=self.logic_to_be_applied_on_dataset_with_map.preprocess,
            **self.additional_arguments,
        )


class TrainPipeline:
    """Train pipeline based on an input configuration."""

    def __init__(self, pipeline_configuration_path: Path) -> None:
        """

        Args:
            pipeline_configuration_path: path to training pipeline configuration.
        """
        with pipeline_configuration_path.open("rt") as fp:
            self.train_pipeline_configuration = TrainPipelineConfiguration(**yaml.full_load(fp))
        seed_everything(self.train_pipeline_configuration.metadata.seed)

        logger.info("Train pipeline initialized")

    @staticmethod
    def fail_safe_conditional_distributed_barrier(condition_fn: Callable[[], bool]) -> None:
        """Apply a distributed barrier in a fail-safe way.

        Args:
            condition_fn: callable to define condition for the barrier.
        """
        try:
            if condition_fn():
                logger.info("Distributed barrier applied")
                torch.distributed.barrier()
        except ValueError:
            # NOTE: catching errors due to uninitialized distributed process group.
            # Never active when running without torchrun. In this case a barrier is never needed.
            logger.info("No distributed barrier applied")

    def run_training_pipeline(
        self,
    ) -> None:
        """Run training pipeline."""
        logger.info("Training pipeline starting...")

        # NOTE: patch to disable MPS when running on silicon
        if not IS_CUDA_AVAILABLE:
            torch.backends.mps.is_available = lambda: False  # type:ignore

        if not torch.distributed.is_initialized():
            try:
                torch.distributed.init_process_group(
                    backend="nccl" if IS_CUDA_AVAILABLE else "gloo",
                    timeout=timedelta(
                        minutes=float(os.getenv("TORCH_PROCESS_GROUP_TIMEOUT_IN_MINUTES", 30))
                    ),
                )
                logger.info("Process group has been initialized successfully")
            except ValueError:
                logger.warning(
                    "Initializing the process group from the environment was not possible!"
                )

        # tokenizer
        logger.info("Tokenizer loading...")
        tokenizer_loader = TokenizerLoader(
            **self.train_pipeline_configuration.pipeline_phases.modeling.tokenizer
        )
        tokenizer = tokenizer_loader.load_tokenizer()
        logger.info("Tokenizer loaded")

        # data collator
        logger.info("Collator loading...")
        data_collator_loader = DataCollatorLoader(
            tokenizer=tokenizer,
            **self.train_pipeline_configuration.pipeline_phases.modeling.collator,
        )
        data_collator = data_collator_loader.load_collator()
        logger.info("Collator loaded")

        # model
        logger.info("Model loading...")
        model_loader = ModelLoader(
            **self.train_pipeline_configuration.pipeline_phases.modeling.model
        )
        model = model_loader.load_model(tokenizer=tokenizer)
        logger.info("Model loaded")

        # load data splitted in train, val and test
        logger.info("Data loading...")
        pipeline_phase_arguments = self.train_pipeline_configuration.pipeline_phases.load_data
        logger.info(f"Phase arguments are: {pipeline_phase_arguments.model_dump_json()}")
        data_loader = LOADER_REGISTRY[pipeline_phase_arguments.phase_name](
            **pipeline_phase_arguments.phase_arguments
        )
        dataset = data_loader.load_splits()
        logger.info("Data loaded")

        # NOTE: we perform the data preprocessing only on the main process
        TrainPipeline.fail_safe_conditional_distributed_barrier(
            lambda: torch.distributed.get_rank() > 0
        )

        # data preprocessing
        logger.info("Data preprocessing...")
        pipeline_phase_arguments_list = (
            self.train_pipeline_configuration.pipeline_phases.process_data
        )
        for pipeline_phase_arguments in pipeline_phase_arguments_list:
            logger.info(f"Phase arguments are: {pipeline_phase_arguments.model_dump_json()}")
            # NOTE: if is a tokenizer preprocessor we pass it as argument
            if pipeline_phase_arguments.phase_name == "TokenizerPreprocessor":
                pipeline_phase_arguments.phase_arguments["tokenizer"] = tokenizer
            data_processing_pipeline_step = TrainPipelineDataProcessingStep(
                pipeline_phase_arguments
            )
            dataset = data_processing_pipeline_step(dataset)
        logger.info("Data preprocessed")

        # NOTE: processing over we lift the barrier
        TrainPipeline.fail_safe_conditional_distributed_barrier(
            lambda: torch.distributed.get_rank() == 0 and IS_CUDA_AVAILABLE
        )

        # trainer
        # NOTE: if not output directory is provided, defaulting to the current working directory
        # using a fixed prefix and the task name.
        if not self.train_pipeline_configuration.pipeline_phases.trainer.args.get("output_dir", ""):
            output_dir = f"./train_pipeline_{self.train_pipeline_configuration.metadata.task_name}"
            self.train_pipeline_configuration.pipeline_phases.trainer.args["output_dir"] = (
                output_dir
            )
        # NOTE: make sure bf16 is used only if supported
        if "bf16" in self.train_pipeline_configuration.pipeline_phases.trainer.args:
            self.train_pipeline_configuration.pipeline_phases.trainer.args["bf16"] = (
                IS_BF16_AVAILABLE
                and self.train_pipeline_configuration.pipeline_phases.trainer.args["bf16"]
            )
        if "bf16_full_eval" in self.train_pipeline_configuration.pipeline_phases.trainer.args:
            self.train_pipeline_configuration.pipeline_phases.trainer.args["bf16_full_eval"] = (
                IS_BF16_AVAILABLE
                and self.train_pipeline_configuration.pipeline_phases.trainer.args["bf16_full_eval"]
            )
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                **self.train_pipeline_configuration.pipeline_phases.trainer.args
            ),
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            **self.train_pipeline_configuration.pipeline_phases.trainer.initialization_args,
        )
        trainer.place_model_on_device = (True,)

        # train
        logger.info("Start training...")
        trainer.train(**self.train_pipeline_configuration.pipeline_phases.trainer.train_args)
        logger.info("Finished training! Saving the model...")
        if not IS_CUDA_AVAILABLE or torch.distributed.get_rank() == 0:
            trainer.save_model(
                *self.train_pipeline_configuration.pipeline_phases.trainer.save_model_args
            )
        logger.info("Model saved")
        logger.info("Training pipeline completed")
