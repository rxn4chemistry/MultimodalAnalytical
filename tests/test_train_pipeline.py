"""Test the training pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
import logging
from pathlib import Path

from click.testing import CliRunner

from mmbart.modeling.cli.training import execute_training_pipeline

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def test_pipeline_invocation() -> None:
    """Test the pipeline via CLI invocation."""
    runner = CliRunner()
    result = runner.invoke(
        execute_training_pipeline,
        [
            "--pipeline_configuration_path",
            str(
                Path(
                    "src/mmbart/modeling/resources/train_pipeline_configuration_example.yaml"
                )
            ),
        ],
    )
    logger.info(result)


def test_pipeline_execution() -> None:
    """Test the pipeline execution."""
    execute_training_pipeline.callback(  # type:ignore
        Path(
            "src/mmbart/modeling/resources/train_pipeline_configuration_example.yaml"
        ),
    )
