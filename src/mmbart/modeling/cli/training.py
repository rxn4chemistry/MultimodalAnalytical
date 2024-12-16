"""Run the training pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
from pathlib import Path

import click
from mmbart.modeling.training.core import TrainPipeline


@click.command()
@click.option(
    "--pipeline_configuration_path",
    help="path to the yaml file defining the pipeline to be executed",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
def execute_training_pipeline(pipeline_configuration_path: Path) -> None:
    """Execute the training pipeline.

    Args:
        pipeline_configuration_path: configuration of the pipeline from the user.
    """
    train_pipeline = TrainPipeline(pipeline_configuration_path)
    train_pipeline.run_training_pipeline()


if __name__ == "__main__":
    execute_training_pipeline()
