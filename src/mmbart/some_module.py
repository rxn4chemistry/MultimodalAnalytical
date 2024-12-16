"""Some module."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def dummy_function(n: int) -> int:
    """
    Do nothing very interesting.

    This function iterates a few times and increments a value.

    Args:
        n: number of iterations to do.

    Returns:
        the double of the given number.
    """
    a = 0
    for i in tqdm(range(n)):
        logger.info(f"Iteration number {i+1}.")
        a += 2
    return a
