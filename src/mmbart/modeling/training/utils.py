"""Utils for training."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed everything.

    Args:
        seed: seed to use.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
