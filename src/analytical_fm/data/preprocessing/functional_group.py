import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

from ...configuration import DEFAULT_SETTINGS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class FunctionalGroupPreprocessor:
    group_definitions: str

    n_features: int = field(init=False)

    def __post_init__(self):

        if self.group_definitions == "default":
            self.functional_groups = DEFAULT_SETTINGS.default_func_groups
        else:
            raise ValueError(f"Unknown func_groups: {self.group_definitions}")

        self.n_features = len(self.functional_groups)

    def initialise(
        self,
        *args,
    ) -> None:
        pass

    # Defer import of get_functional_groups to avoid circular import; Better solution may be required.
    def __call__(self, smiles: List[str]) -> np.ndarray:
        from analytical_fm.data.data_utils import get_functional_groups

        return get_functional_groups(smiles, self.functional_groups)
