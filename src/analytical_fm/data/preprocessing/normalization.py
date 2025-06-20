import logging
from dataclasses import dataclass, field

import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class NormalisePreprocessor:
    mean: float = field(init=False)
    std: float = field(init=False)

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        data = np.array(sampled_dataset[modality])
        self.mean = data.mean()
        self.std = data.std()
        self.n_features = data.shape[-1]

    def normalise(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def denormalise(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.normalise(data)
