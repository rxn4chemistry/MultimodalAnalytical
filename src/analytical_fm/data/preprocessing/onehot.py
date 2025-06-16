import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class OneHotPreprocessor:
    feature_path: str

    classes: Dict[Any, Any] = field(init=False)

    def __post_init__(self):
        features = pd.read_csv(self.feature_path)
        feature_dict = features[["Classes"]].to_dict()["Classes"]
        feature_dict = {val: key for key, val in feature_dict.items()}
        self.classes = feature_dict
        self.n_features = len(self.classes)

    def initialise(
        self,
        *args,
    ) -> None:
        pass

    def __call__(self, features: List[str]) -> np.ndarray:
        class_labels = [self.classes[feature] for feature in features]

        one_hot = np.zeros((len(features), len(self.classes)))
        one_hot[np.arange(len(features)), class_labels] = 1

        return one_hot
