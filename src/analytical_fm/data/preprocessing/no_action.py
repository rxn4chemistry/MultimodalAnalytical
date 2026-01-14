import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class NoActionPreprocessor:

    def initialise(
        self,
        *args,
    ) -> None:
        pass

    def __call__(self, features: List[str]) -> List[str]:
        return features
