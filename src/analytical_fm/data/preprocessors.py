import logging
from typing import Union

from transformers import AutoTokenizer

from .preprocessing.carbon import CarbonPreprocessor
from .preprocessing.functional_group import FunctionalGroupPreprocessor
from .preprocessing.msms_number import MSMSNumberPreprocessor
from .preprocessing.msms_text import MSMSTextPreprocessor
from .preprocessing.multiplets import MultipletPreprocessor
from .preprocessing.no_action import NoActionPreprocessor
from .preprocessing.normalization import NormalisePreprocessor
from .preprocessing.onehot import OneHotPreprocessor
from .preprocessing.patches import PatchPreprocessor
from .preprocessing.text_spectrum import (
    PeakPositionalEncodingPreprocessor,
    RunLengthEncodingPreprocessor,
    TextSpectrumPreprocessor,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PREPROCESSORS = {
    "carbon": CarbonPreprocessor,
    "functional_group": FunctionalGroupPreprocessor,
    "msms_number": MSMSNumberPreprocessor,
    "msms_text": MSMSTextPreprocessor,
    "multiplets": MultipletPreprocessor,
    "normalise": NormalisePreprocessor,
    "class_one_hot": OneHotPreprocessor,
    "1D_patches": PatchPreprocessor,
    "peak_positional_encoding": PeakPositionalEncodingPreprocessor,
    "run_length_encoding": RunLengthEncodingPreprocessor,
    "text_spectrum": TextSpectrumPreprocessor,
    "no_action": NoActionPreprocessor,
}

return_type = Union[
    AutoTokenizer,
    FunctionalGroupPreprocessor,
    MultipletPreprocessor,
    NormalisePreprocessor,
    PatchPreprocessor,
    PeakPositionalEncodingPreprocessor,
    RunLengthEncodingPreprocessor,
    TextSpectrumPreprocessor,
    NoActionPreprocessor,
]
