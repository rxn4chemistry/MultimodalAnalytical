import logging
from typing import Union

from transformers import AutoTokenizer

from multimodal.data.preprocessing.carbon import CarbonPreprocessor
from multimodal.data.preprocessing.functional_group import FunctionalGroupPreprocessor
from multimodal.data.preprocessing.msms_number import MSMSNumberPreprocessor
from multimodal.data.preprocessing.msms_text import MSMSTextPreprocessor
from multimodal.data.preprocessing.multiplets import MultipletPreprocessor
from multimodal.data.preprocessing.normalization import NormalisePreprocessor
from multimodal.data.preprocessing.onehot import OneHotPreprocessor
from multimodal.data.preprocessing.patches import PatchPreprocessor
from multimodal.data.preprocessing.text_spectrum import (
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
]
