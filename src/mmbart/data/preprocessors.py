import logging
from typing import Union

from transformers import AutoTokenizer

from mmbart.data.preprocessing.carbon import CarbonPreprocessor
from mmbart.data.preprocessing.functional_group import FunctionalGroupPreprocessor
from mmbart.data.preprocessing.msms_number import MSMSNumberPreprocessor
from mmbart.data.preprocessing.msms_text import MSMSTextPreprocessor
from mmbart.data.preprocessing.multiplets import MultipletPreprocessor
from mmbart.data.preprocessing.normalization import NormalisePreprocessor
from mmbart.data.preprocessing.onehot import OneHotPreprocessor
from mmbart.data.preprocessing.patches import PatchPreprocessor
from mmbart.data.preprocessing.text_spectrum import (
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
