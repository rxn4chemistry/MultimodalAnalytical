import logging
from typing import Union

from transformers import AutoTokenizer

from analytical_fm.data.preprocessing.patches import PatchPreprocessor
from analytical_fm.data.preprocessing.text_spectrum import (
    PeakPositionalEncodingPreprocessor,
    RunLengthEncodingPreprocessor,
    TextSpectrumPreprocessor,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PREPROCESSORS = {
    "1D_patches": PatchPreprocessor,
    "peak_positional_encoding": PeakPositionalEncodingPreprocessor,
    "run_length_encoding": RunLengthEncodingPreprocessor,
    "text_spectrum": TextSpectrumPreprocessor,
}

return_type = Union[
    AutoTokenizer,
    PatchPreprocessor,
    PeakPositionalEncodingPreprocessor,
    RunLengthEncodingPreprocessor,
    TextSpectrumPreprocessor,
]
