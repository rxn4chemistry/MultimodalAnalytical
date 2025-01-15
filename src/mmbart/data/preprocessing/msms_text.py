import logging
from dataclasses import dataclass, field
from typing import List

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from mmbart.data.tokenizer import build_regex_tokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class MSMSTextPreprocessor:
    tokenizer: AutoTokenizer = field(init=False)
    max_sequence_length: int = field(init=False)

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        msms_spectra = sampled_dataset[modality]
        processed_msms = self.process_msms(msms_spectra)

        self.tokenizer = build_regex_tokenizer(
            processed_msms,
            regex_string=f"(\s)",
            tokenizer_behaviour="removed",
        )

        longest_sequence = max(processed_msms, key=len)
        self.max_sequence_length = longest_sequence.count(" ") + 15

    def __call__(self, msms_spectra: List[List[List[float]]]) -> torch.Tensor:
        processed_msms = self.process_msms(msms_spectra)

        tokenized_input = self.tokenizer(
            processed_msms,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        return tokenized_input

    def process_msms(self, msms_spectra: List[List[List[float]]]) -> List[str]:
        processed_msms = list()

        for msms in msms_spectra:
            msms_string = ""
            for peak in msms:
                if peak[1] < 1:
                    continue
                msms_string = msms_string + "{:.1f} {:.1f} ".format(
                    round(peak[0], 1), round(peak[1], 1)
                )

            msms_string = msms_string.strip()
            processed_msms.append(msms_string)

        return processed_msms
