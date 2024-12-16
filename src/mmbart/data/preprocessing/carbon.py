import logging
from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from multimodal.data.tokenizer import build_regex_tokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class CarbonPreprocessor:
    intensities: bool = False

    tokenizer: AutoTokenizer = field(init=False)
    max_sequence_length: int = field(init=False)

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        carbon_nmrs = sampled_dataset[modality]
        processed_carbon = self.process_carbon(
            carbon_nmrs,
        )

        self.tokenizer = build_regex_tokenizer(
            processed_carbon,
            regex_string=f"(\s)",
            tokenizer_behaviour="removed",
        )

        longest_sequence = max(processed_carbon, key=len)
        self.max_sequence_length = longest_sequence.count(" ") + 15

    def __call__(
        self, carbon_nmrs: List[List[Dict[str, Union[str, float, int]]]]
    ) -> torch.Tensor:
        processed_carbon = self.process_carbon(carbon_nmrs)

        tokenized_input = self.tokenizer(
            processed_carbon,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        return tokenized_input

    def process_carbon(
        self, carbon_nmrs: List[List[Dict[str, Union[str, float, int]]]]
    ) -> List[str]:
        processed_carbon = list()
        for nmr in carbon_nmrs:
            nmr_string = ""

            if self.intensities:
                intensity_sum = 0.0
                for peak in nmr:
                    intensity_sum += float(peak["intensity"])

            for peak in nmr:
                nmr_string = (
                    nmr_string + str(round(float(peak["delta (ppm)"]), 1)) + " "
                    if "delta (ppm)" in peak
                    else "blah" + " "
                )
                if self.intensities:
                    nmr_string = (
                        nmr_string
                        + str(round(float(peak["intensity"]) / intensity_sum, 1))
                        + " "
                    )

            processed_carbon.append(nmr_string.strip())
        return processed_carbon
