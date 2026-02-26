import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class MSMSNumberPreprocessor:
    normalise: bool = True
    encoding_type: str = "linear"

    normalisation_factors: Dict = field(init=False)

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        msms_spectra = sampled_dataset[modality]
        msms_spectra = self.filter_msms_peaks(msms_spectra)
        msms_spectra_flat = np.array([peak for spectra in msms_spectra for peak in spectra])

        self.normalisation_factors = dict()
        self.normalisation_factors["mass"] = {
            "mean": msms_spectra_flat[:, 0].mean(),
            "std": msms_spectra_flat[:, 0].std(),
        }
        self.normalisation_factors["intensity"] = {
            "mean": msms_spectra_flat[:, 1].mean(),
            "std": msms_spectra_flat[:, 1].std(),
        }

    def __call__(self, msms_spectra: List[List[List[float]]]) -> Dict[str, torch.Tensor]:
        msms_spectra = self.filter_msms_peaks(msms_spectra)
        spectra, attention_mask = self.pad_spectra(msms_spectra)

        return {"input_ids": spectra, "attention_mask": attention_mask}

    def filter_msms_peaks(self, msms_spectra: List[List[List[float]]]) -> List[List[List[float]]]:
        msms_spectra = [[peak for peak in spectra if peak[1] >= 1] for spectra in msms_spectra]
        return msms_spectra

    def pad_spectra(
        self, msms_spectra: List[List[List[float]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_max_len = max([len(spectra) for spectra in msms_spectra])

        padded_spectra = list()
        attention_masks = list()
        for spectra in msms_spectra:
            if self.normalise:
                spectra = [
                    [
                        (peak[0] - self.normalisation_factors["mass"]["mean"])
                        / self.normalisation_factors["mass"]["std"],
                        (peak[1] - self.normalisation_factors["intensity"]["mean"])
                        / self.normalisation_factors["intensity"]["std"],
                    ]
                    for peak in spectra
                ]

            spectra_tensor = torch.Tensor(spectra)
            padding = torch.zeros(((batch_max_len - len(spectra_tensor)), 2))

            padded_spectra.append(torch.concat((spectra_tensor, padding)))
            attention_masks.append(
                torch.concat((
                    torch.ones(len(spectra)),
                    torch.zeros(batch_max_len - len(spectra)),
                ))
            )

        return torch.stack(padded_spectra), torch.stack(attention_masks)
