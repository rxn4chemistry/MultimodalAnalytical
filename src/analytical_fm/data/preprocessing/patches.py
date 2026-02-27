import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class PatchPreprocessor:
    mean: float = field(init=False)
    std: float = field(init=False)
    mean_deriv: Optional[torch.Tensor] = field(default=None, init=False)
    std_deriv: Optional[torch.Tensor] = field(default=None, init=False)

    patch_size: int = field(init=True)
    masking: bool = field(init=True)
    interpolation: bool = field(init=True)
    overlap: int = field(init=True, default=1)
    derivative: bool = field(init=True, default=False)

    encoding_type: str = field(init=True, default="")

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        # Initialise processors

        spectra = np.array(sampled_dataset[modality])
        self.mean = spectra[spectra != 0].mean()
        self.std = spectra[spectra != 0].std()

        if self.derivative:
            spectra_tensor = torch.Tensor(spectra)
            gradient = torch.gradient(spectra_tensor, dim=-1)[0]

            self.mean_deriv = gradient.mean()
            self.std_deriv = gradient.std()

    def interpolate(self, spectra: List[float]) -> List[float]:
        old_x = np.arange(400, 4000 if len(spectra) == 1800 else 3982, 2)
        new_x = np.arange(650, 3900, 2)
        interp = interp1d(old_x, spectra)
        return interp(new_x)

    def __call__(self, spectra: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalises spectra based on mean and std. Divides spectra into patches.
        Args:
            spectra: List (batch_size, spectra_length)
        Returns:
            Tensor: (batch_size, spectra_length/patch_size, patch_size)
        """

        spec_size_mask = [len(spectrum) if spectrum is not None else -1 for spectrum in spectra]
        max_spec_size = max(spec_size_mask) if max(spec_size_mask) != -1 else 500
        for i in range(len(spectra)):
            if spectra[i] is None:
                spectra[i] = [0] * max_spec_size

        if self.interpolation:
            spectra = [self.interpolate(spectrum) for spectrum in spectra]

        # Concert to tensor
        spectra_tensor = torch.Tensor(spectra)

        # Standardise
        standardised_spectra = (spectra_tensor - self.mean) / self.std

        # Trim spectrum to fit into even patches (No padding)
        n_patches = standardised_spectra.shape[1] // self.patch_size
        trim_end = n_patches * self.patch_size
        trimmed_spectra = standardised_spectra[:, :trim_end]

        # Dividide into patches
        if self.overlap == 1:
            patched_spectrum = trimmed_spectra.view(-1, n_patches, self.patch_size)
        else:
            patched_spectrum = trimmed_spectra.unfold(
                -1, self.patch_size, self.patch_size // self.overlap
            )

        if self.derivative:
            gradient = torch.gradient(spectra_tensor, dim=-1)[0]
            gradient_trimmed = gradient[:, :trim_end]
            gradient_patched = gradient_trimmed.view(-1, n_patches, self.patch_size)
            patched_spectrum = torch.concat((patched_spectrum, gradient_patched), dim=1)

        # Construct attention mask
        if self.masking:
            summed_patched_spectrum = patched_spectrum.sum(-1)
            attention_mask = summed_patched_spectrum == 0
        else:
            attention_mask = torch.stack([
                torch.full((patched_spectrum.shape[1],), (spec_size_mask[i] == -1))
                for i in range(patched_spectrum.shape[0])
            ])

        return patched_spectrum, attention_mask
