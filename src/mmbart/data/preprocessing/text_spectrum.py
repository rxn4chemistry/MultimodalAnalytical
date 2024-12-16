import logging
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import regex as re
import torch
from datasets import Dataset
from scipy import interpolate
from sklearn.cluster import OPTICS, KMeans
from transformers import AutoTokenizer

from multimodal.data.tokenizer import build_regex_tokenizer
from multimodal.defaults import DEFAULT_SAMPLES

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class TextSpectrumPreprocessor:
    """Preprocessor that merges formula and spectrum into one text representation."""

    spectrum_tokens_x: int = 400
    spectrum_tokens_y: int = 100

    formula_regex: str = r"([A-Z]{1}[a-z]?[0-9]*)"
    spectrum_to_text_x: str = "whole_spectrum"
    spectrum_to_text_y: str = "integer"
    modality_type: str = "ir"
    spectra_only: bool = False
    spectra_column: str = ""
    formula_column: str = ""
    numerical_encoding_strength: int = 10

    tokenizer: AutoTokenizer = field(init=False)
    max_sequence_length: int = field(init=False)

    processing_parameters: Dict[str, Any] = field(init=False)

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        self.modality = modality

        # Assure data is in numpy arrays
        spectra = np.array(sampled_dataset[self.spectra_column])

        if self.modality_type in ["ir", "nmr"]:
            formulae = np.array(sampled_dataset[self.formula_column])
        else:
            formulae = None

        # Initialise processors
        self.processing_parameters = dict()
        self.initialise_x_processors(spectra, self.spectrum_tokens_x)
        processed_spectra_x, _ = self.process_spectra_x(spectra)
        self.initialise_y_processors(processed_spectra_x, self.spectrum_tokens_y)

        # Process spectr
        processed_spectra, _ = self.process_spectra(spectra)

        if not self.spectra_only:
            if formulae is None:
                raise ValueError("formulae is None.")
            processed_formulae = self.process_formulae(formulae)
            combined_formula_spectra = [
                processed_formula + " " + processed_spectrum
                for processed_formula, processed_spectrum in zip(
                    processed_formulae, processed_spectra
                )
            ]
        else:
            combined_formula_spectra = processed_spectra

        self.tokenizer = build_regex_tokenizer(
            combined_formula_spectra,
            regex_string=f"(\s)",
            tokenizer_behaviour="removed",
        )
        longest_sequence = max(combined_formula_spectra, key=len)
        self.max_sequence_length = longest_sequence.count(" ") + 10

    def __call__(
        self, spectra: np.ndarray, formulae: Optional[List[str]] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        processed_spectra, _ = self.process_spectra(spectra)

        if not self.spectra_only:
            if formulae is None:
                raise ValueError("formulae is None.")

            processed_formulae = self.process_formulae(formulae)
            combined_formula_spectra = [
                processed_formula + " " + processed_spectrum
                for processed_formula, processed_spectrum in zip(
                    processed_formulae, processed_spectra
                )
            ]
        else:
            combined_formula_spectra = processed_spectra

        tokenized_input = self.tokenizer(
            combined_formula_spectra,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        # Add numerical encoding to return dict and Pad
        if self.spectrum_to_text_y == "numerical_encoding":
            if self.spectra_only:
                processed_formulae = []

            if not isinstance(spectra, np.ndarray):
                spectra = np.array(spectra)

            padded_numerical_values = self.add_padding_numerical_values(
                spectra,
                processed_formulae,
                tokenized_input,
                self.numerical_encoding_strength,
            )
            tokenized_input["numerical_values"] = padded_numerical_values

        return tokenized_input

    def process_formulae(self, formulae: Union[List[str], np.ndarray]) -> List[str]:
        # Split formula into pieces: C6H12O6 -> C6 H12 O6

        processed_formulae = list()
        for formula in formulae:
            split_formula = re.split(self.formula_regex, formula)
            split_formula = list(filter(None, split_formula))
            processed_formulae.append(" ".join(split_formula))

        return processed_formulae

    def process_spectra(
        self, spectra: np.ndarray
    ) -> Tuple[List[str], List[np.ndarray]]:
        processed_spectra_x, indices = self.process_spectra_x(spectra)
        return self.process_spectra_y(processed_spectra_x), indices

    def process_spectra_x(
        self, spectra: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        processed_spectra_x = list()
        indices = list()
        for spectrum in spectra:
            if not isinstance(spectrum, np.ndarray):
                spectrum = np.array(spectrum)

            if self.spectrum_to_text_x == "no_action":
                processed_spectrum_x = spectrum
            elif self.spectrum_to_text_x == "whole_spectrum":
                processed_spectrum_x = self._process_spectrum_x_fixed(
                    spectrum, x_window="whole"
                )
            elif self.spectrum_to_text_x == "window":
                processed_spectrum_x = self._process_spectrum_x_fixed(
                    spectrum, x_window="merged"
                )
            elif self.spectrum_to_text_x == "variance":
                processed_spectrum_x = self._process_spectrum_x_variance(spectrum)
            elif self.spectrum_to_text_x == "threshold" and isinstance(
                self, PeakPositionalEncodingPreprocessor
            ):
                processed_spectrum_x, index = self._process_spectrum_x_threshold(
                    spectrum, self.modality_type
                )
                indices.append(index)
            elif self.spectrum_to_text_x == "run_length_encoding":
                processed_spectrum_x = self._process_spectrum_x_fixed(
                    spectrum, x_window="run_length_encoding"
                )
            else:
                raise ValueError(
                    f"Processing {self.spectrum_to_text_x} not implemented. Choose from whole_spectrum, window or variance."
                )

            processed_spectra_x.append(processed_spectrum_x)

        return processed_spectra_x, indices

    def process_spectra_y(self, processed_spectra_x: List[np.ndarray]) -> List[str]:
        # Quite inefficient looping twice over all spectra (first for processing x-axis then y-axis)

        # First process x axis of spectrum, either use whole spectrum, window, or variance based

        # Process y axis
        processed_string_spectra = list()
        for processed_spectrum_x in processed_spectra_x:
            if self.spectrum_to_text_y == "integer":
                processed_spectrum_xy = self._process_spectrum_y_integer(
                    processed_spectrum_x, self.spectrum_tokens_y
                )
            elif self.spectrum_to_text_y == "frequency_based_clustering":
                processed_spectrum_xy = self._process_spectrum_y_frequency(
                    processed_spectrum_x
                )
            elif self.spectrum_to_text_y == "k_means_clustering":
                processed_spectrum_xy = self._process_spectrum_y_k_mean(
                    processed_spectrum_x
                )
            elif self.spectrum_to_text_y == "density_based_clustering":
                processed_spectrum_xy = self._process_spectrum_y_density(
                    processed_spectrum_x
                )
            elif self.spectrum_to_text_y == "numerical_encoding":
                processed_spectrum_xy = self._process_spectrum_y_numerical(
                    processed_spectrum_x
                )
            else:
                raise ValueError(
                    f"Processing {self.spectrum_to_text_y} not implemented. Choose from integer, frequency_based_clustering, k_means_clustering or density_based_clustering."
                )

            # Convert spectrum to String
            string_spectrum = " ".join(processed_spectrum_xy.astype(str))
            processed_string_spectra.append(string_spectrum)

        return processed_string_spectra

    def initialise_x_processors(
        self, spectra: np.ndarray, sequence_length: int
    ) -> None:
        if self.spectrum_to_text_x == "variance":
            # Select the indexes with the highest variance and feed only them into the model
            variance = spectra.var(0)
            top_variance_index = np.sort(np.argsort(variance)[-sequence_length:])
            self.processing_parameters["variance"] = {
                "top_variance_index": top_variance_index
            }

    def _process_spectrum_x_fixed(
        self,
        spectrum: np.ndarray,
        x_window: str = "whole",
        orig_x: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Process Spectrum according to methods from: 10.26434/chemrxiv-2023-5v27f Options allow whole spectrum or merged window

        if orig_x is None:
            orig_x = np.arange(0, len(spectrum))

        if x_window == "whole":
            new_x = np.linspace(0, len(spectrum) - 2, self.spectrum_tokens_x)
        elif x_window == "merged":
            orig_x = np.arange(0, 3980, 2)
            resolution = (2000 - 400 + 500) / self.spectrum_tokens_x
            new_x = np.concatenate(
                [
                    np.arange(400, 2000, resolution),
                    np.arange(2800, 3300 - resolution, resolution),
                ]
            )
        elif x_window == "run_length_encoding":
            new_x = np.linspace(0, len(spectrum) - 2, self.spectrum_tokens_x * 2)
        else:
            raise ValueError(f"Invalid option: {x_window}")

        intp = interpolate.interp1d(orig_x, spectrum)

        intp_spectrum = intp(new_x)

        return intp_spectrum

    def _process_spectrum_x_variance(self, spectrum: np.ndarray) -> np.ndarray:
        return spectrum[self.processing_parameters["variance"]["top_variance_index"]]

    def _process_spectrum_x_threshold(
        self, spectrum: np.ndarray, modality_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        if modality_type == "ir":
            orig_x = np.arange(400, 3982, 2)
            intp = interpolate.interp1d(orig_x, spectrum)

            new_x = np.linspace(400, 3980, 2 * self.spectrum_tokens_x)
            intp_spectrum = intp(new_x)

            spectrum_median = np.median(intp_spectrum)
            thresholded_spectrum = intp_spectrum[intp_spectrum > spectrum_median]
            indices_spectrum = np.argwhere(intp_spectrum > spectrum_median)

        elif modality_type in ["nmr", "sc", "weather"]:
            quantile = 1 - (self.spectrum_tokens_x / spectrum.shape[0])
            threshold = np.quantile(spectrum, quantile)
            mask = spectrum > threshold

            thresholded_spectrum = spectrum[mask]
            indices_spectrum = np.argwhere(mask).flatten()

            # Pad thresholded spectrum if necessary
            if len(thresholded_spectrum) < self.spectrum_tokens_x:
                # How many values we have to pad by
                n_padding = self.spectrum_tokens_x - len(thresholded_spectrum)
                padding = np.zeros(n_padding)
                padding_indices = np.arange(len(spectrum), len(spectrum) + n_padding)

                thresholded_spectrum = np.concatenate([thresholded_spectrum, padding])
                indices_spectrum = np.concatenate([indices_spectrum, padding_indices])

        else:
            raise ValueError(f"Unknow modality type {modality_type}")

        return thresholded_spectrum, indices_spectrum

    def initialise_y_processors(
        self, spectra: List[np.ndarray], vocab_size_y: int
    ) -> None:
        indices = np.arange(0, len(spectra), 1)
        chosen_indices = np.random.choice(
            indices, size=min(len(spectra), DEFAULT_SAMPLES), replace=False
        )
        if isinstance(spectra, list):
            sampled_spectra = [spectra[i] for i in chosen_indices]
            flat_spectra = np.concatenate(sampled_spectra, -1).flatten()
        else:
            sampled_spectra = spectra[chosen_indices]
            flat_spectra = np.hstack(sampled_spectra).flatten()

        if self.spectrum_to_text_y in ["integer", "numerical_encoding"]:
            pass

        elif self.spectrum_to_text_y == "frequency_based_clustering":
            if self.modality_type in ["nmr", "sc", "weather"]:
                flat_spectra = np.around(flat_spectra, 6)
                flat_spectra = np.unique(flat_spectra)

            _, bins = pd.qcut(
                flat_spectra, vocab_size_y, retbins=True, duplicates="drop"
            )
            labels = [f"freq_{i}" for i in range(1, vocab_size_y + 1)]
            self.processing_parameters["frequency"] = {"bins": bins, "labels": labels}

        elif self.spectrum_to_text_y == "k_means_clustering":
            k_means = KMeans(n_clusters=100, verbose=1, n_init=5)
            k_means.fit(flat_spectra.reshape(-1, 1))
            self.processing_parameters["k_means"] = {"model": k_means}

        elif self.spectrum_to_text_y == "density_based_clustering":
            density_cluster = OPTICS(n_jobs=-1)
            density_cluster.fit(flat_spectra.reshape(-1, 1))
            self.processing_parameters["density"] = {"model": density_cluster}

        else:
            raise ValueError(f"Invalid option: {self.spectrum_to_text_y}")

    def _process_spectrum_y_integer(
        self, spectrum: np.ndarray, spectrum_tokens_y: int
    ) -> np.ndarray:
        normalised_spectrum = spectrum / max(spectrum) * spectrum_tokens_y
        normalised_spectrum = np.rint(normalised_spectrum)
        normalised_spectrum = np.clip(normalised_spectrum, 0, spectrum_tokens_y).astype(
            int
        )
        return normalised_spectrum

    def _process_spectrum_y_frequency(self, spectrum: np.ndarray) -> np.ndarray:
        spectrum = np.clip(
            spectrum,
            self.processing_parameters["frequency"]["bins"][0] + 1e-7,
            self.processing_parameters["frequency"]["bins"][-1] - 1e-7,
        )
        processed_spectrum = [
            self.processing_parameters["frequency"]["labels"][i - 1]
            for i in np.digitize(
                spectrum, self.processing_parameters["frequency"]["bins"]
            )
        ]

        return np.array(processed_spectrum)

    def _process_spectrum_y_k_mean(self, spectrum: np.ndarray) -> np.ndarray:
        processed_spectrum = self.processing_parameters["k_means"]["model"].predict(
            spectrum.reshape(-1, 1)
        )

        return processed_spectrum

    def _process_spectrum_y_density(self, spectrum: np.ndarray) -> np.ndarray:
        processed_spectrum = self.processing_parameters["density"]["model"].predict(
            spectrum.reshape(-1, 1)
        )

        return processed_spectrum

    def _process_spectrum_y_numerical(self, spectrum: np.ndarray) -> np.ndarray:
        text_encoding = np.full((spectrum.shape), "[NUM]")

        return text_encoding

    def add_padding_numerical_values(
        self,
        spectra: np.ndarray,
        processed_formulae: List[str],
        tokenized_input: Dict[str, torch.Tensor],
        strength: int,
    ) -> torch.Tensor:
        # Bring all values above 1
        processed_spectra, _ = self.process_spectra_x(spectra)
        processed_spectra_np = np.vstack(processed_spectra)

        processed_spectra_np = (
            processed_spectra_np / np.expand_dims(np.max(processed_spectra_np, -1), -1)
        ) * strength

        # Append values at the beginning of the spectra to account for bos, chem formula etc
        batch_size = spectra.shape[0]
        if not self.spectra_only:
            # Create lists of np.ones to append before the spectra: 1 (BOS) + len(tokenized formula)
            start_padding = [
                np.ones((1 + formula.count(" ") + 1)) for formula in processed_formulae
            ]
        else:
            # Only 1 token (BOS) before beginning of spectra
            start_padding = [np.ones((1)) for _ in range(batch_size)]

        padded_spectra = [
            np.concatenate((padding, spectrum))
            for padding, spectrum in zip(start_padding, processed_spectra_np)
        ]

        # Add end padding
        sequence_length = tokenized_input["input_ids"].shape[-1]
        end_padding = [
            np.ones((sequence_length - len(pad_spectrum)))
            for pad_spectrum in padded_spectra
        ]

        padded_spectra = [
            np.concatenate((pad_spectrum, padding))
            for padding, pad_spectrum in zip(end_padding, padded_spectra)
        ]

        padded_spectra_stacked = np.vstack(padded_spectra)

        padded_spectra_tensor = torch.Tensor(padded_spectra_stacked)
        return padded_spectra_tensor


@dataclass
class RunLengthEncodingPreprocessor(TextSpectrumPreprocessor):

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        # Initialise processors

        spectra = np.array(sampled_dataset[modality])
        # Initialise processors
        if self.spectrum_to_text_x not in [
            "run_length_encoding",
            "no_action",
            "whole_spectrum",
        ]:
            raise ValueError(
                "Expected spectrum_to_text_x to be in ['run_length_encoding', 'no_action']"
            )

        if self.spectrum_to_text_y not in ["integer", "frequency_based_clustering"]:
            raise ValueError(
                f"Option {self.spectrum_to_text_y} not available for Run Length Encoding. Choose from ['integer', 'frequency_based_clustering']"
            )

        self.processing_parameters = dict()

        processed_spectra_x, _ = self.process_spectra_x(spectra)
        self.initialise_y_processors(processed_spectra_x, self.spectrum_tokens_y)

        processed_spectra, _ = self.process_spectra(spectra)
        run_length_encodings = self.get_run_length_encoding(processed_spectra)

        self.tokenizer = build_regex_tokenizer(
            run_length_encodings,
            regex_string=f"(\s)",
            tokenizer_behaviour="removed",
        )

        longest_sequence = max(run_length_encodings, key=len)
        self.max_sequence_length = min(4090, longest_sequence.count(" ") + 10)

    def __call__(self, spectra: np.ndarray) -> Dict[str, Any]:  # type: ignore[override]
        processed_spectra, _ = self.process_spectra(spectra)

        run_length_spectra = self.get_run_length_encoding(processed_spectra)

        tokenized_input = self.tokenizer(
            run_length_spectra,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_input

    def get_run_length_encoding(self, spectra: List[str]) -> List[str]:
        string_spectra = list()
        for spectrum in spectra:
            run_lengths = [
                (k, sum(1 for i in g)) for k, g in groupby(spectrum.split(" "))
            ]

            string_spectrum = ""
            for value, n in run_lengths:
                string_spectrum += f"{value} {n} "
            string_spectra.append(string_spectrum.strip())
        return string_spectra


@dataclass
class PeakPositionalEncodingPreprocessor(TextSpectrumPreprocessor):

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        # Initialise processors

        spectra = np.array(sampled_dataset[modality])

        if self.spectrum_to_text_x not in ["variance", "threshold"]:
            raise ValueError(
                f"Option {self.spectrum_to_text_x} not available for Peak Positional Encoding. Choose from ['variance', 'threshold']"
            )

        self.processing_parameters = dict()
        self.initialise_x_processors(spectra, self.spectrum_tokens_x)

        processed_spectra_x, indices = self.process_spectra_x(spectra)
        self.initialise_y_processors(processed_spectra_x, self.spectrum_tokens_y)

        processed_spectra, _ = self.process_spectra(spectra)

        self.tokenizer = build_regex_tokenizer(
            processed_spectra,
            regex_string=f"(\s)",
            tokenizer_behaviour="removed",
        )

        longest_sequence = max(processed_spectra, key=len)
        self.max_sequence_length = longest_sequence.count(" ") + 30

    def __call__(self, spectra: np.ndarray) -> List[str]:  # type: ignore[override]
        processed_spectra, indices = self.process_spectra(spectra)
        tokenized_input = self.tokenizer(
            processed_spectra,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        # Shape token indices to conform with bos token and padding -> Insert one index at the front, max_sequence_length - current length of indices at the end
        if self.spectrum_to_text_x == "threshold":
            indices = [row + 1 for row in indices]

            max_spectra_len = (
                2 * self.spectrum_tokens_x
                if self.modality_type == "ir"
                else len(spectra[0])
            )

            token_indices_np = np.array(
                [
                    np.append(
                        np.append([0], row),
                        np.arange(
                            max_spectra_len + 1,
                            max_spectra_len + (self.max_sequence_length - len(row)),
                        ),
                    )
                    for row in indices
                ]
            )
            token_indices = token_indices_np.tolist()

        elif self.spectrum_to_text_x == "variance":
            token_indices = self.processing_parameters["variance"][
                "top_variance_index"
            ].tolist()

            token_indices.insert(0, min(token_indices) - 1)

            end_indices = list(
                range(
                    max(token_indices) + 1,
                    max(token_indices)
                    + (self.max_sequence_length - len(token_indices))
                    + 1,
                    1,
                )
            )
            token_indices.extend(end_indices)

            # Shape to batch_size * sequence_length
            batch_size = len(spectra)
            token_indices = [token_indices.copy() for _ in range(batch_size)]

        tokenized_input["indices"] = token_indices

        # Add numerical encoding to return dict and Pad
        if self.spectrum_to_text_y == "numerical_encoding":
            if not isinstance(spectra, np.ndarray):
                spectra = np.array(spectra)

            padded_numerical_values = self.add_padding_numerical_values(
                spectra,
                ["" for _ in range(spectra.shape[0])],
                tokenized_input,
                self.numerical_encoding_strength,
            )
            tokenized_input["numerical_values"] = padded_numerical_values

        return tokenized_input
