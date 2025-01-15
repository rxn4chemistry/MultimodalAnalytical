import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from mmbart.data.tokenizer import build_regex_tokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class MultipletPreprocessor:
    encoding: str = "text"
    j_values: bool = False
    normalise: bool = False

    tokenizer: AutoTokenizer = field(init=False)
    max_sequence_length: int = field(init=False)
    normalisation_factors: Dict = field(init=False)

    def initialise(
        self,
        sampled_dataset: Dataset,
        modality: str,
    ) -> None:

        multiplets = sampled_dataset[modality]

        processed_multiplets, numerical_encoding = self.process_multiplets(
            multiplets, self.encoding, self.j_values, initialise=True
        )

        self.tokenizer = build_regex_tokenizer(
            processed_multiplets,
            regex_string=f"(\s)",
            tokenizer_behaviour="removed",
        )

        longest_sequence = max(processed_multiplets, key=len)
        self.max_sequence_length = longest_sequence.count(" ") + 15

        if self.normalise:
            tokenized_input = self.tokenizer(
                processed_multiplets,
                padding="longest",
                max_length=self.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )

            padded_numerical_values = self.add_padding_numerical_values(
                tokenized_input,
                numerical_encoding,
            )
            padded_numerical_values = padded_numerical_values.view(-1)
            padded_numerical_values = padded_numerical_values[
                padded_numerical_values != 1
            ]

            mean = padded_numerical_values.mean()
            std = padded_numerical_values.std()
            self.normalisation_factors = {"mean": mean, "std": std}

    def __call__(
        self, multiplets: List[List[Dict[str, Union[str, float, int]]]]
    ) -> Dict[str, torch.Tensor]:
        processed_multiplets, numerical_encoding = self.process_multiplets(
            multiplets, self.encoding, self.j_values
        )

        tokenized_input = self.tokenizer(
            processed_multiplets,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        # Add numerical encoding to return dict and Pad
        if self.encoding == "numerical_encoding":
            padded_numerical_values = self.add_padding_numerical_values(
                tokenized_input,
                numerical_encoding,
            )
            tokenized_input["numerical_values"] = padded_numerical_values

        return tokenized_input

    def process_multiplets(
        self,
        multiplets: List[List[Dict[str, Union[str, float, int]]]],
        encoding: str,
        j_values: bool,
        initialise: bool = False,
    ) -> Tuple[List[str], List[np.ndarray]]:
        processed_multiplets = list()
        numerical_encodings = list()

        for multiplet in multiplets:
            multiplet_str, numerical_encoding_vector = self.process_multiplet(
                multiplet, encoding, j_values, initialise
            )
            processed_multiplets.append(multiplet_str)
            numerical_encodings.append(numerical_encoding_vector)

        return processed_multiplets, numerical_encodings

    def normalise_float(self, value: float) -> float:
        return (
            value - self.normalisation_factors["mean"]
        ) / self.normalisation_factors["std"]

    def process_multiplet(
        self,
        multiplets: List[Dict[str, Union[str, float, int]]],
        encoding: str,
        j_values: bool,
        initialise: bool = False,
    ) -> Tuple[str, np.ndarray]:
        if encoding not in ["text", "centroid", "numerical_encoding"]:
            raise ValueError(f"Unknown encoding type {encoding}")

        multiplet_str = "1HNMR "
        numerical_encoding_vector = [1.0]

        for peak in multiplets:
            if encoding == "text":
                formatted_peak = "{:.2f} {:.2f} {} {}H ".format(
                    float(peak["rangeMax"]),
                    float(peak["rangeMin"]),
                    peak["category"],
                    peak["nH"],
                )
            elif encoding == "centroid":
                formatted_peak = "{:.2f} {} {}H ".format(
                    float(peak["centroid"]),
                    peak["category"],
                    peak["nH"],
                )

            elif encoding == "numerical_encoding":
                formatted_peak = "[NUM] [NUM] {} {}H ".format(
                    peak["category"], peak["nH"]
                )

                if self.normalise and not initialise:
                    range_max = self.normalise_float(float(peak["rangeMax"]))
                    range_min = self.normalise_float(float(peak["rangeMin"]))
                else:
                    range_max = float(peak["rangeMax"])
                    range_min = float(peak["rangeMin"])

                numerical_encoding_vector.extend([range_max, range_min, 1.0, 1.0])

            js = str(peak["j_values"])
            if j_values and js != "None":
                split_js = js.split("_")
                split_js = list(filter(None, split_js))

                if encoding == "text":
                    processed_js = ["{:.2f}".format(float(j)) for j in split_js]
                    formatted_js = "J " + " ".join(processed_js)

                elif encoding == "numerical_encoding":
                    processed_js_numerical = [float(j) for j in split_js]

                    formatted_js = "J " + "[NUM] " * len(processed_js_numerical)

                    if self.normalise and not initialise:
                        processed_js_numerical = [
                            self.normalise_float(j) for j in processed_js_numerical
                        ]

                    numerical_encoding_vector.extend([1.0] + processed_js_numerical)

                formatted_peak += formatted_js
            multiplet_str += formatted_peak.strip() + " | "

            if encoding == "numerical_encoding":
                numerical_encoding_vector.append(1)

        # Remove last separating token
        multiplet_str = multiplet_str[:-3]
        numerical_encoding_vector_np = np.array(numerical_encoding_vector[:-1])

        return multiplet_str, numerical_encoding_vector_np

    def add_padding_numerical_values(
        self,
        tokenized_input: Dict[str, torch.Tensor],
        numerical_encodings: List[np.ndarray],
    ) -> torch.Tensor:
        # Get batch size and max batch sequence length
        batch_size = tokenized_input["input_ids"].shape[0]
        batch_sequence_length = tokenized_input["input_ids"].shape[1]

        # Only 1 token (BOS) before beginning of spectra
        start_padding = [np.ones((1)) for _ in range(batch_size)]

        padded_multiplet = [
            np.concatenate((padding, multiplet_vector))
            for padding, multiplet_vector in zip(start_padding, numerical_encodings)
        ]

        # Add end padding
        end_padding = [
            np.ones((batch_sequence_length - len(multiplet_vector)))
            for multiplet_vector in padded_multiplet
        ]

        padded_multiplet = [
            np.concatenate((multiplet_vector, padding))
            for padding, multiplet_vector in zip(end_padding, padded_multiplet)
        ]

        padded_multiplet_stacked = np.vstack(padded_multiplet)

        padded_multiplet_tensor = torch.Tensor(padded_multiplet_stacked)
        return padded_multiplet_tensor
