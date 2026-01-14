import re
from typing import Dict, List

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor


class GuidedFormulaProcessor(LogitsProcessor):
    """Constrained Beam search to account for the correct chemical formula."""

    def __init__(
        self, n_beams: int, chemical_formula: List[str], target_tokenizer: AutoTokenizer
    ):
        """
        Args:
            n_beams: Beam size
            chemical_formula: Chemical formula to guide generation with (i.e. chemical formula of the target)
            target_tokenizer: Tokenizer for the target modality
        """
        super().__init__()

        # To do: Make atom list flexible
        self.atom_list = list(["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "H", "Se", "As", "Li", "Na", "Sn", "Te", "Al", "Zn", "Pb", "Hg", "Ag", "Ge", "Ti", "Sb", "Mg", "Bi", "K"])
        self.n_beams = n_beams
        self.target_tokenizer = target_tokenizer
        self.eos_token_id = target_tokenizer.eos_token_id
        self.vocab_size = target_tokenizer.vocab_size

        self.atom_id_token_id_dict: Dict[int, List[int]] = {
            i: list() for i in range(len(self.atom_list))
        }

        for token, token_id in target_tokenizer.vocab.items():
            # To do: Ensure all special tokens are ignored
            if token in ["<bos>", "<unk>", "<eos>", "<pad>"]:
                continue

            for i, atom in enumerate(self.atom_list):
                if atom == "H":
                    continue

                if atom.lower() in token.lower():
                    if (atom.lower() == 'c' and token.lower() == 'cl'):
                        continue
                    self.atom_id_token_id_dict[i].append(token_id)

        chemical_formula_encoded = np.stack(
            [self.make_formula_encoding(formula) for formula in chemical_formula],
            axis=0,
        )
        self.chemical_formula_beams: np.ndarray = np.repeat(
            chemical_formula_encoded, self.n_beams, axis=0
        )

    def make_formula_encoding(self, formula: str) -> np.ndarray:
        """Makes a vector corresponding to the number of atoms present in the chemical formula. Atom position is determined by index in self.atom_list
        Args:
            formula: Chemical Formula string
        Returns:
            np.ndarray: Vector with the number of atoms
        """

        pattern = r"([A-Z][a-z]?)(\d*)"
        matches = re.findall(pattern, formula)

        formula_encoding = np.zeros(len(self.atom_list))
        for atom, count in matches:
            formula_encoding[self.atom_list.index(atom)] = int(count) if count else 1

        return formula_encoding

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Guides generation using the target chemical formula:
            1. If smiles is valid and chemical formula correct                      -> Force EOS
            2. If current formula smaller than target formula                       -> Disallow EOS
            3. If next token would make current formula larger than target formula  -> Disallow token
        Args:
            input_ids: Already generated sequence
            scores: Logits for the next token
        Returns:
            torch.FloatTensor: Processed Scores
        """

        RDLogger.DisableLog("rdApp.*") # type: ignore

        # Decode Smiles
        decoded_smiles = self.target_tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
        decoded_smiles = [smiles.replace(" ", "") for smiles in decoded_smiles]
        decoded_smiles = [
            (
                Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                if Chem.MolFromSmiles(smiles)
                else ""
            )
            for smiles in decoded_smiles
        ]

        # Decode Formula and make numerical representation
        decoded_formula = list()
        for smiles in decoded_smiles:
            try:
                decoded_formula.append(
                    rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))
                )
            except:  # noqa E722
                decoded_formula.append("")
        decoded_formula = np.stack( # type: ignore
            [self.make_formula_encoding(formula) for formula in decoded_formula], axis=0
        )

        # If formula matches and valid smiles, set eos to 0
        decoded_formula_matching = np.all(
            self.chemical_formula_beams == decoded_formula, axis=1
        )
        scores[decoded_formula_matching, self.eos_token_id] = 0

        # If pred Formula smaller than valid formula, tank eos s
        decoded_formula_too_small = np.any(
            decoded_formula < self.chemical_formula_beams, axis=1
        )
        scores[decoded_formula_too_small, self.eos_token_id] = -float("inf")

        # Look ahead, do not check hydrogen
        target_formula = self.chemical_formula_beams.repeat(  # type: ignore
            self.vocab_size, axis=0
        ).reshape(
            decoded_formula.shape[0], self.vocab_size, len(self.atom_list)  # type: ignore
        )

        next_formula = decoded_formula.repeat(self.vocab_size, axis=0).reshape(  # type: ignore
            decoded_formula.shape[0], self.vocab_size, len(self.atom_list)  # type: ignore
        )

        for atom_index, token_ids in self.atom_id_token_id_dict.items():
            next_formula[:, token_ids, atom_index] += 1

        # Only up to index 9 -> Skip hydrogen. Very much work in progress
        next_formula_too_large = np.any(
            next_formula[:, :, :9] > target_formula[:, :, :9], axis=2
        )
        scores[next_formula_too_large] = -float("inf")

        return scores
