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

        self.atom_list = list(
            [
                "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba", "Be", "Bh", "Bi", "Bk",
                "Br", "C", "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Co", "Cr", "Cs", "Cu", "Db", "Dy",
                "Er", "Es", "Eu", "F", "Fe", "Fm", "Fr", "Ga", "Gd", "Ge", "H", "He", "Hf", "Hg",
                "Ho", "Hs", "I", "In", "Ir", "K", "Kr", "La", "Li", "Lr", "Lu", "Md", "Mg", "Mn",
                "Mo", "Mt", "N", "Na", "Nb", "Nd", "Ne", "Ni", "No", "Np", "O", "Os", "P", "Pa",
                "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu", "Ra", "Rb", "Re", "Rf", "Rh", "Rn", "Ru",
                "S", "Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th",
                "Ti", "Tl", "Tm", "U", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr"
            ]
        )
        self.n_beams = n_beams
        self.target_tokenizer = target_tokenizer
        self.eos_token_id = target_tokenizer.eos_token_id
        self.vocab_size = target_tokenizer.vocab_size

        # Map each token_id to the atoms it contains
        # Structure: {token_id: {atom_index: count}}
        # Example: token "CC" -> {C_index: 2}, token "[NH]" -> {N_index: 1, H_index: 1}
        self.token_atom_counts: Dict[int, Dict[int, int]] = {}

        for token, token_id in target_tokenizer.vocab.items():
            # Skip special tokens
            if token in ["<bos>", "<unk>", "<eos>", "<pad>"]:
                continue

            atom_counts: Dict[int, int] = {}

            # Scan through the token and extract atom symbols (aromatic lowercase included)
            i = 0
            while i < len(token):
                if not token[i].isalpha():
                    # [, ], (, ), =, #, @, +, -, digits, ...
                    i += 1
                    continue

                # Try two-letter atom first (e.g., Br, Cl)
                if i + 1 < len(token) and token[i + 1].islower():
                    two_letter = token[i].upper() + token[i + 1]
                    if two_letter in self.atom_list:
                        atom_idx = self.atom_list.index(two_letter)
                        atom_counts[atom_idx] = atom_counts.get(atom_idx, 0) + 1
                        i += 2
                        continue

                # Try single-letter atom (e.g., C, N, O, c, n, o)
                single_letter = token[i].upper()
                if single_letter in self.atom_list:
                    atom_idx = self.atom_list.index(single_letter)
                    atom_counts[atom_idx] = atom_counts.get(atom_idx, 0) + 1
                i += 1

            if atom_counts:
                self.token_atom_counts[token_id] = atom_counts

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

        # Look ahead: check if adding each token would exceed the target formula.
        # Hydrogen is skipped because it is mostly implicit in SMILES (except in brackets like [NH])
        h_index = self.atom_list.index("H")

        target_formula = self.chemical_formula_beams.repeat(  # type: ignore
            self.vocab_size, axis=0
        ).reshape(
            decoded_formula.shape[0], self.vocab_size, len(self.atom_list)  # type: ignore
        )

        next_formula = decoded_formula.repeat(self.vocab_size, axis=0).reshape(  # type: ignore
            decoded_formula.shape[0], self.vocab_size, len(self.atom_list)  # type: ignore
        )

        for token_id, atom_counts in self.token_atom_counts.items():
            for atom_idx, count in atom_counts.items():
                next_formula[:, token_id, atom_idx] += count

        # Disallow the token if any non-hydrogen atom count would exceed the target
        exceeds_target = next_formula > target_formula
        exceeds_target[:, :, h_index] = False
        next_formula_too_large = np.any(exceeds_target, axis=2)
        scores[next_formula_too_large] = -float("inf")

        return scores
