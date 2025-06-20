from typing import Dict

import numpy as np
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from rdkit import Chem


class DefaultSettings(BaseSettings):

    default_seed: int = Field(default=3247, description="Seed")

    default_val_set_size: int = Field(default=10000, description="Default validation set size")
    default_test_set_size: int = Field(default=10000, description="Default test set size")
    default_samples: int = Field(default=10000, description="Default no. of samples to use to build preprocessors/tokenizers")
    configs_path: str = Field(default="../../../configs", description="Default path location for the hydra configurations")

    default_func_groups: Dict[str, Chem.Mol] = Field(
        default_factory = lambda : {
            "Acid anhydride": Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX3](=[OX1])"),
            "Acyl halide": Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]"),
            "Alcohol": Chem.MolFromSmarts("[#6][OX2H]"),
            "Aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6,H]"),
            "Alkane": Chem.MolFromSmarts("[CX4;H3,H2]"),
            "Alkene": Chem.MolFromSmarts("[CX3]=[CX3]"),
            "Alkyne": Chem.MolFromSmarts("[CX2]#[CX2]"),
            "Amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
            "Amine": Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]"),
            "Arene": Chem.MolFromSmarts("[cX3]1[cX3][cX3][cX3][cX3][cX3]1"),
            "Azo compound": Chem.MolFromSmarts("[#6][NX2]=[NX2][#6]"),
            "Carbamate": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2H0]"),
            "Carboxylic acid": Chem.MolFromSmarts("[CX3](=O)[OX2H]"),
            "Enamine": Chem.MolFromSmarts("[NX3][CX3]=[CX3]"),
            "Enol": Chem.MolFromSmarts("[OX2H][#6X3]=[#6]"),
            "Ester": Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
            "Ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
            "Haloalkane": Chem.MolFromSmarts("[#6][F,Cl,Br,I]"),
            "Hydrazine": Chem.MolFromSmarts("[NX3][NX3]"),
            "Hydrazone": Chem.MolFromSmarts("[NX3][NX2]=[#6]"),
            "Imide": Chem.MolFromSmarts("[CX3](=[OX1])[NX3][CX3](=[OX1])"),
            "Imine": Chem.MolFromSmarts(
                "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]"
            ),
            "Isocyanate": Chem.MolFromSmarts("[NX2]=[C]=[O]"),
            "Isothiocyanate": Chem.MolFromSmarts("[NX2]=[C]=[S]"),
            "Ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
            "Nitrile": Chem.MolFromSmarts("[NX1]#[CX2]"),
            "Phenol": Chem.MolFromSmarts("[OX2H][cX3]:[c]"),
            "Phosphine": Chem.MolFromSmarts("[PX3]"),
            "Sulfide": Chem.MolFromSmarts("[#16X2H0]"),
            "Sulfonamide": Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]"),
            "Sulfonate": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]"),
            "Sulfone": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[#6]"),
            "Sulfonic acid": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H]"),
            "Sulfoxide": Chem.MolFromSmarts("[#16X3]=[OX1]"),
            "Thial": Chem.MolFromSmarts("[CX3H1](=O)[#6,H]"),
            "Thioamide": Chem.MolFromSmarts("[NX3][CX3]=[SX1]"),
            "Thiol": Chem.MolFromSmarts("[#16X2H]"),
        }
    )

    @field_validator('default_func_groups', mode='before')
    @classmethod
    def parse_func_groups(cls, env_variable):
        # Validate default_func_groups
        if isinstance(env_variable, dict):
            # If all are Chem.Mol do nothing
            if np.all([isinstance(val, Chem.Mol) for val in env_variable.values()]):
                return env_variable
            # Try to convert to Chem.Mol
            else:
                try:
                    return {key: Chem.MolFromSmarts(val) for key, val in env_variable.items()}
                except Exception:
                    raise ValueError(f"Tried to convert func groups to Chem.Mol. Failed {env_variable}.")
        
        # Try loading from string
        elif isinstance(env_variable, str):
            import json
            try:
                data = json.loads(env_variable)
                return {key: Chem.MolFromSmarts(val) for key, val in data.items()}
            except Exception:
                raise ValueError(f"Tried to convert func groups to Chem.Mol. Failed {env_variable}.")
        else:
            raise ValueError(f"Can't handle variable {env_variable}.")


DEFAULT_SETTINGS = DefaultSettings()
