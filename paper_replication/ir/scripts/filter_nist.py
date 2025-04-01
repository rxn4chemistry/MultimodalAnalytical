import re
from pathlib import Path

import click
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def canonicalise_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

@click.command()
@click.option("--data_path", type=Path, required=True)
@click.option("--smiles_column", type=str, default='smiles')
def main(data_path: Path, smiles_column: str):
    
    data = pd.read_parquet(data_path)

    # Remove any invalid smiles
    data[smiles_column] = data[smiles_column].map(canonicalise_smiles)
    data = data.dropna(subset=smiles_column)

    # Filter based on heavy atom count
    hac = data[smiles_column].map(lambda smiles : rdMolDescriptors.CalcNumHeavyAtoms(Chem.MolFromSmiles(smiles)))
    data = data[(hac > 5) & (hac < 14)]

    # Filter out any smiles consisting of more than one molecule
    filter_dot = data[smiles_column].map(lambda smiles : '.' not in smiles)
    data = data[filter_dot]

    # Filter out stereo molecule
    filter_dot = data[smiles_column].map(lambda smiles : '@' not in smiles)
    data = data[filter_dot]

    # Filter out any charged molecules
    filter_charged = data[smiles_column].map(lambda smiles : Chem.rdmolops.GetFormalCharge(Chem.MolFromSmiles(smiles)))
    data = data[filter_charged == 0]

    # Filter out any elements other than carbon, hydrogens, oxygen, nitrogen, sulfur, phosphorous and the halogens
    allowed_elements = set(['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
    molecular_formula = data[smiles_column].map(lambda smiles : rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles)))
    molecule_elements = [re.findall(r"[A-Z][a-z]?", formula) for formula in molecular_formula]
    filter_formula = [set(elements).issubset(allowed_elements) for elements in molecule_elements]
    data = data[filter_formula]

    # Save data
    save_path = data_path.parent / (data_path.name.replace(".parquet", '') + "_filtered.parquet")
    data.to_parquet(save_path)


if __name__ == '__main__':
    main()
