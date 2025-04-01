from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors


def canonicalise(smiles: str) -> Optional[str]:
    RDLogger.DisableLog("rdApp.*") # type:ignore
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_env_smiles(smiles: str, radius: int) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'P':
            continue

        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx(), useHs=True)
        submol = Chem.PathToSubmol(mol, env)
        submol_smiles = Chem.MolToSmiles(submol)

        if submol_smiles == '':
            if radius == 1:
                return None
            submol_smiles = get_env_smiles(smiles, radius-1) # type:ignore

        return submol_smiles
    
    return None


@click.command()
@click.option('--data_path', type=Path, required=True)
@click.option('--output_path', type=Path, required=True)
def main(data_path: Path, output_path: Path) -> None:

    exp_data_raw = pd.read_csv(data_path, sep=' ')

    exp_data_proc = exp_data_raw[['shift', 'cansmi']].copy()

    # Canonicalise smiles, drop duplicates
    exp_data_proc['smiles'] = exp_data_proc['cansmi'].map(canonicalise)
    exp_data_proc = exp_data_proc.dropna(subset='smiles')
    exp_data_proc = exp_data_proc.drop_duplicates(subset='smiles')

    # Filter heavy atoms between 5 and 35
    hac = exp_data_proc['smiles'].map(lambda smiles : rdMolDescriptors.CalcNumHeavyAtoms(Chem.MolFromSmiles(smiles)))
    exp_data_proc = exp_data_proc[(hac >= 5) & (hac < 35)]

    # Make Chemical Formula, local smiles environment
    exp_data_proc['formula'] = exp_data_proc['smiles'].map(lambda smiles : rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles)))

    for i in range(1, 4):
        exp_data_proc[f'smiles_rad_{i}'] = exp_data_proc['smiles'].map(lambda smiles : get_env_smiles(smiles, i))
    exp_data_proc = exp_data_proc.dropna(subset=['smiles_rad_1', 'smiles_rad_2', 'smiles_rad_3'])

    exp_data_proc['phosphor_shift'] = exp_data_proc['shift'].map(lambda shift : [shift])

    exp_data_proc = exp_data_proc[['smiles', 'formula', 'phosphor_shift', 'smiles_rad_1', 'smiles_rad_2', 'smiles_rad_3']]
    exp_data_proc.to_parquet(output_path / 'hack_clean.parquet')


if __name__ == '__main__':
    main()
