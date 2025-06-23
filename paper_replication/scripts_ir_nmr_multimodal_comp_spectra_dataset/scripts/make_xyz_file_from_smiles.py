#!/usr/bin/env python
"""
make_xyz_file_from_smiles.py

This script reads a list of smiles and creates conf_**.xyz files.

Usage:
    python make_xyz_file_from_smiles.py --smiles <list_of_smiles.txt>

Arguments:
    --smiles      str : Path to the list of smiles in text format, one for each line
"""
import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def get_mol_from_smiles(smiles, id_):
    m1 = Chem.MolFromSmiles(smiles)
    m2 = Chem.AddHs(m1)
    AllChem.EmbedMolecule(m2,randomSeed=0xf00d)
    m2txt = Chem.MolToMolBlock(m2)
    if False:
        print(Chem.MolToMolBlock(m2),file=Path('./data/info_'+str(id_)+'.mol').open('w+'))
    natom = len([atom.GetSymbol() for atom in m2.GetAtoms()])
    total_charge = np.sum([atom.GetFormalCharge() for atom in m2.GetAtoms()])
    return m2, m2txt, natom, total_charge


def m2txt_to_dump_xyz(m2txt, id_, smiles, natom, charge):
    lines = m2txt.split('\n')
    #print('LINES', id_, smiles)
    #print(lines[0:5])
    filename_ = './conf_'+str(id_)+'.xyz'
    fo = Path(filename_).open('w')
    #natom = int(str(lines[3].split()[0]).strip())
    #print('NATOMS', natom)
    fo.write(str(natom)+'\n')
    fo.write(str(id_)+' '+str(smiles)+' charge: '+str(float(charge))+'\n')
    for i in range(4, 4+natom):
        tmp = lines[i].split()
        fo.write(tmp[3]+' '+tmp[0]+' '+tmp[1]+' '+tmp[2]+'\n')
    fo.close()
    return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--smiles',
        type=str,
        help=(
            'file_name of the list of smiles in text format, one smiles for each line'
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    path_smiles = args.smiles
    list_of_smiles = []
    with Path(path_smiles).open('r') as fin:
        for line in fin:
            list_of_smiles.append(str(line).strip())
    print('number of smiles:', len(list_of_smiles))
    print('uniq number of smiles:', len(set(list_of_smiles)))

    for id_, smiles in enumerate(list_of_smiles):
        m2, m2txt, natom, total_charge = get_mol_from_smiles(smiles, id_)
        m2txt_to_dump_xyz(m2txt, id_, smiles, natom, total_charge)

