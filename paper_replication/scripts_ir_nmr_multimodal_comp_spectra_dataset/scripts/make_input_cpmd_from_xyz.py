#!/usr/bin/env python
"""
make_input_cpmd_from_xyz.py

This script prepares input files for CPMD to perform a geometry optimization
and/or NMR computation on a molecule provided in an XYZ file format.
Note: the geometry optimization uses BFGS, so it may fail to converge.
In most of the cases it worked faster, but CG may be more robust.

Usage:
    python make_input_cpmd_from_xyz.py --filexyz <input_file.xyz> [--do_geop] [--do_nmr]

Arguments:
    --filexyz      str : Path to the input XYZ file containing molecular geometry.
    --do_geop      str : yes/no - include this string to prepare input for geometry optimization (default: yes).
    --do_nmr       str : yes/no - include this string to prepare input for NMR computation (default: yes).
    --do_both      str : yes/no - set both geometry optimization and NMR computation (default: no).
    
Description:
    This script reads the molecular geometry from the specified XYZ file and prepares 
    CPMD input files for either geometry optimization, NMR computation, or both, 
    depending on the flags provided. It generates the necessary CPMD directives 
    and configurations based on user input.

Examples:
    1. Generate CPMD input for geometry optimization only:
       python make_input_cpmd_from_xyz.py --filexyz molecule.xyz [--do_geop yes] --do_nmr no

    2. Generate CPMD input for NMR computation only:
       python make_input_cpmd_from_xyz.py --filexyz molecule.xyz --do_geop no [--do_nmr yes]

    3. Generate CPMD input for both geometry optimization and NMR:
       python make_input_cpmd_from_xyz.py --filexyz molecule.xyz --do_both yes

"""

import argparse
import math
import numpy as np

do_geop = True
do_nmr = True
do_both = False

cpmd_input_GEOP = '''&CPMD
OPTIMIZE GEOMETRY
CONVERGENCE ORBITALS
1.0E-6
PCG MINIMIZE
MAXSTEPS
1000
PRINT FORCES ON
PRINT
100
&END

&SYSTEM
  ANGSTROM
  SYMMETRY
    1
  CELL
  A_CELL 1.0 1.0  0.0 0.0 0.0
  CUTOFF
    100.
&END

&DFT
 FUNCTIONAL PBE
&END

&ATOMS
'''


cpmd_input_LR = '''&CPMD
LINEAR RESPONSE
restart wavefunction coordinates latest
CONVERGENCE ORBITALS
1.d-6
PCG MINIMIZE
&END

&SYSTEM
  ANGSTROM
  SYMMETRY
    1
  CELL
  A_CELL 1.0 1.0  0.0 0.0 0.0
  CUTOFF
    100.
&END

&RESP
 NMR
 CONVERGENCE
 1.d-6
 OVERLAP 
 0.1
 CURRENT
 PSI0
 RHO0
&END

&DFT
 FUNCTIONAL PBE
&END

&ATOMS
'''

cpmd_input_2 = '''
&END

'''

pseudo = {
 'Ag': 'Ag-q11-pbe',
 'Al': 'Al-q3-pbe',
 'Ar': 'Ar-q8-pbe',
 'As': 'As-q5-pbe',
 'At': 'At-q7-pbe',
 'Au': 'Au-q19-pbe',
 'B': 'B-q3-pbe',
 'Ba': 'Ba-q10-pbe',
 'Be': 'Be-q4-pbe',
 'Bi': 'Bi-q5-pbe',
 'Br': 'Br-q7-pbe',
 'C': 'C-q4-pbe',
 'Ca': 'Ca-q10-pbe',
 'Cd': 'Cd-q12-pbe',
 'Cl': 'Cl-q7-pbe',
 'Co': 'Co-q17-pbe',
 'Cr': 'Cr-q14-pbe',
 'Cs': 'Cs-q9-pbe',
 'Cu': 'Cu-q11-pbe',
 'F': 'F-q7-pbe',
 'Fe': 'Fe-q16-pbe',
 'Ga': 'Ga-q13-pbe',
 'Ge': 'Ge-q4-pbe',
 'H': 'H-q1-pbe',
 'He': 'He-q2-pbe',
 'Hf': 'Hf-q12-pbe',
 'Hg': 'Hg-q12-pbe',
 'I': 'I-q7-pbe',
 'In': 'In-q13-pbe',
 'Ir': 'Ir-q17-pbe',
 'K': 'K-q9-pbe',
 'Kr': 'Kr-q8-pbe',
 'La': 'La-q11-pbe',
 'Li': 'Li-q3-pbe',
 'Mg': 'Mg-q10-pbe',
 'Mn': 'Mn-q15-pbe',
 'Mo': 'Mo-q14-pbe',
 'N': 'N-q5-pbe',
 'Na': 'Na-q9-pbe',
 'Nb': 'Nb-q13-pbe',
 'Ne': 'Ne-q8-pbe',
 'Ni': 'Ni-q18-pbe',
 'O': 'O-q6-pbe',
 'Os': 'Os-q16-pbe',
 'P': 'P-q5-pbe',
 'Pb': 'Pb-q4-pbe',
 'Pd': 'Pd-q18-pbe',
 'Po': 'Po-q6-pbe',
 'Pt': 'Pt-q18-pbe',
 'Rb': 'Rb-q9-pbe',
 'Re': 'Re-q15-pbe',
 'Rh': 'Rh-q17-pbe',
 'Rn': 'Rn-q8-pbe',
 'Ru': 'Ru-q16-pbe',
 'S': 'S-q6-pbe',
 'Sb': 'Sb-q5-pbe',
 'Sc': 'Sc-q11-pbe',
 'Se': 'Se-q6-pbe',
 'Si': 'Si-q4-pbe',
 'Sn': 'Sn-q4-pbe',
 'Sr': 'Sr-q10-pbe',
 'Ta': 'Ta-q13-pbe',
 'Tc': 'Tc-q15-pbe',
 'Te': 'Te-q6-pbe',
 'Ti': 'Ti-q12-pbe',
 'Tl': 'Tl-q13-pbe',
 'V': 'V-q13-pbe',
 'W': 'W-q14-pbe',
 'Xe': 'Xe-q8-pbe',
 'Y': 'Y-q11-pbe',
 'Zn': 'Zn-q12-pbe',
 'Zr': 'Zr-q12-pbe'
}


def make_atoms(atoms, pos):
    input_atoms = ''
    n = len(pos)
    list_set_atoms = sorted(list(set(atoms)))
    count_atoms = {}
    for ato in list_set_atoms:
        count_atoms[ato] = atoms.count(ato)
        input_atoms = input_atoms + '*' + pseudo[ato] + '\n'
        input_atoms = input_atoms + ' LMAX=S'  + '\n'
        input_atoms = input_atoms + ' ' + str(count_atoms[ato]) + '\n'
        for i in range(n):
            atom = atoms[i]
            if ato == atom:
                input_atoms = input_atoms+' '+f"{pos[i,0]:12.6f} {pos[i,1]:12.6f} {pos[i,2]:12.6f}\n"
        input_atoms = input_atoms+'\n'
    print('count_atoms:', count_atoms)
    return input_atoms


def round_up_to_nearest_5(number):
    return float(math.ceil(number / 5) * 5)


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filexyz',
        type=str,
        help=(
            'name of the input xyz file '
        )
    )

    parser.add_argument(
        '--do_geop',
        type=str,
        help='str: `yes` or `no`; set this flag to write cpmd input file for the geom_opt str',
        default="yes"
    )

    parser.add_argument(
        '--do_nmr',
        type=str,
        help=(
            'str: `yes` or `no`; set this flag to write cpmd input file for the nmr with LR'
        ),
        default="yes"
    )

    parser.add_argument(
        '--do_both',
        type=str,
        help=(
            '''str: `yes` or `no`; set this flag to run geop. and nmr in the same pod one after each other;
               the nmr input enable RESTART from LATEST which needs to be present.'''
        ),
        default="no"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    filexyz = args.filexyz

    if args.do_geop == "no":
        make_geop = False
    else:
        make_geop = True
    
    if args.do_nmr == "no":
        make_nmr = False
    else:
        make_nmr = True
    
    if args.do_both == "yes":
        run_both = True
        make_nmr = True
        make_geop = True
    else:
        run_both = False
    
    with open(filexyz, 'r') as fxyz:
        atoms = []
        pos = []
        lines = fxyz.readlines()
        natom = int(lines[0].split()[0])
        # print(filexyz, "natom in filexyz:", natom)
        atoms = [lines[i].split()[0] for i in range(2,2+natom)]
        pos = [[float(lines[i].split()[1]),
                float(lines[i].split()[2]),
                float(lines[i].split()[3])] for i in range(2,2+natom)]
        # print(atoms, pos)
        list_set_atoms = sorted(list(set(atoms)))
        count_atoms = {}
        for ato in list_set_atoms:
            count_atoms[ato] = atoms.count(ato)
        print(filexyz, 'natom:', natom, '/ count_atoms:', count_atoms)
        # print(filexyz, "set of atoms", list_set_atoms)
    pos_np = np.array(pos)
    box_min = np.max(pos_np, axis=0) - np.min(pos_np, axis=0)
    print(filexyz, 'box_min', box_min)
    lato_ = float(int(np.max(box_min) + 10.0))
    lato = round_up_to_nearest_5(lato_)
    print(filexyz, 'cubic size rounded (next 5Å)', lato, "larger", lato_)
    box = np.array([lato, lato, lato])
    rcm = np.mean(pos_np, axis=0)
    # print('rcm', rcm)
    shift = box/2.0 - rcm
    new_pos = pos_np + shift

    if False:
        # write shifted molecules on xyz file
        filexyz_new = filexyz.replace('.xyz', '_new_pos.xyz')
        with open(filexyz_new, 'w') as fout:
            fout.write(str(natom)+'\n')
            fout.write('BOX '+str(box[0])+' '+str(box[1])+' '+str(box[2])+'\n')
            for i in range(natom):
                fout.write(atoms[i]+' '+str(new_pos[i,0])+' '+str(new_pos[i,1])+' '+str(new_pos[i,2])+'\n')
    input_atoms = make_atoms(atoms, new_pos)

    if make_geop:
        # write GEOP cpmd
        filecpmd = filexyz.replace('.xyz', '_geop_cpmd.in')
        with open(filecpmd, 'w') as fout:
            fout.write(cpmd_input_GEOP.replace('A_CELL', str(lato)))
            fout.write(input_atoms)
            fout.write(cpmd_input_2)

    if make_nmr:
        # write LR for NMR cpmd
        filecpmd = filexyz.replace('.xyz', '_nmr_cpmd.in')
        if run_both:
            print("INFO: run_both: NMR input expects to find a CPMD RESTART file and LATEST")
            str1 = "restart wavefunction coordinates latest"
            str2 = "RESTART WAVEFUNCTION COORDINATES LATEST"
            cpmd_input_LR.replace(str1, str2)
        with open(filecpmd, 'w') as fout:
            fout.write(cpmd_input_LR.replace('A_CELL', str(lato)))
            fout.write(input_atoms)
            fout.write(cpmd_input_2)

