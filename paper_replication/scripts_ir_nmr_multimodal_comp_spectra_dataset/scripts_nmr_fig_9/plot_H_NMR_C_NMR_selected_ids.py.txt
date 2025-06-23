#!/usr/bin/env python


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import Draw

# load the simulations from the dataset related to this git
# `NMR_data.parquet` dataset, an integral part of this work,
# is publicly available on Zenodo under the Community Data License Agreement Permissive 2.0:
# https://doi.org/10.5281/zenodo.15669241
# point to the "NMR_data.parquet" file
df_cpmd = pd.read_parquet("NMR_data.parquet")

# If available load a json file with experimental/reference data
# a sample is given in this folder "sample_exp_data.json"

# with open("sample_exp_data.json", "r") as fj:
#     data_exp = json.load(fj)
# df_exp = pd.DataFrame(data_exp)

# if data from exp are not available uncomment below
# is will create a dummy `df_exp` with no data
# [-999 is a place holder to allow rendering even without exp./reference data]
data_exp = [{
        "smiles": smiles,
        "h_nmr_peaks": [-999],
        "c_nmr_peaks": [-999]} for smiles in df_cpmd['smiles'].tolist()]

df_exp = pd.DataFrame(data_exp)


print("Original shapes:")
print(f"df_exp shape: {df_exp.shape}")
print(f"df_cpmd shape: {df_cpmd.shape}")

print("\n--- Processing Duplicate SMILES ---")
exp_smiles_duplicates_count = df_exp['smiles'].duplicated().sum()
cpmd_smiles_duplicates_count = df_cpmd['smiles'].duplicated().sum()

print(f"Number of duplicate SMILES in df_exp (before dropping): {exp_smiles_duplicates_count}")
print(f"Number of duplicate SMILES in df_cpmd (before dropping): {cpmd_smiles_duplicates_count}")

# Drop duplicates from df_exp based on 'smiles', keeping the first occurrence
if exp_smiles_duplicates_count > 0:
    df_exp_unique = df_exp.drop_duplicates(subset=['smiles'], keep='first').copy()
    print(f"df_exp shape after dropping duplicates: {df_exp_unique.shape}")
else:
    df_exp_unique = df_exp.copy() # No duplicates, so just copy

if cpmd_smiles_duplicates_count > 0:
    df_cpmd_unique = df_cpmd.drop_duplicates(subset=['smiles'], keep='first').copy()
    print(f"df_cpmd shape after dropping duplicates: {df_cpmd_unique.shape}")
else:
    df_cpmd_unique = df_cpmd.copy() # No duplicates, so just copy

df_combo = pd.merge(df_exp_unique, df_cpmd_unique, on='smiles', how='inner')

print("Shape of df_combo:", df_combo.shape)
print("Columns in df_combo:", df_combo.columns.tolist())
print("\nFirst 5 rows of df_combo:")
print(df_combo.head())

combo_smiles_duplicates = df_combo['smiles'].duplicated().sum()

# select 5 `selected_indices` to be displayed [5 is to allow easier visualization]
selected_indices = [0, 2, 10, 12, 14]
n_molecules = len(selected_indices)
fig, axes = plt.subplots(nrows=n_molecules, ncols=3, figsize=(14, 2.5 * n_molecules))

# x-axis range (shared)
x_min_h, x_max_h = 0, 12     # for H-NMR
x_min_c, x_max_c = 0, 220    # for 13C-NMR

h_all_peaks_global = []
c_all_peaks_global = []
for i, idx in enumerate(selected_indices):
    mol = df_combo.iloc[idx]
    smiles = mol['smiles']
    mol_obj = Chem.MolFromSmiles(smiles)
    mol_img = Draw.MolToImage(mol_obj, size=(200, 200))

    # Plot H-NMR (left)
    ax_h = axes[i, 0]
    for peak in mol['h_nmr_peaks']:
        ax_h.vlines(x=peak, ymin=0.5, ymax=1.0, color='red', linewidth=2)
    for peak in mol['averaged_frames']['h_nmr_peaks_grouped_frame_ave']:
        ax_h.vlines(x=peak, ymin=0.0, ymax=0.5, color='blue', linestyle='--', linewidth=1.5)
    ax_h.set_xlim(x_max_h, x_min_h)
    ax_h.set_ylim(0, 1)
    ax_h.set_yticks([])
    ax_h.tick_params(axis='x', labelsize=20)

    # Plot Molecule (center)
    ax_img = axes[i, 1]
    ax_img.imshow(mol_img)
    ax_img.axis('off')
    if i == 1:
        ax_img.text(0.5, -0.1, smiles, fontsize=8, ha='center', va='top', transform=ax_img.transAxes)
    else:
        ax_img.text(0.5, -0.1, smiles, fontsize=14, ha='center', va='top', transform=ax_img.transAxes)

    # Plot 13C-NMR (right)
    ax_c = axes[i, 2]
    for peak in mol['c_nmr_peaks']:
        ax_c.vlines(x=peak, ymin=0.5, ymax=1.0, color='red', linewidth=2)
    for peak in mol['averaged_frames']['c_nmr_peaks_grouped_frame_ave']:
        ax_c.vlines(x=peak, ymin=0.0, ymax=0.5, color='blue', linestyle='--', linewidth=1.5)
    ax_c.set_xlim(x_max_c, x_min_c)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([])
    ax_c.tick_params(axis='x', labelsize=20)

    h_all_peaks_global.extend(mol['h_nmr_peaks'])
    h_all_peaks_global.extend(mol['averaged_frames']['h_nmr_peaks_grouped_frame_ave'])

    c_all_peaks_global.extend(mol['c_nmr_peaks'])
    c_all_peaks_global.extend(mol['averaged_frames']['c_nmr_peaks_grouped_frame_ave'])

    if i == n_molecules - 1:
        ax_h.set_xlabel('Chemical Shift (ppm)', fontsize=24)
        ax_c.set_xlabel('Chemical Shift (ppm)', fontsize=24)

    if i == 0:
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Exp.'),
            Line2D([0], [0], color='blue', linestyle='--', lw=1.5, label='Comp.')
        ]
        ax_h.legend(handles=legend_elements, loc='upper left', fontsize=14,
            handlelength=1.0, handletextpad=0.5, borderpad=0.3)
        ax_c.legend(handles=legend_elements, loc='upper left', fontsize=14,
            handlelength=1.0, handletextpad=0.5, borderpad=0.3)

        ax_h.text(
           0.01, 1.08,
           "1H-NMR",
           transform=ax_h.transAxes,
           fontsize=20,
           # fontweight='bold',
           va='bottom'
        )
        ax_c.text(
           0.01, 1.08,
           "13C-NMR",
           transform=ax_c.transAxes,
           fontsize=20,
           # fontweight='bold',
           va='bottom'
        )



plt.tight_layout(h_pad=0.6)
# save to file a png
# plt.savefig(f"nmr_with_structure.png", dpi=600)
plt.show()

