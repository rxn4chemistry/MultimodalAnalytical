# Plotting Simulated vs Experimental NMR Spectra

This script visualizes **simulated vs experimental (or reference)** NMR spectra for a small set of molecules using their SMILES representation.

## Requirements

- Python ≥ 3.7
- `rdkit`
- `pandas`
- `matplotlib`

## Input Data

### 1. `NMR_data.parquet`

This file contains the **simulated CPMD-derived** NMR data and must be present in the folder. It is publicly available:

> DOI: [10.5281/zenodo.15669241](https://doi.org/10.5281/zenodo.15669241)
> License: Community Data License Agreement Permissive 2.0

Download and place the file as:

```bash
NMR_data.parquet
```

### 2. `sample_exp_data.json` (optional)

If you have **experimental or reference** data, place it as:

```bash
sample_exp_data.json
```

You can load it by uncommenting the appropriate lines in `plot_H_NMR_C_NMR_selected_ids.py`.

If you don’t have this file, the script creates dummy entries with placeholder peak values (`-999`) so the visualization still works.

## Usage

### Option 1: Run the Python script

```bash
python plot_H_NMR_C_NMR_selected_ids.py
```

### Option 2: Use the Jupyter notebook

A Jupyter version of this script can also be run interactively for customization.

## Output

The script generates a **3-column plot** for 5 selected molecules:

1. **1H-NMR peaks**: Experimental (red) and Computed (blue, dashed)
2. **Molecular structure**
3. **13C-NMR peaks**: Experimental (red) and Computed (blue, dashed)

You can optionally save the output as a PNG by uncommenting the save line at the end of the script.

