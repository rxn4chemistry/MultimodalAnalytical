# Analytical Foundation Models

This repository contains the official implementation of the research presented in ["From Spectra to Structure: AI-Powered <sup>31</sup>P-NMR Interpretation"]() as well as the addendum to ["Leveraging infrared spectroscopy for automated structure elucidation"](https://www.nature.com/articles/s42004-024-01341-w). It provides the complete codebase needed to reproduce our results and train models on <sup>31</sup>P-NMR spectra and IR spectra. The framework is build on PyTorch, PyTorch Lightning and Hugginface. To install it follow the instructions below.

## Installation
To install the code base ensure that you have at least Python 3.10 installed. Then follow the steps below:

```
pip install uv
uv pip install -e .
uv pip install -e .[dev]
```

## Usage
To run the code all the paths and parameters in `scripts/train_model.sh` need to be changed accordingly.
Especially, to change the data for the training the config, column, modality, ... parameters need to be changed.
Ex.  
`data.IR.column=ir_spectra \`

## Replication
Complete instructions for reproducing the results presented in our papers are provided in the [papers](paper_replication/) folder. These documents contains step-by-step guidance, including data preparation, model training parameters, and evaluation procedures to replicate our experiments.