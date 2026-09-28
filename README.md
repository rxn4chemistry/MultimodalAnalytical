# Analytical Foundation Models

This repository contains the official implementation of the results shown in the work "Test-Time Tuned Language Models Enable End-to-end De Novo Molecular Structure Generation from MS/MS Spectra".

It provides the complete codebase needed to reproduce the results and train models on spectra obtained via MS/MS spectroscopy. The framework is built on PyTorch, PyTorch Lightning and Hugging Face. To install it follow the instructions below.


## Installation
To install the code base ensure that you have at least Python 3.10 installed. Then follow the steps below. We recommend using `uv`.
Typically installation takes less than two minutes.

```
git clone -b ttt-msms https://github.com/rxn4chemistry/MultimodalAnalytical.git
cd MultimodalAnalytical

pip install uv
uv venv --python 3.10.16 .venv
uv pip install -r requirements.txt   # installs this package in editable mode
uv pip install -e ".[dev]"          # optional, development tools
```

## Usage
Training and evaluation are run with the command line entry points `analytical_fm.cli.training`, `analytical_fm.cli.training_ttt` (test-time tuning) and `analytical_fm.cli.predict`, configured with [Hydra](https://hydra.cc/) (configs in `configs/`). Ready-to-use scripts for every step of the pipeline and the pre-trained checkpoints ([Hugging Face](https://huggingface.co/laura-mismetti/ttt-msms), [Zenodo](https://zenodo.org/records/22961942)) are described in the replication guide below.

## Replication
Complete instructions for reproducing the results presented are provided in the [paper_replication](paper_replication/msms) folder. It contains step-by-step guidance, including data preparation, model training parameters, and evaluation procedures to replicate our experiments. 
