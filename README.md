# Analytical Foundation Models

This repository contains the official implementation of the results shown in the work "Test-Time Tuned Language Models Enable End-to-end De Novo Molecular Structure Generation from MS/MS Spectra".

It provides the complete codebase needed to reproduce the results and train models on spectra obtained via MS/MS spectroscopy. The framework is build on PyTorch, PyTorch Lightning and Hugginface. To install it follow the instructions below.


## Installation
To install the code base ensure that you have at least Python 3.10 installed. Then follow the steps below. Recommended the use of `uv` package. 
Typically installation takes less than two minutes.

```
git clone https://github.com/rxn4chemistry/MultimodalAnalytical.git
cd MultimodalAnalytical

pip install uv
uv venv --python 3.10.16 .venv
uv pip install -r requirements.txt

uv pip install -e .
uv pip install -e .[dev]
```

## Usage
An example to train a model is provided in `scripts/train_model.sh`. The parameters present need to be changed according to the desired settings. To change the data for the training the config, column, modality, ... parameters need to be changed. As an example to change the column in the datafile the IR spectra are drawn from change the following parameters. However, we recommended to follow the instructions in the paper replication guides.
`data.IR.column=ir_spectra \`

## Replication
Complete instructions for reproducing the results presented are provided in the [paper_replication](paper_replication/msms) folder. These documents contains step-by-step guidance, including data preparation, model training parameters, and evaluation procedures to replicate our experiments. 
