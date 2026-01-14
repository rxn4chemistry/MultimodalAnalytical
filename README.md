# Analytical Foundation Models

This repository contains the official implementation of the results shown in:
- ["From Spectra to Structure: AI-Powered <sup>31</sup>P-NMR Interpretation"](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00131e)
- ["Setting New Benchmarks in AI-driven Infrared Structure Elucidation"](https://pubs.acs.org/doi/10.1021/acs.analchem.5c01460)
- ["Automated Structure Elucidation at Human-Level Accuracy via a Multimodal Multitask Language Model"](https://chemrxiv.org/engage/chemrxiv/article-details/682eccb7c1cb1ecda0b3c633)
- ["Language Model Enabled Structure Prediction from Infrared Spectra of Mixtures"](https://chemrxiv.org/engage/chemrxiv/article-details/686249a91a8f9bdab5bfefee)
- ["IR–NMR Multimodal Computational Spectra Dataset for 177K Patent-Extracted Organic Molecules"](https://chemrxiv.org/engage/chemrxiv/article-details/684f1f86c1cb1ecda0230ceb)
- ["Test-Time Tuned Language Models Enable End-to-end De Novo Molecular Structure Generation from MS/MS Spectra"](https://arxiv.org/abs/2510.23746)
  
It provides the complete codebase needed to reproduce our results and train models on spectra obtained via IR, NMR and MS/MS spectroscopy. The framework is build on PyTorch, PyTorch Lightning and Hugginface. To install it follow the instructions below.


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
Complete instructions for reproducing the results presented in our papers are provided in the [papers](paper_replication/) folder. These documents contains step-by-step guidance, including data preparation, model training parameters, and evaluation procedures to replicate our experiments. 

- Phosphor: [here](paper_replication/phosphor)
- IR: [here](paper_replication/ir)
- Mixtures: [here](paper_replication/mixture)
- Multimodal: [here](paper_replication/multimodal)
- Dataset: [here](paper_replication/scripts_ir_nmr_multimodal_comp_spectra_dataset)
- MS/MS: [here](paper_replication/msms)
