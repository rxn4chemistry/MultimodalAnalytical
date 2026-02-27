# IR Mixture paper

## Overview

The following document contains the steps to reproduce the results in [IR mixture](). All scripts were tested on Linux and MacOS. The scripts below allow the pretraining of the models in the paper on the synthetic data in addition to downloading all synthetic data used as part of the paper as well as our best pretrained model. Due licencing of the experimental dataset we can not release the finetuned models but provide functionality to easily finetune them yourself.

<p align='center'>
  <img src='../figures/mixture.png'>
</p>

## Abstract

The application of Transformer-based language models to structure elucidation represents a breakthrough in analytical chemistry. While these advances have enabled direct molecular structure prediction from IR spectra, current methods are constrained by their requirement for the spectra of pure  compounds spectra. In this work, we address this constraint by developing a language model-driven approach that directly predicts individual molecular components from IR spectra of mixtures, expanding the practical applicability of AI-assisted spectroscopic analysis tools.
On binary balanced mixtures, our model achieves a Top-10 accuracy of up to 61.4%. Validation on 15 experimentally measured mixtures demonstrates robust transferability, maintaining a 44.0% Top-10 accuracy despite significant instrumental differences between training (gas-phase IR spectra) and test (Attenuated Reflectance spectra) data. Our models and code are openly available, facilitating adoption in chemical laboratories, with the goal to advance the analysis and interpretation of IR spectra.

## Prerequisites

To reproduce the the results you need to have this repo installed and the data used to train the models downloaded and processed. Installation of the codebase can be accomplished by following the steps in the [ReadMe](../../README.md). To download the data follow the steps below. All scripts are expected to be run from the directory `MultimodalAnalytical/`.


### Data Downloading and Processing

A total of two datasets were used for this paper: One synthetic data containing simulated IR spectra and one experimental one. The synthetic dataset can be obtained by downloading them either from Zendodo ([simulated_data](https://zenodo.org/records/14770232)) or using the script below.

To download the synthetic dataset:

```bash
./scripts/download_process_data.sh -o data/
```

This script downloads the synthetic data and processes it into a format compatible with our models. 

The experimental data used in this paper consists of the [NIST/EPA Gas-Phase Infrared Database](https://www.nist.gov/srd/nist-standard-reference-database-35). A license for the dataset can be obtained at the link above. We have supplied a [script](../ir/scripts/filter_nist.py) to filter the molecules from the dataset to produce the same finetuning set as used in our paper.


## Replicating Table 1

Running this scripts replicates the ablation study found in Table 1.  
We evaluated the impat of 3 factors:

- reconstruction_network: *convolutional* and *mlp*.
- reconstruction_loss: *mae* and *mse*.
- lambda: *1*, *5* and *50*.

The scripts **pretrains** the models on the synthetic data and **fine-tunes** on real data for *binary balanced mixtures*. Use the script `replicate_table_1.sh`. See below for usage:

```bash
./scripts/replicate_table_1.sh -r runs/ -p pretrain_data -f finetune_data 

-r: The folder in which the runs are saved
-p: The path to the pretraining data
-f: The path to the finetuning data
```


## Replicating Table 2 and 3

To replicate the results in Table 2 or 3 use the script `replicate_table_2_and_3.sh`. See below for usage:

```bash
./scripts/replicate_table_2_and_3.sh -r runs/ -p pretrain_data -f finetune_data -t multitask_w_pure

-r: The folder in which the runs are saved
-p: The path to the pretraining data
-f: The path to the finetuning data
-t: Which task to run. To replicate the results in Table 2 use multitask_w_pure for Table 3 use ternary
```

## Replicating Table 4

Table 4 shows our multitask models capability, when using the best configuration for the **encoder alignment**, on predicting components from *real mixtures*. The evaluation is done with and without rejection sampling, i.e. rejecting samples at evaluation time when the prediction does not match with the target chemical formula. To run the evaluation use the script `replicate_table_4.sh`. See below for usage:

```
./scripts/replicate_table_8.sh -r runs/ -d data

-r: The folder in which the runs are saved
-d: The path to the training data
```

where `data` should contain the real data.

