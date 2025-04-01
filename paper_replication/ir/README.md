# Setting New Benchmarks in AI-driven Infrared Structure Elucidation

## Overview

The following document contains the steps to reproduce the results in [Setting New Benchmarks in AI-driven Infrared Structure Elucidation](). All scripts were tested on Linux and MacOS. The scripts below allow the pretraining of the models in the paper on the synthetic data in addition to downloading all synthetic data used as part of the paper as well as our best pretrained model. Due licencing of the experimental dataset we can not release the finetuned models but provide functionality to easily finetune them yourself.

<p align='center'>
  <img src='../figures/ir.png'>
</p>

## Abstract

Automated structure elucidation from infrared (IR) spectra represents a significant breakthrough in analytical chemistry, having recently gained momentum through the application of Transformer-based language models. In this work, we improve our original Transformer architecture, refine spectral data representations, and implement novel augmentation and decoding strategies to significantly increase performance. We report a Top–1 accuracy of 63.79% and a Top–10 accuracy of 83.95% compared to the current performance of state-of-the-art models of 53.56% and 80.36%, respectively. Our findings not only set a new performance benchmark but also strengthen confidence in the promising future of AI-driven IR spectroscopy as a practical and powerful tool for structure elucidation. To facilitate broad adoption among chemical laboratories and domain experts, we openly share our models and code.

## Prerequisites

To reproduce the the results you need to have this repo installed and the data used to train the models downloaded and processed. Installation of the codebase can be accomplished by following the steps in the [ReadMe](../../README.md). To download the data follow the steps below. All scripts are expected to be run from the directory `analytical_models/paper_replication/ir`.


### Data Downloading and Processing

A total of three datasets were used for this paper: Two synthetic ones containing simulated IR spectra and one experimental one. The synthetic datasets can be obtained by downloading them either from Zendodo ([Dataset 1](https://zenodo.org/records/14770232) and [Dataset 2](https://zenodo.org/records/7928396)) or using the script below.

To download and merge the two synthetic datasets:

```
./scripts/download_process_data.sh -o data/
```

This script downloads the synthetic data and processes it into a format compatible with our models. 

The experimental data used in this paper consists of the [NIST/EPA Gas-Phase Infrared Database](https://www.nist.gov/srd/nist-standard-reference-database-35). A license for the dataset can be obtained at the link above. We have supplied a [script](scripts/filter_nist.py) to filter the molecules from the dataset to produce the same finetuning set as used in our paper.

## Replicating Table 1

Table 1 contains ablations on different advances of the transformer architecture to replicate the results and pretrain the models on the synthetic data for this use the script `replicate_table_1.sh`. See below for usage:

```
./scripts/replicate_table_1.sh -r runs/ -d data/pretraining

-r: The folder in which the runs are saved
-d: The path to the training data
```

## Replicating Table 2

Table 2 evalutes the performance of the model when different patch sizes are used. The script pretrains a model with patch sizes 25, 50, 75, 100, 125 and 150. See below for usage:

```
./scripts/replicate_table_2.sh -r runs/ -d data/pretraining

-r: The folder in which the runs are saved
-d: The path to the training data
```

## Replicating Table 3

In Table 3 in the main paper we assess how different augmentation techniques impact the performance of the model. In total four different augmentations are evaluated in addition to the combination of all four. For pseudo-experimental spectra augmentation a path to the pseudo experimental spectra. If you downloaded the spectra with `download_process_data.sh` the default path is `data/pseudo_experimental`.

```
./scripts/replicate_table_2.sh -r runs/ -d data/pretraining -a data/pseudo_experimental

-r: The folder in which the runs are saved
-d: The path to the training data
-a: Path to the folder containing the pseudo experimental spectra
```

## Finetuning a pretrained model

All of the above scripts pretrain a model on the simulated data. To actually finetune a model we provide a script below. The script expects a path to a folder containing the finetuning data as a parquet file. The parquet file needs to contain at least three columns: `molecular_formula`, `smiles` and `ir_spectra`. The column `ir_spectra` needs to consist of lists of the ir spectra with each list containing 1791 values corresponding to a range of 400 to 3982cm<sup>-1</sup> with a resolution of 2cm<sup>-1</sup>. During training only the part of the spectrum from 650 to 3900cm<sup>-1</sup> is used. The checkpoint as well as preprocessors for our best pretrained model are available [here](https://zenodo.org/records/15116374). Use the below script to perform five-fold cross validated finetuning:

```
./scripts/replicate_table_2.sh -r runs/ -d data/pretraining -a data/pseudo_experimental

-r: The folder in which the runs are saved
-d: The path to the training data
-c: Path to the model checkpoint
-p: Path to the preprocessor 
```