# Setting New Benchmarks in AI-driven Infrared Structure Elucidation

## Overview

The following document contains the steps to reproduce the results in [Setting New Benchmarks in AI-driven Infrared Structure Elucidation](). All scripts were tested on Linux and MacOS.

<p align='center'>
  <img src='../figures/ir.png'>
</p>

## Abstract

Automated structure elucidation from infrared (IR) spectra represents a significant breakthrough in analytical chemistry, having recently gained momentum through the application of Transformer-based language models. In this work, we improve our original Transformer architecture, refine spectral data representations, and implement novel augmentation and decoding strategies to significantly increase performance. We report a Top–1 accuracy of 63.79% and a Top–10 accuracy of 83.95% compared to the current performance of state-of-the-art models of 53.56% and 80.36%, respectively. Our findings not only set a new performance benchmark but also strengthen confidence in the promising future of AI-driven IR spectroscopy as a practical and powerful tool for structure elucidation. To facilitate broad adoption among chemical laboratories and domain experts, we openly share our models and code.

## Prerequisites

To reproduce the the results you need to have this repo installed and the data used to train the models downloaded and processed. Installation of the codebase can be accomplished by following the steps in the [ReadMe](../../README.md). To download the data follow the steps below. All scripts are expected to be run from the directory `analytical_models/paper_replication/ir`.


### Data Download and Processing

We use two datasets to train the models: One pretraining dataset of synthetic <sup>31</sup>P-NMR spectra which we generated ourselves and the dataset by Hack et al. The synthetic dataset is available on [Zenodo](link_here) and the one by Hack et al. is availabe [here](https://github.com/clacor/Ilm-NMR-P31) (last accessed 25.02.2025).

To download and process the two datasets execute the following script:


```
./scripts/download_process_data.sh -o data/
```

This script downloads the experimental as well as the synthetic data and processes the experimental data into a format compatible with our models. At the same time the experimental data is filtered to remove molecules with more than 35 and less than 5 heavy atoms and duplicates are removed. The data is saved to the specified folder, in this case `data/`.


## Replicating Table 1

## Replicating Table 2

## Replicating Table 3

## Finetuning a pretrained model