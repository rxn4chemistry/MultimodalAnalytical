#!/bin/bash

export TOKENIZERS_PARALLELISM=False

top_dir=$1
exp_dir=$2
data_path=$3


python ./src/cli/training.py \
    working_dir=${top_dir} \
    job_name=${exp_dir} \
    data_path=${data_path} \
    data=ir/patches \
    model=custom_model \
    molecules=True \
    trainer.epochs=60 \
    model.lr=5e-4 \