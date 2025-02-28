#!/bin/bash

export HF_DATASETS_CACHE=$1
export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.1/lib64
export TOKENIZERS_PARALLELISM=False

top_dir=$2
exp_dir=$3
data_path=$4


python ./src/cli/training.py \
    working_dir=${top_dir} \
    job_name=${exp_dir} \
    data_path=${data_path} \
    data=ir/patches \
    model=custom_model \
    molecules=True \
    trainer.epochs=60 \
    model.lr=5e-4 \