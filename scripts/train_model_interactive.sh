#!/bin/bash

export HF_DATASETS_CACHE=$1
export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.1/lib64
export TOKENIZERS_PARALLELISM=False


python ./src/cli/training.py \
    working_dir=$2 \
    job_name=msms_cfmid_P10 \
    data_path=$3 \
    data=msms/msms_config \
    model=hf_bart_medium \
    molecules=True \
    trainer.epochs=60 \
    model.lr=5e-4 \