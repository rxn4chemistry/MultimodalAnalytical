#!/bin/bash

export HF_DATASETS_CACHE=/dccstor/lau_storage/hf_cache
export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.1/lib64
export TOKENIZERS_PARALLELISM=False

echo nvidia-smi

python ./src/cli/training.py \
    working_dir=/dccstor/lau_storage/msms_project/multimodal-bart/runs \
    job_name=msms_cfmid_P10 \
    data_path=/dccstor/lau_storage/msms_project/multimodal-bart/data \
    data=msms/msms_config \
    model=hf_bart_medium \
    molecules=True \
    trainer.epochs=60 \
    model.lr=5e-4 \