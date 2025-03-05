#!/bin/bash

export TOKENIZERS_PARALLELISM=False

model=custom_model
patch_size=250
lr=1.e-3

top_dir=$1
exp_dir=$2
data_path=$3


mkdir -p ${top_dir}/${exp_dir}
jbsub -queue x86_24h -cores 7+1 -mem 60g \
        -out ${top_dir}/${exp_dir}/out.txt \
        -err ${top_dir}/${exp_dir}/err.txt \
        python ./src/cli/training.py\
        working_dir=${top_dir} \
        data_path=${data_path} \
        data=ir/patches \
        model=${model} \
        molecules=True \
        trainer.epochs=60 \
        model.lr=${lr} \
        job_name=${exp_dir}\