#!/bin/bash

export TOKENIZERS_PARALLELISM=False
export HYDRA_FULL_ERROR=1

modality=patches
model=custom_bart
patch_size=125

top_dir=$1
exp_dir=$2
data_path=$3
checkpoint_path=$4
preprocessor_path=$5



mkdir -p ${top_dir}/${exp_dir}
jbsub -queue x86_6h -cores 4+1 -mem 30g \
        -out ${top_dir}/${exp_dir}/out.txt \
        -err ${top_dir}/${exp_dir}/err.txt \
        python ./src/analytical_fm/modeling/cli/predict.py\
        job_name=${exp_dir} \
        data=multimodal_analytical/${modality} \
        data_path=${data_path} \
        model.model_checkpoint_path=${checkpoint_path} \
        preprocessor_path=${preprocessor_path}
        model=${model} \
        molecules=True \

