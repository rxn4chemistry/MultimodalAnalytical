#!/bin/bash

export HF_DATASETS_CACHE=$1
export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64
export TOKENIZERS_PARALLELISM=False
export HYDRA_FULL_ERROR=1

modality=patches
model=custom_bart
patch_size=125

top_dir=$2
exp_dir=$3
data_path=$4
checkpoint_path=$5
preprocessor_path=$6



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

