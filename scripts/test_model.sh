#!/bin/bash

export HF_DATASETS_CACHE=$1
export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64
export TOKENIZERS_PARALLELISM=False
export HYDRA_FULL_ERROR=1

modality=carbon
model=custom_hf_bart
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
        python ./src/mmbart/modeling/cli/predict.py\
        job_name=${exp_dir} \
        data=multimodal_analytical/${modality} \
        data_path=${data_path} \
        model.model_checkpoint_path=${checkpoint_path} \
        preprocessor_path=${preprocessor_path}
        model=${model} \
        molecules=True \


# For IR
#data=ir/patches  \
#data_path=/dccstor/ltlws3emb/analytical_data/data/simulated/IR \
#data.IR.preprocessor_arguments.patch_size=${patch_size} \
#model.model_checkpoint_path=/dccstor/ltlws3emb/multimodal_bart/runs/train_custom_residualfirst_ir_5e-4/version_0/checkpoints/epoch_59-step_156840.ckpt \
#model.model_checkpoint_path=/dccstor/ltlws3emb/multimodal_bart/runs/train_original_5e-4/version_0/checkpoints/last.ckpt \

#For HNMR
#data_path=/dccstor/ltlws3emb/analytical_data/data/simulated/1H_NMR \
#data=h_nmr/${modality} \
#data.Spectra.preprocessor_arguments.patch_size=${patch_size} \
#model.model_checkpoint_path=/dccstor/ltlws3emb/multimodal_bart/runs/train_custom_residualfirst_hnmr_annotated_1e-3/version_0/checkpoints/last.ckpt \
#model.model_checkpoint_path=/dccstor/ltlws3emb/multimodal_bart/runs/train_custom_residualfirst_hnmr_vector_1e-3/version_0/checkpoints/last.ckpt \

#For CNMR
#data_path=/dccstor/ltlws3emb/analytical_data/data/simulated/13C_NMR \
#data=multimodal_analytical/${modality} \
