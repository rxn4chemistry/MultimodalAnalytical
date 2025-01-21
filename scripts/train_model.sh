#!/bin/bash
source .venv/bin/activate

export HF_DATASETS_CACHE=/dccstor/lau_storage/hf_cache
export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.1/lib64
export TOKENIZERS_PARALLELISM=False

modality=msms_config # .yaml file in src/config/data/msms/
model=hf_bart_medium
patch_size=250
lr=5e-4

top_dir=/dccstor/lau_storage/msms_project/multimodal-bart/runs

column="msms_cfmid_positive_10ev" 
exp_dir="msms_cfmid_p10"


echo "---------------- ${column} - ${exp_dir} ---------------- "

mkdir -p ${top_dir}/${exp_dir}
jbsub -queue x86_24h -cores 7+1 -mem 60g \
        -out ${top_dir}/${exp_dir}/out.txt \
        -err ${top_dir}/${exp_dir}/err.txt \
        python ./src/cli/training.py\
        working_dir=${top_dir} \
        data_path=/dccstor/lau_storage/msms_project/multimodal-bart/data \
        data=msms/${modality} \
        data.MSMS.column=${column} \
        model=${model} \
        molecules=True \
        trainer.epochs=60 \
        model.lr=${lr} \
        job_name=${exp_dir} \


deactivate

# For IR
#data=ir/patches  \
#data_path=/dccstor/ltlws3emb/analytical_data/data/simulated/IR \
#data.IR.preprocessor_arguments.patch_size=${patch_size} \

#For NMR
#data_path=/dccstor/ltlws3emb/analytical_data/data/simulated/1H_NMR \
#data=h_nmr/${modality} \
#data.Spectra.preprocessor_arguments.patch_size=${patch_size} \

#For CNMR
#data_path=/dccstor/ltlws3emb/analytical_data/data/simulated/13C_NMR \
#data=multimodal_analytical/${modality} \
#model.batch_size=64 \