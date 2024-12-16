#!/bin/bash

export HF_DATASETS_CACHE=/dccstor/ltlws3emb/cache/hf_cache
export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64
export TOKENIZERS_PARALLELISM=False

modality=hnmr_vector
model=hf_bart_medium
patch_size=250
lr=5e-4

top_dir=/dccstor/ltlws3emb/multimodal_bart/runs/final_results
exp_dir=train_HFModel_customBeam_${modality}_${lr}_repeated

mkdir -p ${top_dir}/${exp_dir}
jbsub -queue x86_24h -cores 7+1 -mem 60g \
        -out ${top_dir}/${exp_dir}/out.txt \
        -err ${top_dir}/${exp_dir}/err.txt \
        python /dccstor/ltlws3emb/multimodal_bart/src/multimodal/cli/train_multimodal.py\
        job_name=${exp_dir} \
        data_path=/dccstor/ltlws3emb/analytical_data/data/simulated/1H_NMR \
        data=h_nmr/${modality} \
        model=${model} \
        molecules=True \
        trainer.epochs=60 \
        model.lr=${lr} \

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