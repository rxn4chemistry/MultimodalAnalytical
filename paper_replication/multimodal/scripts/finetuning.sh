#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

model=custom_model
lr=1e-3

pos_enc=learned
gated_linear=True

for cv_split in 0 1 2 3 4; do

    mkdir -p ${run_folder}/finetuning/cv_split_${cv_split}
    python -m analytical_fm.cli.training \
        working_dir=${run_folder} \
        job_name=finetuning/cv_split_${cv_split} \
        data_path=${data_folder} \
        data=multimodal/multimodal \
        model=${model} \
        molecules=True \
        trainer.epochs=90 \
        model.lr=${lr} \
        model.positional_encoding_type=${pos_enc} \
        model.gated_linear=${gated_linear} \
        model.optimiser=adamw \
        model.model_checkpoint_path=${top_dir}/version_0/checkpoints/last.ckpt \
        preprocessor_path=${top_dir}/preprocessor.pkl \
        finetuning=True \
        cv_split=${cv_split} \
        splitting=unique_target \
        modality_dropout=[IR,Multiplets,Carbon]

done