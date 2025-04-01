#!/bin/bash

while getopts "r:d:a:c:p:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    c) ckpt_path="$OPTARG" ;;
    p) preprocessor_path="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

model=custom_model
lr=1e-3
pos_enc=learned
gated_linear=True
patch_size=75


for cv_split in 0 1 2 3 4; do

    mkdir -p ${run_folder}/finetune_augment_combined/cv_split_${cv_split}
    python -m analytical_fm.cli.training \
        working_dir=${run_folder} \
        job_name=finetune_augment_combined/cv_split_${cv_split} \
        finetuning=True \
        data_path=${data_folder} \
        data=ir/patches \
        data.IR.preprocessor_arguments.patch_size=${patch_size} \
        data.IR.preprocessor_arguments.interpolation=True \
        model=${model} \
        model.model_checkpoint_path=${ckpt_path} \
        model.lr=${lr} \
        model.positional_encoding_type=learned \
        model.gated_linear=True \
        preprocessor_path=${preprocessor_path} \
        molecules=True \
        trainer.epochs=30 \
        cv_split=${cv_split}

done
