#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

mkdir -p ${run_folder}/ft

python -m analytical_fm.cli.training \
    working_dir=${run_folder} \
    job_name=ft \
    data_path=${data_folder} \
    data=msms/text_fingerprint \
    model=custom_model_align \
    model.model_checkpoint_path=${run_folder}/version_0/checkpoints/last.ckpt \
    preprocessor_path=${run_folder}/preprocessor.pkl \
    model.batch_size=32 \
    model.lr=5e-5 \
    trainer.epochs=60 \
    finetuning=True \
    splitting=given_splits \
    molecules=True
