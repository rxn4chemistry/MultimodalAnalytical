#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False


mkdir -p ${run_folder}/eval

python -m analytical_fm.cli.predict \
    working_dir=${run_folder} \
    job_name=eval \
    data_path=${data_folder} \
    data=msms/text_fingerprint \
    model=custom_model_align \
    model.batch_size=64 \
    model.model_checkpoint_path=${run_folder}/version_0/checkpoints/best.ckpt \
    preprocessor_path=${run_folder}/preprocessor.pkl \
    splitting=given_splits \
    molecules=True \