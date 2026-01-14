#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False


mkdir -p ${run_folder}/from-scratch
    python -m analytical_fm.cli.training \
        working_dir=${run_folder} \
        job_name=from-scratch \
        data_path=${data_folder} \
        data=msms/text_fingerprint \
        model=custom_model_align \
        model.batch_size=16 \
        model.lr=1e-4 \
        trainer.epochs=60 \
        trainer.save_checkpoints=every_5_epochs \
        finetuning=False \
        molecules=True
done