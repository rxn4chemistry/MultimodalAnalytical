#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

mkdir -p ${run_folder}/pt
    python -m analytical_fm.cli.training_ttt \
        working_dir=${run_folder} \
        job_name=ttt \
        data_path=${data_folder} \
        data=msms/text_fingerprint \
        model=custom_model_align \
        model.model_checkpoint_path=${run_folder}/version_0/checkpoints/last.ckpt \
        preprocessor_path=${run_folder}/preprocessor.pkl \
        model.batch_size=128 \
        model.lr=5e-5 \
        trainer.epochs=60 \
        trainer.early_stopping_patience=10000 \
        trainer.save_checkpoints=best_5 \
        activeft=activeft \
        activeft.n_clusters=1000 \
        activeft.update_embeddings=20 \
        eval_ckpt=last \
        splitting=given_splits \
        molecules=True
done