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

for modality in ir carbon hnmr multimodal; do

    mkdir -p ${run_folder}/singeltask/${modality}/
    python -m analytical_fm.cli.training \
        working_dir=${run_folder} \
        job_name=singeltask/${modality}/ \
        data_path=${data_folder} \
        data=multimodal/${modality} \
        model=${model} \
        molecules=True \
        trainer.epochs=60 \
        model.lr=${lr} \
        model.positional_encoding_type=${pos_enc} \
        model.gated_linear=${gated_linear} \
        model.optimiser=adamw 
done

