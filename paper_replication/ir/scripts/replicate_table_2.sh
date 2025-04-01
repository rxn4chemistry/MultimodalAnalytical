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

for patch_size in 25 50 75 100 125 150; do

    mkdir -p ${run_folder}/patch_size_ablation/patch_size_${patch_size}
    python -m analytical_fm.cli.training \
            working_dir=${run_folder} \
            job_name=patch_size_ablation/patch_size_${patch_size} \
            data_path=${data_folder} \
            data=ir/patches \
            data.IR.preprocessor_arguments.patch_size=${patch_size} \
            data.IR.preprocessor_arguments.interpolation=True \
            data.Formula.column=molecular_formula \
            model=${model} \
            molecules=True \
            trainer.epochs=60 \
            model.lr=${lr} \
            model.positional_encoding_type=${pos_enc} \
            model.gated_linear=${gated_linear} \
            model.optimiser=adamw


done
