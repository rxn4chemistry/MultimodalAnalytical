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
patch_size=125

for pos_enc in learned sin_cos; do
    for gated_linear in True False; do

        mkdir -p ${run_folder}/pos_enc_ablation/pos_enc_${pos_enc}/ir_ps_${patch_size}_gated_linear_${gated_linear}
        python -m analytical_fm.cli.training \
            working_dir=${run_folder} \
            job_name=pos_enc_ablation/pos_enc_${pos_enc}/ir_ps_${patch_size}_gated_linear_${gated_linear} \
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
done
