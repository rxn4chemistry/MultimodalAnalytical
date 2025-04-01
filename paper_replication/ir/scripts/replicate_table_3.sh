#!/bin/bash

while getopts "r:d:a:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    a) augment_path="$OPTARG"
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

model=custom_model
lr=1e-3
pos_enc=learned
gated_linear=True
patch_size=75

for augment in smooth horizontal smiles pseudo combined; do

    if [ "$augment" == "pseudo" ] || [ "$augment" == "combined" ]; then
        used_augment_path=$augment_path
    else
        used_augment_path=
    fi

    mkdir -p ${run_folder}/augmentations/${augment}
    python -m analytical_fm.cli.training \
            working_dir=${run_folder} \
            job_name=augmentations/${augment} \
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
            model.optimiser=adamw \
            augment=ir/${augment} \
            augment.augment_data_path=${used_augment_path}

done
