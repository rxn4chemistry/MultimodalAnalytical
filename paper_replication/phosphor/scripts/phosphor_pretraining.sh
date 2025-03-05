#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

for set in smiles_rad_1 smiles_rad_2 smiles_rad_3; do

    mkdir -p ${run_folder}/${set}_num_2
    jbsub -queue x86_6h -cores 7+1 -mem 25g \
        -out ${run_folder}/${set}_num_2/out.txt \
        -err ${run_folder}/${set}_num_2/err.txt \
        python -m analytical_fm.cli.training \
            working_dir=${run_folder} \
            job_name=${set}_num_2 \
            data_path=${data_folder} \
            data=phosphor/num \
            data.Smiles.column=${set} \
            data.Phosphor_NMR.preprocessor_arguments.encoding_type=linear_2_layer \
            model=custom_model \
            molecules=False \
            model.lr=1e-3 \
            trainer.epochs=10 

done


for set in smiles_rad_1 smiles_rad_2 smiles_rad_3; do

    mkdir -p ${run_folder}/formula_${set}_num_2
    jbsub -queue x86_6h -cores 7+1 -mem 25g \
        -out ${run_folder}/formula_${set}_num_2/out.txt \
        -err ${run_folder}/formula_${set}_num_2/err.txt \
        python /dccstor/analytical/experiments/analytical_v2/multimodal-bart/src/mmbart/cli/training.py \
            working_dir=${run_folder} \
            job_name=formula_${set}_num_2 \
            data_path=${data_folder} \
            data=phosphor/formula_num \
            data.Phosphor_NMR.preprocessor_arguments.encoding_type=linear_2_layer \
            data.Smiles.column=${set} \
            model=custom_model \
            molecules=False \
            model.lr=1e-3 \
            trainer.epochs=60 

done
