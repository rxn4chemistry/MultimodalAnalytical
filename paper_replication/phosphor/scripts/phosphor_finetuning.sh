#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

for cv_split in 0 1 2 3 4; do
        
        for set in smiles_rad_1 smiles_rad_2 smiles_rad_3; do

            mkdir -p ${run_folder}/${set}_num_2/finetuning/cv_split_${cv_split}
            python -m analytical_fm.cli.training \
                working_dir=${run_folder} \
                job_name=${set}_num_2/finetuning/cv_split_${cv_split} \
                data_path=${data_folder} \
                data=phosphor/num \
                data.Smiles.column=${set} \
                data.Phosphor_NMR.preprocessor_arguments.encoding_type=linear_2_layer \
                model=custom_model \
                molecules=False \
                cv_split=${cv_split} \
                model.lr=1e-3 \
                trainer.epochs=60 \
                finetuning=True \
                preprocessor_path=${run_folder}/${set}_num_2/preprocessor.pkl \
                model.model_checkpoint_path=${run_folder}/${set}_num_2/version_0/checkpoints/epoch_2-step_3063.ckpt

        done
        
        
        for set in smiles_rad_1 smiles_rad_2 smiles_rad_3; do

            mkdir -p ${run_folder}/formula_${set}_num_2/finetuning/cv_split_${cv_split}
            python -m analytical_fm.cli.training \
                working_dir=${run_folder} \
                job_name=formula_${set}_num_2/finetuning/cv_split_${cv_split} \
                data_path=${data_folder} \
                data=phosphor/formula_num \
                data.Smiles.column=${set} \
                data.Formula.column=formula \
                data.Phosphor_NMR.preprocessor_arguments.encoding_type=linear_2_layer \
                model=custom_model \
                molecules=False \
                cv_split=${cv_split} \
                model.lr=1e-3 \
                trainer.epochs=60 \
                finetuning=True \
                preprocessor_path=${run_folder}/formula_${set}_num_2/preprocessor.pkl \
                model.model_checkpoint_path=${run_folder}/formula_${set}_num_2/version_0/checkpoints/last.ckpt
            
        done        
        
done
