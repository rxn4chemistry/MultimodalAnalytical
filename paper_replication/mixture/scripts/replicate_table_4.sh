#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

n_epochs=1
val_check_int=1000
learning_rate=1e-3
num_points_for_patch=75
n_cpus=64
model=custom_model

# pretrain and finetune without alignment
echo "Binary imbalanced without alignment"


for task in binary_1_9 binary_3_7 binary_4_6; do
    patience=99

    echo "Pretrain on task $task"
    # pretrain on the binary mixtures
    python3 src/analytical_fm/cli/training.py \ 
        working_dir=${run_folder} \ 
        job_name=${task} \ 
        data_path=${data_folder}/pretraining \ 
        data=ir/patches_mixture_text_percentage \ 
        model=${model} \ 
        molecules=True \ 
        trainer.epochs=${n_epochs} \ 
        trainer.val_check_interval=${val_check_int} \ 
        trainer.early_stopping_patience=${patience} \ 
        model.lr=${learning_rate} \ 
        data.IR.preprocessor_arguments.patch_size=${num_points_for_patch} \ 
        model.positional_encoding_type=learned \ 
        model.gated_linear=True \ 
        model.optimiser=adamw \ 
        mixture=ir/${task} \ 
        num_cpu=${n_cpus} \ 
        splitting=unique_target \ 
        predict_class=Percentage

    # to print the result of the test after the training
    cat ${run_folder}/${task}/metrics*

    done
done



