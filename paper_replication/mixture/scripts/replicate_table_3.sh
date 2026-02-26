#!/bin/bash

while getopts "r:p:f;" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    p) pretrain_data_folder="$OPTARG" ;;
    f) finetune_data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False
export HF_DATASETS_CACHE=/dccstor/analytical/hf_cache/

n_epochs=1
val_check_int=1000
learning_rate=1e-3
num_points_for_patch=75
n_cpus=64

echo "Ternary balanced"
model=custom_model_align
reconstruction_net=mlp
reconstruction_loss=mse
lambda=5
task=ternary
patience=99

output_folder=${run_folder}/${task}
mkdir -p ${output_folder}

bsub -q normal -gpu "num=4:mode=exclusive_process" -o ${output_folder}/out.txt -e ${output_folder}/err.txt \
    python3 /dccstor/analytical/experiments/mixture_ir/MultimodalAnalytical/src/analytical_fm/cli/training.py \
    working_dir=${run_folder} \
    job_name=${task} \
    data_path=${data_folder} \
    data=ir/patches_mixture_text_align \
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
    model.align_config.loss_lambda=${lambda} \
    model.align_config.loss_function=${reconstruction_loss} \
    model.align_config.align_network=${reconstruction_net}