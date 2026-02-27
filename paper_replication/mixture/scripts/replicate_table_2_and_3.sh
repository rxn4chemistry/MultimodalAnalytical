#!/bin/bash



while getopts "r:p:f:t:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    p) pretrain_data_folder="$OPTARG" ;;
    f) finetune_data_folder="$OPTARG" ;;
    t) task="$OPTARG" ;; # Choose multitask_w_pure to replicate the results found in Table 2 or ternary to replicate the results in table 3
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

n_epochs=1
val_check_int=1000
learning_rate=1e-3
num_points_for_patch=75
n_cpus=64

echo "Binary imbalanced with multitask and alignment"
model=custom_model_align
reconstruction_net=mlp
reconstruction_loss=mse
lambda=5
patience=99

echo ${output_folder}
mkdir -p ${output_folder}

python3 src/analytical_fm/cli/training.py \
    working_dir=${run_folder} \
    job_name=${task} \
    data_path=${pretrain_data_folder} \
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

for cv_split in {0..4}; do 
    output_folder=${run_folder}/${task}/cv_split_${cv_split}
    echo ${output_folder}
    mkdir -p ${output_folder}

    python3 src/analytical_fm/cli/training.py \
        working_dir=${run_folder}/${task} \
        job_name=cv_split_${cv_split} \
        data_path=${finetune_data_folder} \
        cv_split=${cv_split} \
        data=ir/patches_mixture_text_align \
        model=${model} \
        molecules=True \
        trainer.epochs=${n_epochs} \
        trainer.val_check_interval=${val_check_int} \
        trainer.early_stopping_patience=20 \
        model.lr=${learning_rate} \
        data.IR.preprocessor_arguments.patch_size=${num_points_for_patch} \
        model.positional_encoding_type=learned \
        model.gated_linear=True \
        model.optimiser=adamw \
        mixture=ir/${task} \
        num_cpu=${n_cpus} \
        splitting=unique_target \
        predict_class=Percentage \
        model.align_config.loss_lambda=${lambda} \
        model.align_config.loss_function=${reconstruction_loss} \
        model.align_config.align_network=${reconstruction_net} \
        model.batch_size=64 \
        finetuning=True \
        preprocessor_path=${run_folder}/${task}/preprocessor.pkl \
        model.model_checkpoint_path=${run_folder}/${task}/version_0/checkpoints/best.ckpt

done