#!/bin/bash

while getopts "r:p:f:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    p) pretrain_data_folder="$OPTARG" ;;
    f) finetune_data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

n_epochs=1
val_check_int=1000
learning_rate=1e-3
num_points_for_patch=75
n_cpus=64
model=custom_model_align
task=binary

# ablation study on the alignment
for reconstruction_net in convolutional mlp; do
    for reconstruction_loss in mae mse; do
        for lambda in 1 5 50; do

            # Pretraining
            echo "Running reconstruction_net=$reconstruction_net, reconstruction_loss=$reconstruction_loss and lambda=$lambda"
            output_folder=${run_folder}/${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}
            mkdir -p ${output_folder}
            python3 src/analytical_fm/cli/training.py \
                working_dir=${run_folder} \
                job_name=${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda} \
                data_path=${pretrain_data_folder} \
                data=ir/patches_mixture_text_align \
                model=${model} \
                molecules=True \
                trainer.epochs=${n_epochs} \
                trainer.val_check_interval=${val_check_int} \
                trainer.early_stopping_patience=99 \
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

            cat ${run_folder}/${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}/metrics*

            # Finetuning
            for cv_split in {0..4}; do
                patience=20
                cv_folder=${output_folder}/cv_split_${cv_split}
                mkdir -p ${cv_folder}
                python3 src/analytical_fm/cli/training.py \
                        working_dir=${output_folder} \
                        job_name=cv_split_${cv_split} \
                        data_path=${finetune_data_folder} \
                        data=ir/patches_mixture_text_align \
                        cv_split=${cv_split} \
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
                        model.align_config.loss_lambda=${lambda} \
                        model.align_config.loss_function=${reconstruction_loss} \
                        model.align_config.align_network=${reconstruction_net} \
                        model.batch_size=64 \
                        finetuning=True \
                        preprocessor_path=${output_folder}/preprocessor.pkl \
                        model.model_checkpoint_path=${output_folder}/version_0/checkpoints/best.ckpt

            done
        done
    done
done