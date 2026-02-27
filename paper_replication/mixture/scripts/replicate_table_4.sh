#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False

# here data folder should contain a parqet files in which the columns are:
# - `molecular_formula`: A string. The chemical formula of the target molecule.
# - `ir_spectra`: An array of floats. The IR spectrum of the mixture in which the target is present.
# - `smiles`: A string. SMILES string representing the target molecule.

n_epochs=1
val_check_int=1000
patience=99
learning_rate=1e-3
num_points_for_patch=75
n_cpus=64
model=custom_model_align
task=multitask
reconstruction_net=mlp
reconstruction_loss=mse
lambda=5
num_beams=30

echo "Predict on real mixtures"

# finetune on 5 folds
for cv_split in {0..4}; do

    out_folder=${run_folder}/cv_split_${cv_split}/

    echo "Predicting with fine-tuned model $cv_split"

    python3 src/analytical_fm/cli/training.py \
      working_dir=${out_folder} \
      job_name=predict_real_mixtures_synthetic \
      data_path=${data_folder} \
      data=ir/patches_mixture_text_align \
      model=${model} \
      molecules=True \
      data.IR.preprocessor_arguments.patch_size=${num_points_for_patch} \
      model.positional_encoding_type=learned \
      model.gated_linear=True \
      model.optimiser=adamw \
      num_cpu=${n_cpus} \
      splitting=test_only \
      model.align_config.loss_lambda=${lambda} \
      model.align_config.loss_function=${reconstruction_loss} \
      model.align_config.align_network=${reconstruction_net} \
      preprocessor_path=${run_folder}/preprocessor.pkl \
      model.model_checkpoint_path=${out_folder}/version_0/checkpoints/best.ckpt \
      model.n_beams=${num_beams} 

done


echo "Predict on real mixtures with rejection sampling"

# finetune on 5 folds
for cv_split in {0..4}; do

    out_folder=${run_folder}/cv_split_${cv_split}/
    echo "Predicting with fine-tuned model $i"

    python3 src/analytical_fm/cli/training.py \
      working_dir=${out_folder} \
      job_name=predict_real_mixtures_rejection \
      data_path=${data_folder} \
      data=ir/patches_mixture_text_align \
      model=${model} \
      molecules=True \
      data.IR.preprocessor_arguments.patch_size=${num_points_for_patch} \
      model.positional_encoding_type=learned \
      model.gated_linear=True \
      model.optimiser=adamw \
      num_cpu=${n_cpus} \
      splitting=test_only \
      model.align_config.loss_lambda=${lambda} \
      model.align_config.loss_function=${reconstruction_loss} \
      model.align_config.align_network=${reconstruction_net} \
      preprocessor_path=${run_folder}/preprocessor.pkl \
      model.model_checkpoint_path=${out_folder}/version_0/checkpoints/best.ckpt \
      model.n_beams=${num_beams} \
      model.rejection_sampling=True

done