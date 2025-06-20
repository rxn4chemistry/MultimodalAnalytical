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
reconstruction_net=convolutional
reconstruction_loss=mae
lambda=50
num_beams=30

echo "Predict on real mixtures"

# finetune on 5 folds
for i in {0..4}
do
    echo "Predicting with fine-tuned model $i"

    python3 src/analytical_fm/cli/training.py \ 
      working_dir=${run_folder} \ 
      job_name=${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}/predict_real_mixtures \ 
      data_path=${data_folder}/real_mixtures \  
      data=ir/patches_mixture_text_percentage_align 
      model=${model} 
      molecules=True \ 
      data.IR.preprocessor_arguments.patch_size=${num_points_for_patch} \ 
      model.positional_encoding_type=learned \ 
      model.gated_linear=True \ 
      model.optimiser=adamw \ 
      num_cpu=${n_cpus} \  
      splitting=test_only \ 
      predict_class=Percentage \ 
      preprocessor_path=${run_folder}/${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}/preprocessor.pkl 
      model.model_checkpoint_path=${run_folder}/${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}/finetuned_${i}/version_0/checkpoints/best.ckpt \ 
      model.n_beams=${num_beams} 

    cat ${run_folder}/${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}/predict_real_mixtures/finetuned_${i}/metrics*

done


echo "Predict on real mixtures with rejection sampling"

# finetune on 5 folds
for i in {0..4}
do
    echo "Predicting with fine-tuned model $i"

    python3 src/analytical_fm/cli/predict.py \ 
      working_dir=${run_folder} \ 
      job_name=${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}/predict_real_mixtures_rejection \ 
      data_path=${data_folder}/real_mixtures \  
      data=ir/patches_mixture_text_percentage_align 
      model=${model} 
      molecules=True \ 
      data.IR.preprocessor_arguments.patch_size=${num_points_for_patch} \ 
      model.positional_encoding_type=learned \ 
      model.gated_linear=True \ 
      model.optimiser=adamw \ 
      num_cpu=${n_cpus} \  
      splitting=test_only \ 
      predict_class=Percentage \ 
      preprocessor_path=${run_folder}/${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}/preprocessor.pkl 
      model.model_checkpoint_path=${run_folder}/${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}/finetuned_${i}/version_0/checkpoints/best.ckpt \ 
      model.n_beams=${num_beams} \ 
      model.rejection_sampling=True

    cat ${run_folder}/${task}_align_${reconstruction_net}_${reconstruction_loss}_${lambda}/predict_real_mixtures_rejection/finetuned_${i}/metrics*
done