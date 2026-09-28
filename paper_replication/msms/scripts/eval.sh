#!/bin/bash

while getopts "r:d:" opt; do
  case $opt in
    r) run_folder="$OPTARG" ;;
    d) data_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export TOKENIZERS_PARALLELISM=False


# run_folder is the folder of one step, e.g. runs/<exp>/ft
# TTT is evaluated on its last checkpoint, the other steps on the best one
checkpoint=${run_folder}/version_0/checkpoints/best.ckpt
[ -f ${checkpoint} ] || checkpoint=${run_folder}/version_0/checkpoints/last.ckpt
# fine-tuning and TTT reuse the preprocessor of the pre-training step
preprocessor=${run_folder}/preprocessor.pkl
[ -f ${preprocessor} ] || preprocessor=${run_folder}/../pt/preprocessor.pkl

# a folder with train/val/test parquets is evaluated on test, a single parquet file entirely
splitting=given_splits
[ -f ${data_folder} ] && splitting=test_only

mkdir -p ${run_folder}/eval

python -m analytical_fm.cli.predict \
    working_dir=${run_folder} \
    job_name=eval \
    data_path=${data_folder} \
    data=msms/text_fingerprint \
    model=custom_model_align \
    model.batch_size=64 \
    model.rejection_sampling=formula \
    model.model_checkpoint_path=${checkpoint} \
    preprocessor_path=${preprocessor} \
    splitting=${splitting} \
    molecules=True
