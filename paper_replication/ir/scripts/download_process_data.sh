# bin/bash 

while getopts "o:" opt; do
  case $opt in
    o) output_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

# Make data folder
mkdir -p ${output_folder}/raw_data

# Download Multimodal Dataset
echo "Downloading Multimodal Dataset"
curl -o ${output_folder}/raw_data/mm_dataset.zip https://zenodo.org/records/14770232/files/multimodal_spectroscopic_dataset.zip?download=1
unzip ${output_folder}/raw_data/mm_dataset.zip -d ${output_folder}/raw_data/
rm ${output_folder}/raw_data/mm_dataset.zip

# Downloading Synthetic IR Data
echo "Downloading Synthetic IR Data"
curl -o ${output_folder}/raw_data/synth_ir_data.zip https://zenodo.org/records/7928396/files/IRtoMol.zip?download=1
unzip ${output_folder}/raw_data/synth_ir_data.zip -d ${output_folder}/raw_data/
rm ${output_folder}/raw_data/synth_ir_data.zip

# Downloading Pseudo Exp. data for augmentations
echo "Downloading pseudo experimental spectra for augmentations"
mkdir -p ${output_folder}/pseudo_experimental
curl -o ${output_folder}/pseudo_experimental/pseudo_experimental.zip https://zenodo.org/records/15116374/files/pseudo_experimental.zip?download=1
unzip ${output_folder}/raw_data/synth_ir_data.zip -d ${output_folder}/pseudo_experimental/

# Process IR spectra
python scripts/process_data.py --data_folder ${output_folder}