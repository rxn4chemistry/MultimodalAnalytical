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
curl -o ${output_folder}/mm_dataset.zip https://zenodo.org/records/14770232/files/multimodal_spectroscopic_dataset.zip?download=1
unzip ${output_folder}/mm_dataset.zip -d ${output_folder}/pretrain_data/
rm ${output_folder}/mm_dataset.zip
