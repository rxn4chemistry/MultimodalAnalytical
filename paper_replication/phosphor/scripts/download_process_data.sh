# bin/bash 

while getopts "o:" opt; do
  case $opt in
    o) output_folder="$OPTARG" ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Make data folder
mkdir -p ${output_folder}

# Download Hack Data
echo "Downloading Hack Data"
curl -o ${output_folder}/hack_data.csv https://raw.githubusercontent.com/clacor/Ilm-NMR-P31/refs/heads/master/Ilm-NMR-P31.csv

# Process Hack data
echo "Processing Hack Data"
mkdir -p ${output_folder}/hack_clean
python scripts/process_hack_data.py --data_path ${output_folder}/hack_data.csv --output_path ${output_folder}/hack_clean

# Download Synthetic data
echo "Downloading Synthetic Data"
mkdir -p ${output_folder}/pretraining
#curl -o ${output_folder}/pretraining/pretraining_data.parquet <tbd>
