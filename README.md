# multimodal-bart

Code for ???

## TO DO:
- [ ] fix issue with None in last rows of some parquets
- [ ] implement new config for the data to handle the fragmentsgit 

## Usage
To run the code all the paths and parameters in `scripts/train_model.sh` need to be changed accordingly.
Especially, to change the data for the training the config, column, modality, ... parameters need to be changed.
Ex.  
`data.MSMS.column=msms_cfmid_positive_10ev \`


## Add CI badges
Add the CI badges by adding the following line to the README.md: 
```console
[![Build Status](https://v3.travis.ibm.com/[REPO_ACCESS_LINK]branch=main)](https://v3.travis.ibm.com/[REPO_NAME])
```

## Scan Open Source Software Usage
Generate a requirements.txt file that can be scanned for whitesourcing with Mend. 
```console
uv export --format requirements-txt --no-hashes > requirements.txt
```
Activate the scanning. 

## Development setup

Set the environment variables using username and password: 

```console
UV_HTTP_BASIC_INTERNAL_PUBLIC_USERNAME=<username> 
UV_HTTP_BASIC_INTERNAL_PRIVATE_USERNAME=<username>

UV_HTTP_BASIC_INTERNAL_PUBLIC_PASSWORD=<password>
UV_HTTP_BASIC_INTERNAL_PRIVATE_PASSWORD=<password>
```
To install the package run:

```console
uv init .
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e .[dev]
```

## Using Ruff

```console
uv run ruff check .          # Lint all files in the current directory.
uv run ruff check . --fix    # Lint all files in the current directory, and fix any fixable errors.
uv run ruff check . --watch  # Lint all files in the current directory, and re-lint on change.
```

## Using the modeling module

Install the `modeling` extras:

```console
uv pip install -e ".[modeling]"
```

Run a training using a training pipeline configuration (see [sample](./src/mmbart/modeling/resources/train_pipeline_configuration_example.yaml))

```console
torchrun --nproc_per_node {NUMBER_OF_GPUS} src/mmbart/modeling/cli/training.py --pipeline_configuration_path ./src/mmbart/modeling/resources/train_pipeline_configuration_example.yaml
```

## Using the data_analysis module

```console
uv pip install -e ".[data_analysis]"


# analyze a dataset that can be downloaded from Hugging Face library

python src/data_analysis/run_data_analysis.py --dataset_name <hugging face dataset name> --config_name <hugging face dataset config>

# analyze a dataset that is stored in a local fileset 
python src/data_analysis/run_data_analysis.py --data_folder <path to main dataset folder>
```

## Guidelines

Don't forget to follow the guidelines decided here [here](https://github.ibm.com/AI4SD/ai4sd-misc/blob/main/markdown/coding_guidelines.md).
