from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from rdkit import Chem
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def interpolate(spec: np.ndarray, x: np.ndarray, upscale_val: int) -> np.ndarray:
    interp = interp1d(x, spec)
    new_x = np.arange(0, upscale_val, 1)
    new_spec = interp(new_x)
    return new_spec


def horizontal_shift_augment(spectrum: np.ndarray, n_augments: int = 2) -> List[np.ndarray]:

    old_x = np.linspace(0, len(spectrum), len(spectrum) // n_augments)

    augmented_specs = []
    for i in range(n_augments):

        spec_shifted = spectrum[i : (-n_augments + i) : n_augments]
        spec_shifted = interpolate(spec_shifted, old_x, len(spectrum))
        augmented_specs.append(spec_shifted)

    return augmented_specs


def smooth_augment(
    spectrum: np.ndarray, sigmas: List[float]
) -> List[np.ndarray]:
    
    smoothed_spectra = list()
    for sigma in sigmas:
        smooth_spectrum = gaussian_filter1d(spectrum, sigma)
        smoothed_spectra.append(smooth_spectrum)

    return smoothed_spectra

def smiles_augment(smiles: str, n_augments: int) -> List[str]:
    
    mol = Chem.MolFromSmiles(smiles)
    augments = [Chem.MolToSmiles(mol, canonical=False, doRandom=True) for _ in range(n_augments)]
    return augments

AUGMENT_OPTIONS = {"horizontal": horizontal_shift_augment, "smooth": smooth_augment, "smiles_aug": smiles_augment}

# Todo: Move Augmentations into datamodule and do augmentations on the fly
def augment(dataset: Dataset, augment_config: Optional[DictConfig]) -> Dataset:

    if not isinstance(augment_config, DictConfig):
        return dataset

    # Only perform augmentation if augment config has necessary keys
    augmented_datasets = list()
    if isinstance(augment_config['augmentations'], ListConfig) and len(augment_config['augmentations']) != 0:
        for augment_fields in augment_config['augmentations']:

            augment_column = augment_fields['augment_column']
            augment_fns = augment_fields['augment_fns']

            augmented_datasets.append(dataset.map(lambda row : apply_augment(row, augment_column, augment_fns),
                                                  batched=True,
                                                  batch_size=1,
                                                  #num_proc=7
                                    ))
                                        
    dataset = concatenate_datasets([dataset, *augmented_datasets])

    if augment_config and augment_config['augment_data_path']:
        augment_dataset = load_from_disk(augment_config['augment_data_path'])
        dataset = concatenate_datasets([dataset, augment_dataset])

    return dataset

def apply_augment(row, augment_column: str, augment_config: Dict[str, Any]):

    augmented_data = list()

    # Perform Augmentations
    for augment_type, augment_params in augment_config.items():
        augmented_data.extend(AUGMENT_OPTIONS[augment_type](row[augment_column][0], **augment_params)) # type:ignore
    
    # Duplicate Data in other columns
    augmented_row = {column: row[column] * len(augmented_data) for column in row.keys() if column != augment_column}
    augmented_row[augment_column] = augmented_data

    return augmented_row
