from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
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


def augment_smooth(
    spectrum: np.ndarray, sigmas: List[float]
) -> List[np.ndarray]:
    
    smoothed_spectra = list()
    for sigma in sigmas:
        smooth_spectrum = gaussian_filter1d(spectrum, sigma)
        smoothed_spectra.append(smooth_spectrum)

    return smoothed_spectra

AUGMENT_OPTIONS = {"horizontal": horizontal_shift_augment, "smooth": augment_smooth}

# Todo: Move Augmentations into datamodule and do augmentations on the fly
def augment(dataset: Dataset, augment_config: Optional[Dict[str, Any]]) -> Dataset:

    # Only perform augmentation if augment config has necessary keys
    if augment_config and augment_config['augment_column'] and len(augment_config['augmentations']) != 0:
        dataset = dataset.map(lambda row : augment_spec(row, augment_config['augment_column'], augment_config['augmentations']),
                              batched=True,
                              batch_size=1,
                            )

    if augment_config and augment_config['augment_data_path']:
        augment_dataset = load_dataset('parquet', data_dir=augment_config['augment_data_path'])
        dataset = concatenate_datasets(dataset, augment_dataset)

    return dataset

def augment_spec(row, augment_column: str, augment_config: Dict[str, Any]):

    # Keep original
    augmented_data = [row[augment_column][0]]

    # Perform Augmentations
    for augment_type, augment_params in augment_config.items():
        augmented_data.extend(AUGMENT_OPTIONS[augment_type](row[augment_column][0], **augment_params)) # type:ignore
    
    # Duplicate Data in rows
    augmented_row = {column: row[column] * len(augmented_data) for column in row.keys() if column != augment_column}
    augmented_row[augment_column] = augmented_data

    return augmented_row
