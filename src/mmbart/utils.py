# mypy: ignore-errors

import logging
import math
from typing import Dict, List, Optional, Any

import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from rdkit import Chem, RDLogger

from mmbart.data.datamodules import MultiModalDataModule
from mmbart.defaults import DEFAULT_SEED

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def clean_sample(sample: str, canonicalise: bool) -> str:
    """Clean sampled string from a model.
    
    Removes eos, bos, pad. If canonicalise, returns canonical smiles string.

    Args:
        sample: Model string sample
        canonicalise: Wether to canonicalise or not
    Returns:
        clean string
    """

    sample = sample.replace("<bos>", "").replace("<pad>", "").replace("<eos>", "").strip()

    if canonicalise:
        sample = sample.replace(" ", "")
        mol = Chem.MolFromSmiles(sample)
        sample = Chem.MolToSmiles(mol) if mol else None

    return sample


def calc_sampling_metrics(samples: List[Any], targets: List[str], molecules: bool = True) -> Dict[str, float]:
    """Calculate Top-N accuracies for a model

    Args:
        sampled_smiles: SMILES strings produced by decode function,
        target_smiles: target molecules as canonicalised SMILES strings

    Returns:
        dict containing results
    """

    n_beams = len(samples[0])
    prediction_df = pd.DataFrame({"predictions": samples, "targets": targets})
    
    # Clean Predictions and Target
    RDLogger.DisableLog('rdApp.*')
    prediction_df["predictions_clean"] = prediction_df["predictions"].map(lambda prediction : [clean_sample(pred, molecules) for pred in prediction])
    prediction_df["targets_clean"] = prediction_df["targets"].map(lambda target : clean_sample(target, molecules))

    # Calculate rank
    prediction_df['rank'] = prediction_df.apply(lambda row :
                                                row['predictions_clean'].index(row['targets_clean']) if row['targets_clean'] in row['predictions_clean'] else n_beams, axis=1)

    # Calculate metrics
    metrics = {}

    #all_preds = np.stack(prediction_df["predictions"].to_list())

    for i in range(n_beams):
        top_n_acc = (prediction_df['rank'] <= i).sum() / len(prediction_df) * 100
        metrics[f"Top-{i+1}"] = top_n_acc
        logger.info(f"Top-{i+1}: {top_n_acc:.3f}")
    
    return metrics


def calculate_training_steps(
    data_module: MultiModalDataModule, config: DictConfig
) -> int:
    batches_per_gpu = math.ceil(
        len(data_module.train_dataloader())
        / float(1)  # Number of gpus, for now hardcoded to 1
    )
    train_steps = (
        math.ceil(batches_per_gpu / config["trainer"]["acc_batches"])
        * config["trainer"]["epochs"]
    )

    return train_steps

def seed_everything(seed: Optional[int] = None) -> None:
    if seed is None:
        seed = DEFAULT_SEED

    pl.seed_everything(seed)

