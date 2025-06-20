# mypy: ignore-errors

import logging
import math
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset
from omegaconf import DictConfig
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from .configuration import DEFAULT_SETTINGS
from .data.data_utils import IterableDatasetWithLength

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

    sample = sample.replace("<bos>", "").replace("<pad>", "").replace("<eos>", "").replace(" ", '')

    if canonicalise:
        sample = sample.replace(" ", "")
        mol = Chem.MolFromSmiles(sample)
        sample = Chem.MolToSmiles(mol) if mol else None

    return sample

def reject_sample(predictions: Dict[str, Any], molecules: bool = True):
    RDLogger.DisableLog('rdApp.*')

    n_beams = len(predictions["predictions"][0])
    logger.info(f"Doing rejection sampling with n_beams: {n_beams}")
    for i in range(len(predictions["predictions"])):
        pred = []
        for p in predictions["predictions"][i]:
            sample = clean_sample(p, molecules)
            try:
                pred_mol = Chem.MolFromSmiles(sample)
                pred_formula = rdMolDescriptors.CalcMolFormula(pred_mol)
            except TypeError as e:
                logger.error(e)
                continue
            
            try:
                target_mol = Chem.MolFromSmiles(predictions["targets"][i])
                target_formula = rdMolDescriptors.CalcMolFormula(target_mol)
            except TypeError as e:
                logger.error(e)
                continue

            if pred_formula == target_formula:
                pred.append(sample)

        predictions["predictions"][i] = pred + [""]*(n_beams - len(pred))

    assert len(predictions["predictions"]) == len(predictions["targets"]), f"Predictions and targets do not match in size: {len(predictions['predictions'])} != {len(predictions['targets'])}"

    for i in range(len(predictions["predictions"])):
        assert len(predictions["predictions"][i]) == n_beams, f"{len(predictions['predictions'][i])}/{n_beams}"

    logger.info(f"Num targets: {len(predictions['targets'])}")
    logger.info(f"Num predictions: {len(predictions['predictions'][0])}")
    return predictions

def calc_sampling_metrics(samples: List[Any], targets: List[str], classes: List[Any] = None, molecules: bool = True, logging: bool = False) -> Dict[str, float]:
    """Calculate Top-N accuracies for a model

    Args:
        sampled_smiles: SMILES strings produced by decode function,
        target_smiles: target molecules as canonicalised SMILES strings
        molecules: Wether to canonicalise or not
        training: Log results or not. Disable during training

    Returns:
        dict containing results
    """

    n_beams = len(samples[0])
    prediction_df = pd.DataFrame({"predictions": samples, "targets": targets})
    if classes:
        prediction_df["prediction_classes"] = classes
    
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
        if classes:
            for cl in prediction_df["prediction_classes"].unique():
                cls_df = prediction_df[prediction_df["prediction_classes"]==cl]
                top_n_acc = float((cls_df['rank'] <= i).sum() / len(cls_df))
                if float(cl) not in metrics:
                    metrics[float(cl)] = {}
                metrics[float(cl)][f"Top-{i+1}"] = top_n_acc
                if logging:
                    logger.info(f"Class: {cl}. Samples per class: {len(cls_df)}. Top-{i+1}: {top_n_acc:.3f}")
        else:
            top_n_acc = float((prediction_df['rank'] <= i).sum() / len(prediction_df))
            metrics[f"Top-{i+1}"] = top_n_acc

            if logging:
                logger.info(f"Top-{i+1}: {top_n_acc:.3f}")
    
    return metrics


def calculate_training_steps(
    train_set: Dataset, config: DictConfig
) -> int:
    
    len_train = 0
    if isinstance(train_set, IterableDatasetWithLength):
        len_train = train_set._length
    else:
        len_train = len(train_set)
    
    batches_per_gpu = math.ceil(
        (len_train / config['model']['batch_size'])
        / float(1)  # Number of gpus, for now hardcoded to 1
    )
    train_steps = (
        math.ceil(batches_per_gpu / config["trainer"]["acc_batches"])
        * config["trainer"]["epochs"]
    )

    return train_steps

def seed_everything(seed: Optional[int] = None) -> None:
    if seed is None:
        seed = DEFAULT_SETTINGS.default_seed

    pl.seed_everything(seed)

def fail_safe_conditional_distributed_barrier(condition_fn: Callable[[], bool]) -> None:
    """Apply a distributed barrier in a fail-safe way.

    Args:
        condition_fn: callable to define condition for the barrier.
    """
    try:
        if condition_fn():
            logger.info("Distributed barrier applied")
            torch.distributed.barrier()
    except ValueError:
        # NOTE: catching errors due to uninitialized distributed process group.
        # Never active when running without torchrun. In this case a barrier is never needed.
        logger.info("No distributed barrier applied")
