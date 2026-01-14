# mypy: ignore-errors

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
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

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

RDLogger.DisableLog('rdApp.*')
def calculate_tanimoto(pred, truth):
    # print('Calculating Tanimoto...')
    fpgen = AllChem.GetRDKitFPGenerator()
    tan = TanimotoSimilarity(fpgen.GetFingerprint(Chem.MolFromSmiles(pred)), fpgen.GetFingerprint(Chem.MolFromSmiles(truth)))
    
    return tan

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

        predictions["predictions"][i] = pred + ["null"]*(n_beams - len(pred))

    assert len(predictions["predictions"]) == len(predictions["targets"]), f"Predictions and targets do not match in size: {len(predictions['predictions'])} != {len(predictions['targets'])}"

    for i in range(len(predictions["predictions"])):
        assert len(predictions["predictions"][i]) == n_beams, f"{len(predictions['predictions'][i])}/{n_beams}"

    logger.info(f"Num targets: {len(predictions['targets'])}")
    logger.info(f"Num predictions: {len(predictions['predictions'][0])}")
    return predictions

def calc_sampling_metrics(samples: List[Any], targets: List[str], classes: List[Any] | None = None, molecules: bool = True, logging: bool = False) -> Dict[str, float]:
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

    # Calculate invalid
    invalid = [mol[0] is None for mol in prediction_df['predictions_clean']]
    if logging:
        logger.info(f"\nInvalid SMILES = {sum(invalid)/len(prediction_df)*100} %")


    # Calculate metrics
    metrics = {}

    #all_preds = np.stack(prediction_df["predictions"].to_list())

    for i in range(n_beams):
        if i <= 10:
            if classes:
                for cl in prediction_df["prediction_classes"].unique():
                    cls_df = prediction_df[prediction_df["prediction_classes"]==cl]
                    top_n_acc = float((cls_df['rank'] <= i).sum() / len(cls_df))
                    if float(cl) not in metrics:
                        metrics[float(cl)] = {}
                    metrics[float(cl)][f"Top-{i+1}"] = top_n_acc
                    if logging:
                        logger.info(f"Class: {cl}. Samples per class: {len(cls_df)}. Top-{i+1}: {top_n_acc:.5f}")
            else:
                top_n_acc = float((prediction_df['rank'] <= i).sum() / len(prediction_df))
                metrics[f"Top-{i+1}"] = top_n_acc

                if logging:
                    logger.info(f"Top-{i+1}: {top_n_acc:.5f}")
    
    return metrics

def evaluate(predict_class, data_config, config, data_module, trainer, model, n_beams):
    logger.info(f"Calculating metrics for class: {predict_class}")
    if predict_class and predict_class in data_config.keys():
        logger.info("Class is present in the dataset.")
        classes = []
        for batch in data_module.predict_dataloader():
            classes.extend(batch[predict_class])

        if isinstance(classes[0],list):
            classes = [cl[0] for cl in classes]

        logger.info(f"Classes: {set(classes)}")
        logger.info(f"Len of classes array: {len(classes)}")

    batch_predictions: List[Dict[str, Any]] = trainer.predict(model, datamodule=data_module) # type:ignore
    
    # Concatenate Predictions
    predictions = {
        'avg_loss': np.mean([batch['loss'] for batch in batch_predictions]),
        'predictions': [batch['predictions'][i * n_beams : (i+1) * n_beams] for batch in batch_predictions for i in range(len(batch['predictions']) // n_beams)],
        'targets': [target for batch in batch_predictions for target in batch['targets']],
        'classes': [cl for batch in batch_predictions for cl in batch[predict_class]] if predict_class else None,
    }

    return predictions

def save_to_files(predictions, metrics, config, n_beams, name_file):
    rank = torch.distributed.get_rank() if torch.cuda.is_available() else 0
    paths = []
    if predictions:
        predictions_path = (
            Path(config["working_dir"])
            / config["job_name"]
            / f"{name_file}test_data_logits_beam_{n_beams}_{rank}.pkl"
        )
        with (predictions_path).open("wb") as predictions_file:
            pickle.dump(predictions, predictions_file)
        paths.append(predictions_path)

    if metrics:
        metrics_path = (
            Path(config["working_dir"])
            / config["job_name"]
            / f"{name_file}metrics_beam_{n_beams}_{rank}.json"
        )
        with (metrics_path).open("w") as metrics_file:
            json.dump(metrics, metrics_file)
        paths.append(metrics_path)
    
    return tuple(paths)

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
        / float(torch.cuda.device_count() if torch.cuda.is_available() else 1)  # Number of gpus
    )
    train_steps = (
        math.ceil(batches_per_gpu / config["trainer"]["acc_batches"])
        * config["trainer"]["epochs"]
    )

    return train_steps

def seed_everything(seed: Optional[int] = None) -> None:
    if seed is None:
        seed = DEFAULT_SETTINGS.default_seed
    
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

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
