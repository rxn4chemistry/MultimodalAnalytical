"""Run the prediction pipeline."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2024
ALL RIGHTS RESERVED
"""

import logging
import pickle
from pathlib import Path
import os
os.environ["HF_DATASETS_CACHE"] = "/dccstor/ltlws3emb/cache/hf_cache"
os.environ["LD_LIBRARY_PATH"] = "/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import pandas as pd
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from mmbart.data.data_utils import load_preprocessors
from mmbart.data.datamodules import MultiModalDataModule
from mmbart.data.datasets import (  # noqa: F401
    build_dataset_multimodal,
)
from mmbart.modeling.wrapper import HFWrapper
from mmbart.util import calculate_training_steps, seed_everything

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def calc_sampling_metrics(sampled_smiles, target_smiles, molecules: bool = True):
        """Calculate sampling metrics for the model

        If sampled_smiles is a List[List[str]] then the following metrics for beam search are calculated (up to the
        maximum given by the number of elements in the inner lists):
            - "top_1_accuracy"
            - "top_5_accuracy"
            - "top_10_accuracy"
            - "top_20_accuracy"
            - "top_50_accuracy"
        The SMILES strings must be sorted in decreasing order of their predicted likelihood

        If the sampled_smiles is a List[str] then "accuracy" is calculated

        The the number of invalid SMILES "invalid" is also returned (for beam search this is just from the top_1)

        Args:
            sampled_smiles: SMILES strings produced by decode function,
            target_smiles: target molecules as canonicalised SMILES strings

        Returns:
            dict containing results
        """

        num_sampled = len(sampled_smiles)
        num_target = len(target_smiles)
        err_msg = f"The number of sampled and target molecules must be the same, got {num_sampled} and {num_target}"
        assert num_sampled == num_target, err_msg

        if molecules:
            mol_targets = [
                Chem.MolFromSmiles(
                    smi.replace(" ", "")
                    .replace("<bos>", "")
                    .replace("<pad>", "")
                    .replace("<eos>", "")
                    .replace(" ", '')
                )
                for smi in target_smiles
            ]
            canon_targets = [Chem.MolToSmiles(mol) for mol in mol_targets]
        else:
            canon_targets = [text.replace("<bos>", "").replace("<pad>", "").replace("<eos>", "").strip() for text in target_smiles]

        data_type = type(sampled_smiles[0])
        if data_type == str:
            results = _calc_greedy_metrics(sampled_smiles, canon_targets, molecules=molecules)
        elif data_type == list:
            results = _calc_beam_metrics(sampled_smiles, canon_targets, molecules=molecules)
        else:
            raise TypeError(
                f"Elements of sampled_smiles must be either a str or a list, got {data_type}"
            )

        return results


def _calc_greedy_metrics(sampled_smiles, target_smiles, molecules: bool = True):
    if molecules:
        sampled_mols = [
            Chem.MolFromSmiles(
                smi.replace(" ", "")
                .replace("<bos>", "")
                .replace("<pad>", "")
                .replace("<eos>", "")
                .replace(' ', '')
            )
            for smi in sampled_smiles
        ]
    else:
        sampled_mols = [
            text.replace("<bos>", "").replace("<pad>", "").replace("<eos>", "").replace(" ", "").strip() for text in sampled_smiles
        ]
    invalid = [mol is None for mol in sampled_mols]

    if molecules:
        canon_smiles = [
            "Unknown" if mol is None else Chem.MolToSmiles(mol) for mol in sampled_mols
        ]
    else:
        canon_smiles = sampled_mols
    correct_smiles = [
        target_smiles[idx] == smi for idx, smi in enumerate(canon_smiles)
    ]

    num_correct = sum(correct_smiles)
    total = len(correct_smiles)
    num_invalid = sum(invalid)
    perc_invalid = num_invalid / total
    accuracy = num_correct / total

    metrics = {"accuracy": accuracy, "invalid": perc_invalid}

    return metrics

def _calc_beam_metrics(sampled_smiles, target_smiles, molecules: bool = True):
    top_1_samples = [mols[0] for mols in sampled_smiles]
    top_1_results = _calc_greedy_metrics(top_1_samples, target_smiles)

    metrics = {
        "top_1_accuracy": top_1_results["accuracy"],
        "invalid": top_1_results["invalid"],
    }

    ks = [2, 3, 5, 10, 20, 50]
    num_samples_list = [k for k in ks if k <= len(sampled_smiles[0])]

    for num_samples in num_samples_list:
        top_k_correct = []
        num_mols = len(sampled_smiles)

        for batch_idx, mols in enumerate(sampled_smiles):
            samples = mols[:num_samples]

            if molecules:
                samples_mols = [
                    Chem.MolFromSmiles(
                        smi.replace(" ", "")
                        .replace("<bos>", "")
                        .replace("<pad>", "")
                        .replace("<eos>", "")
                        .replace(' ', '')
                    )
                    for smi in samples
                ]
                samples_smiles = [
                    "Unknown" if mol is None else Chem.MolToSmiles(mol)
                    for mol in samples_mols
                ]
            else:
                samples_smiles = [
                    text.replace("<bos>", "").replace("<pad>", "").replace("<eos>", "").replace(" ", "").strip() for text in samples
                ]
            correct_smiles = [
                smi == target_smiles[batch_idx] for smi in samples_smiles
            ]
            is_correct = sum(correct_smiles) >= 1
            top_k_correct.append(is_correct)
        accuracy = sum(top_k_correct) / num_mols
        metrics[f"top_{str(num_samples)}_accuracy"] = accuracy

    return metrics

def score_molecules(predictions, ground_truth, n_beams = 10):
    preds = pd.DataFrame({"predictions": predictions, "ground_truth": ground_truth})
    preds['predictions'] = preds['predictions'].map(lambda pred_list : [pred.replace(' ', '').replace("<bos>", "").replace("<pad>", "").replace("<eos>", "").replace(" ", "").strip() for pred in pred_list])
    preds['rank'] = preds.apply(lambda row : row['predictions'].index(row['ground_truth']) if row['ground_truth'] in row['predictions'] else n_beams, axis=1)

    for i in range(n_beams):
        logger.info(f"Top-{i+1}: {(preds['rank'] <= i).sum() / len(preds) * 100 :.3f}")

@hydra.main(version_base=None, config_path="../configs", config_name="config_predict")
def main(config: DictConfig):

    seed_everything()
    print(config)


    if config.model.model_checkpoint_path is None:
        raise ValueError(
            "Please supply model_checkpoint_path with config.model_checkpoint_path=..."
        )

    # Load dataset
    data_config = config["data"].copy()
    print(data_config)
    data_config = OmegaConf.to_container(data_config, resolve=True)

    data_path = config["data_path"]
    cv_split = config["cv_split"]
    func_group_split = (
        config["func_group_split"] if "func_group_split" in config else False
    )
    smiles_split = config["smiles_split"] if "smiles_split" in config else False
    augment_path = config["augment_path"] if "augment_path" in config else None
    augment_names = config["augment_names"] if "augment_names" in config else None
    augment_model_config = (
        config["augment_model"] if "augment_model" in config else None
    )
    augment_fraction = (
        config["augment_fraction"] if "augment_fraction" in config else 0.0
    )

    data_config, dataset = build_dataset_multimodal(
        data_config,
        data_path=data_path,
        cv_split=cv_split,
        func_group_split=func_group_split,
        smiles_split=smiles_split,
        augment_path=augment_path,
        augment_fraction=augment_fraction,
        augment_names=augment_names,
        augment_model_config=augment_model_config,
    )
    logging.info("Build dataset")

    # Load/build tokenizers and preprocessors
    if config["preprocessor_path"] is None:
        preprocessor_path = (
            Path(config["working_dir"]) / config["job_name"] / "preprocessor.pkl"
        )
    else:
        preprocessor_path = Path(config["preprocessor_path"])

    if preprocessor_path.is_file():
        data_config, preprocessors = pd.read_pickle(preprocessor_path)
    else:
        data_config, preprocessors = load_preprocessors(dataset["train"], data_config)
        with open(preprocessor_path, "wb") as f:
            pickle.dump((data_config, preprocessors), f)
    logging.info("Build preprocessors")

    # Load datamodule
    model_type = config["model"]["model_type"]
    batch_size = config["model"]["batch_size"]
    mixture = config["mixture"]


    data_module = MultiModalDataModule(
        dataset=dataset,
        preprocessors=preprocessors,
        data_config=data_config,
        model_type=model_type,
        batch_size=batch_size,
        mixture=mixture
    )
    target_modality = data_module.collator.target_modality
    target_tokenizer = preprocessors[target_modality]
    logging.info("Build Datamodule")

    # Load Model
    train_steps = calculate_training_steps(data_module, config)
    checkpoint_path = config["model"]["model_checkpoint_path"]

    model = HFWrapper(
        data_config=data_config,
        target_tokenizer=target_tokenizer,
        num_steps=train_steps,
        **config["model"],
    )

    checkpoint_path = config["model"]["model_checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    # Evaluate model
    test_loader = data_module.test_dataloader(test_idx=Path("./CNMR_test_idx.npy"))

    predictions = list()
    ground_truth = list()

    n_beams =10
    decode_method = "generate"

    for i, batch in enumerate(tqdm.tqdm(test_loader)):
        batch["encoder_input"] = {
            modality: modality_input.to(device)
            for modality, modality_input in batch["encoder_input"].items()
        }
        batch["decoder_input"] = {
            modality: modality_input.to(device)
            for modality, modality_input in batch["decoder_input"].items()
        }

        for key in ["encoder_pad_mask", "decoder_pad_mask", "target_mask", "target"]:
            batch[key] = batch[key].to(device)

        
        if decode_method == "custom":

            with torch.no_grad():
                predictions_batch, log_lhs_batch = model.sample_molecules(
                    batch, n_beams=n_beams, sampling_alg="beam"
                )

            if isinstance(predictions_batch, torch.Tensor):
                predictions_batch = predictions_batch.cpu().tolist()
            if isinstance(log_lhs_batch, torch.Tensor):
                log_lhs_batch = log_lhs_batch.cpu().tolist()

            if "target_smiles" in batch:
                if isinstance(batch["target_smiles"], torch.Tensor):
                    ground_truth.extend(batch["target_smiles"].cpu().tolist())
                else:
                    ground_truth.extend(batch["target_smiles"])
            else:
                if isinstance(batch["target"], torch.Tensor):
                    ground_truth.extend(batch["target"].cpu().tolist())
                else:
                    ground_truth.extend(batch["target"])

            predictions.extend(predictions_batch)
        
        else:

            generated_sequences = model.generate(batch, n_beams=n_beams)

            detokenized_sequences = preprocessors[target_modality].batch_decode(
                generated_sequences, skip_special_tokens=True
            )
            detokenized_sequences = [
                detokenized_sequences[j * n_beams : (j + 1) * n_beams]
                for j in range(len(detokenized_sequences) // n_beams)
            ]

            predictions.extend(detokenized_sequences)
            ground_truth.extend(batch["target_smiles"])

    metrics = calc_sampling_metrics(predictions, ground_truth)
    logger.info(metrics) 
    
    save_path = (
        Path(config["working_dir"])
        / config["job_name"]
        / f"test_data_logits_beam_{n_beams}.pkl"
    )
    with (save_path).open("wb") as save_file:
        pickle.dump(
            {"predictions": predictions, "ground_truth": ground_truth},
            save_file,
        )  # To do: Move away from pickle

if __name__ == "__main__":
    main()
