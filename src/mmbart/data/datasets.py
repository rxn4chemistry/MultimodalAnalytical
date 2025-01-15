import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from rdkit import Chem  # RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch import nn

from mmbart.defaults import (
    DEFAULT_SEED,
    DEFAULT_TEST_SET_SIZE,
    DEFAULT_VAL_SET_SIZE,
)

valid_modalities = ["text", "vector"]


functional_groups = {
    "Acid anhydride": Chem.MolFromSmarts("[CX3](=[OX1])[OX2][CX3](=[OX1])"),
    "Acyl halide": Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]"),
    "Alcohol": Chem.MolFromSmarts("[#6][OX2H]"),
    "Aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6,H]"),
    "Alkane": Chem.MolFromSmarts("[CX4;H3,H2]"),
    "Alkene": Chem.MolFromSmarts("[CX3]=[CX3]"),
    "Alkyne": Chem.MolFromSmarts("[CX2]#[CX2]"),
    "Amide": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
    "Amine": Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]"),
    "Arene": Chem.MolFromSmarts("[cX3]1[cX3][cX3][cX3][cX3][cX3]1"),
    "Azo compound": Chem.MolFromSmarts("[#6][NX2]=[NX2][#6]"),
    "Carbamate": Chem.MolFromSmarts("[NX3][CX3](=[OX1])[OX2H0]"),
    "Carboxylic acid": Chem.MolFromSmarts("[CX3](=O)[OX2H]"),
    "Enamine": Chem.MolFromSmarts("[NX3][CX3]=[CX3]"),
    "Enol": Chem.MolFromSmarts("[OX2H][#6X3]=[#6]"),
    "Ester": Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
    "Ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
    "Haloalkane": Chem.MolFromSmarts("[#6][F,Cl,Br,I]"),
    "Hydrazine": Chem.MolFromSmarts("[NX3][NX3]"),
    "Hydrazone": Chem.MolFromSmarts("[NX3][NX2]=[#6]"),
    "Imide": Chem.MolFromSmarts("[CX3](=[OX1])[NX3][CX3](=[OX1])"),
    "Imine": Chem.MolFromSmarts(
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]"
    ),
    "Isocyanate": Chem.MolFromSmarts("[NX2]=[C]=[O]"),
    "Isothiocyanate": Chem.MolFromSmarts("[NX2]=[C]=[S]"),
    "Ketone": Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
    "Nitrile": Chem.MolFromSmarts("[NX1]#[CX2]"),
    "Phenol": Chem.MolFromSmarts("[OX2H][cX3]:[c]"),
    "Phosphine": Chem.MolFromSmarts("[PX3]"),
    "Sulfide": Chem.MolFromSmarts("[#16X2H0]"),
    "Sulfonamide": Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]"),
    "Sulfonate": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]"),
    "Sulfone": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[#6]"),
    "Sulfonic acid": Chem.MolFromSmarts("[#16X4](=[OX1])(=[OX1])([#6])[OX2H]"),
    "Sulfoxide": Chem.MolFromSmarts("[#16X3]=[OX1]"),
    "Thial": Chem.MolFromSmarts("[CX3H1](=S)[#6,H]"),
    "Thioamide": Chem.MolFromSmarts("[NX3][CX3]=[SX1]"),
    "Thiol": Chem.MolFromSmarts("[#16X2H]"),
}


class MLP_Bottleneck(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_layers: int,
        bottle_neck_dim: int,
        final_activation: str = "exp",
    ) -> None:
        super().__init__()

        io_dim = 1625
        self.encoder = nn.Sequential(
            *self.make_mlp_component(in_dim, bottle_neck_dim, n_layers)
        )
        self.decoder = nn.Sequential(
            *self.make_mlp_component(bottle_neck_dim, io_dim, n_layers)[:-1]
        )

        self.final_activation = final_activation

        if self.final_activation == "abs_smoothing":
            self.smoother = nn.Conv1d(1, 1, 5, padding=2)

    def make_mlp_component(
        self, input_dim: int, output_dim: int, n_layers: int
    ) -> List[Any]:
        layers = list()

        io_dims = np.linspace(input_dim, output_dim, n_layers + 1).astype(int)
        for i in range(n_layers):
            layer = nn.Linear(in_features=io_dims[i], out_features=io_dims[i + 1])
            layers.append(layer)
            layers.append(nn.ReLU())  # type: ignore

        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        out = self.decoder(x)

        if self.final_activation == "exp":
            return torch.exp(out)
        elif self.final_activation == "sigmoid":
            return torch.sigmoid(out)
        elif self.final_activation == "none":
            return out
        elif self.final_activation == "abs":
            return torch.abs(out)
        elif self.final_activation == "abs_normed":
            spec_abs = torch.abs(out)
            return spec_abs / torch.max(spec_abs, -1)[0].unsqueeze(-1)
        elif self.final_activation == "abs_smoothing":
            smooth_spec = self.smoother(out.view(-1, 1, 1625))
            return torch.abs(smooth_spec.view(-1, 1625))
        else:
            raise ValueError(f"{self.final_activation}: Unknown Activation function.")


class CNN_1D(nn.Module):

    def __init__(self, final_activation: str = "exp"):

        super().__init__()
        self.io_dim = 1625

        self.downsample_1 = self.make_downsample_block(1, 32)
        self.downsample_2 = self.make_downsample_block(32, 64)

        self.upsample_1 = self.make_upsample_block(64, 32)
        self.upsample_2 = self.make_upsample_block(32, 1)

        self.final_layer = nn.Linear(1592, 1625)
        self.final_activation = final_activation

    def make_downsample_block(self, in_channels: int, out_channels: int):
        conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=2)
        batch_norm = nn.BatchNorm1d(out_channels)
        act_fn = nn.ReLU()
        sample_layer = nn.AvgPool1d(3, stride=2)
        conv_block = nn.Sequential(*[conv, batch_norm, act_fn, sample_layer])
        return conv_block

    def make_upsample_block(self, in_channels: int, out_channels: int):
        conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        batch_norm = nn.BatchNorm1d(out_channels)
        act_fn = nn.ReLU()
        sample_layer = nn.Upsample(scale_factor=8, mode="nearest")
        conv_block = nn.Sequential(*[conv, batch_norm, act_fn, sample_layer])
        return conv_block

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.downsample_1(x)
        x = self.downsample_2(x)

        x = self.upsample_1(x)
        x = self.upsample_2(x)

        x = x.view(x.shape[0], -1)
        x = self.final_layer(x)

        if self.final_activation == "exp":
            return torch.exp(x)
        elif self.final_activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.final_activation == "abs":
            return torch.abs(x)
        elif self.final_activation == "none":
            return x


class MLP3(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5):
        super(MLP3, self).__init__()
        layers = []
        current_input_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())  # type: ignore
            layers.append(nn.Dropout(dropout_prob))  # type: ignore
            current_input_size = hidden_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RealSpecAugmentation:

    def __init__(self, model_path: Path, mol_info: str, needs_init: bool = False):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if needs_init:
            hidden_sizes = [1024, 64, 1024]
            output_size = 1625
            dropout_prob = 0.5  # Dropout probability
            model = MLP3(1625, hidden_sizes, output_size, dropout_prob)
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device(self.device))
            )
        else:
            model = torch.load(model_path, map_location=torch.device(self.device))

        self.mol_info = mol_info
        self.model = model.to(self.device)

    def process_noisy_double(self, array):
        new_array = np.copy(array)
        mean_odd = np.mean(array[0::2])
        mean_even = np.mean(array[1::2])
        std_odd = np.std(array[0::2])
        std_even = np.std(array[1::2])

        thr = abs(mean_odd - mean_even) / (std_odd + std_even)
        if thr > 0.3:
            n = len(array)
            for i in range(n - 1):
                new_array[i] = (array[i] + array[i + 1]) * 0.5
            new_array[n - 1] = (array[n - 2] + array[n - 1]) * 0.5

        new_array = new_array / max(new_array)
        return new_array

    def preprocess_gaussian(self, spec, sigma):
        blur_spec = gaussian_filter1d(spec, sigma=sigma)
        blur_spec = blur_spec / max(blur_spec)
        blur_spec[blur_spec < 0] = 0
        return blur_spec

    def match_group(self, mol: Chem.Mol, func_group) -> int:
        if isinstance(func_group, Chem.Mol):
            n = len(mol.GetSubstructMatches(func_group))
        else:
            n = func_group(mol)
        return n

    def get_functional_groups(self, smiles: str) -> torch.Tensor:
        # RDLogger.DisableLog("rdApp.*")
        smiles = smiles.strip().replace(" ", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid smiles: {smiles}")
        func_groups = list()
        for func_group_name, smarts in functional_groups.items():
            func_groups.append(self.match_group(mol, smarts))

        return torch.Tensor(func_groups)

    def get_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5, fpSize=2048)
        return torch.Tensor(fp_gen.GetFingerprint(mol))

    def postprocess_spectra(self, spec):
        spec = spec / max(spec)
        spec = np.flip(gaussian_filter1d(spec, sigma=0.75))
        spec = np.concatenate((np.zeros(125), spec, np.zeros(41)))
        return spec

    def process_batch(self, batch, remove_stereo: bool = True):

        preprocessed_spectra = list()

        no_stereo_smiles = list()
        for smiles in batch["smiles"]:
            mol = Chem.MolFromSmiles(smiles)
            Chem.RemoveStereochemistry(mol)
            no_stereo_smiles.append(Chem.MolToSmiles(mol))

        for i, spec in enumerate(batch["ir_spectra"]):
            proc_spec = self.process_noisy_double(spec[125:-41])
            proc_spec = self.preprocess_gaussian(proc_spec, sigma=1)
            proc_spec = torch.Tensor(proc_spec)

            if self.mol_info == "combined":
                fp = self.get_fingerprint(no_stereo_smiles[i])
                func_group = self.get_functional_groups(no_stereo_smiles[i])
                proc_spec = torch.concat([func_group, fp, proc_spec])

            preprocessed_spectra.append(proc_spec)

        preprocessed_spectra_tensor = torch.vstack(preprocessed_spectra).to(self.device)

        transformed_spectra = self.model(preprocessed_spectra_tensor).detach().cpu()

        postprocessed_spectra = list()
        for spec in transformed_spectra:
            postprocessed_spectra.append(self.postprocess_spectra(spec))

        return {"smiles": no_stereo_smiles, "ir_spectra": postprocessed_spectra}


def build_dataset(
    dataset_type: str,
    dataset_path: Optional[Path] = None,
) -> Dataset:
    if dataset_type == "json":
        if dataset_path is None:
            raise ValueError(
                "dataset_path needs to be supplied if using dataset_type json."
            )

        dataset = build_dataset_from_json(dataset_path)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    return dataset


def build_dataset_from_json(dataset_path: Path) -> DatasetDict:
    train_path, test_path, val_path = (
        dataset_path / "data_train.json",
        dataset_path / "data_test.json",
        dataset_path / "data_val.json",
    )

    train_data, test_data = Dataset.from_json(str(train_path)), Dataset.from_json(
        str(test_path)
    )

    if val_path.exists():
        val_data = Dataset.from_json(str(val_path))
    else:
        split_train_data = test_data.train_test_split(
            test_size=min(len(test_data), DEFAULT_VAL_SET_SIZE)
        )
        train_data, val_data = split_train_data["train"], split_train_data["test"]

    dataset = DatasetDict(
        {"train": train_data, "validation": val_data, "test": test_data}
    )

    return dataset


def split(dataset: Dataset, cv_split: int = 0) -> DatasetDict:
    all_indices_shuffled = np.arange(0, len(dataset), 1)
    np.random.shuffle(all_indices_shuffled)

    splits = np.array_split(all_indices_shuffled, int(1 / DEFAULT_TEST_SET_SIZE))
    test_indices = splits[cv_split]
    train_indices = all_indices_shuffled[~np.isin(all_indices_shuffled, test_indices)]

    test_set = dataset.select(test_indices)
    train_set = dataset.select(train_indices)

    split_data_val = train_set.train_test_split(
        test_size=min(int(0.1 * len(train_set)), DEFAULT_VAL_SET_SIZE),
        shuffle=True,
        seed=DEFAULT_SEED,
    )

    return DatasetDict(
        {
            "train": split_data_val["train"],
            "test": test_set,
            "validation": split_data_val["test"],
        }
    )


def func_split(data_path, cv_split: int = 0, seed: int = 3453) -> DatasetDict:

    data_path = Path(data_path)
    parquet_paths = data_path.glob("*.parquet")

    data_chunks = list()
    for parquet_path in parquet_paths:
        chunk = pd.read_parquet(parquet_path)
        data_chunks.append(chunk)
    data = pd.concat(data_chunks)

    data["functional_group_names"] = data["functional_group_names"].apply(
        lambda x: ".".join(sorted(x))
    )

    counts: Dict[str, int] = {}
    for sample in data["functional_group_names"]:
        if sample in counts.keys():
            counts[sample] += 1
        else:
            counts[sample] = 1

    counts_df = pd.DataFrame(counts.items(), columns=["functional_groups", "counts"])
    single_counts = counts_df[counts_df["counts"] == 1].copy()
    multi_counts = counts_df[counts_df["counts"] > 1].copy()

    single_counts_df = data[
        data["functional_group_names"].isin(single_counts["functional_groups"])
    ]
    multi_counts_df = data[
        data["functional_group_names"].isin(multi_counts["functional_groups"])
    ]

    if cv_split == -1:
        train_set, test_set = train_test_split(
            multi_counts_df,
            stratify=multi_counts_df["functional_group_names"],
            test_size=0.1,
            random_state=3453,
            shuffle=True,
        )
    else:
        k_folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        splits = list(
            k_folds.split(
                X=multi_counts_df, y=multi_counts_df["functional_group_names"]
            )
        )
        train_indices, test_indices = splits[cv_split][0], splits[cv_split][1]
        train_set, test_set = (
            multi_counts_df.iloc[train_indices],
            multi_counts_df.iloc[test_indices],
        )

    train_set, val_set = train_test_split(
        train_set,
        test_size=min(int(0.05 * len(train_set)), DEFAULT_VAL_SET_SIZE),
        random_state=seed,
        shuffle=True,
    )
    train_set = pd.concat([train_set, single_counts_df])

    train_set = Dataset.from_pandas(train_set)
    val_set = Dataset.from_pandas(val_set)
    test_set = Dataset.from_pandas(test_set)

    return DatasetDict({"train": train_set, "test": test_set, "validation": val_set})


def smiles_split_fn(
    data_path: Path, target_column: str, cv_split: int = 0, seed: int = 3453
):
    data_path = Path(data_path)
    parquet_paths = data_path.glob("*.parquet")

    data_chunks = list()
    for parquet_path in parquet_paths:
        chunk = pd.read_parquet(parquet_path)
        data_chunks.append(chunk)
    data = pd.concat(data_chunks)

    unique_smiles = pd.unique(data[target_column])

    k_folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(k_folds.split(X=unique_smiles))
    train_indices, test_indices = splits[cv_split][0], splits[cv_split][1]
    train_smiles, test_smiles = (
        unique_smiles[train_indices],
        unique_smiles[test_indices],
    )

    train_set, test_set = (
        data[data[target_column].isin(train_smiles)],
        data[data[target_column].isin(test_smiles)],
    )
    train_set, val_set = train_test_split(
        train_set,
        test_size=min(int(0.05 * len(train_set)), DEFAULT_VAL_SET_SIZE),
        random_state=seed,
        shuffle=True,
    )

    train_set = Dataset.from_pandas(train_set)
    val_set = Dataset.from_pandas(val_set)
    test_set = Dataset.from_pandas(test_set)

    return DatasetDict({"train": train_set, "test": test_set, "validation": val_set})


def interpolate(spec, old_x):
    interp = interp1d(old_x, spec)
    new_x = np.linspace(400, 3980, 1791)
    new_spec = interp(new_x)
    return new_spec


def horizontal_shift_augment(spectrum: np.ndarray):

    old_x = np.linspace(400, 3982, 895)

    spec_shift_1 = spectrum[:-1:2]
    spec_shift_1 = interpolate(spec_shift_1, old_x)

    spec_shift_2 = spectrum[1::2]
    spec_shift_2 = interpolate(spec_shift_2, old_x)

    return [spec_shift_1, spec_shift_2]


def horizontal_shift_augment3(spectrum: np.ndarray):

    old_x = np.linspace(400, 3982, 597)

    spec_shift_1 = spectrum[:-2:3]
    spec_shift_1 = interpolate(spec_shift_1, old_x)

    spec_shift_2 = spectrum[1::3]
    spec_shift_2 = interpolate(spec_shift_2, old_x)

    spec_shift_3 = spectrum[2::3]
    spec_shift_3 = interpolate(spec_shift_3, old_x)

    return [spec_shift_1, spec_shift_2, spec_shift_3]


def horizontal_shift_augment4(spectrum: np.ndarray):

    old_x = np.linspace(400, 3982, 447)

    spec_shift_1 = spectrum[:-4:4]
    spec_shift_1 = interpolate(spec_shift_1, old_x)

    spec_shift_2 = spectrum[1:-3:4]
    spec_shift_2 = interpolate(spec_shift_2, old_x)

    spec_shift_3 = spectrum[2:-2:4]
    spec_shift_3 = interpolate(spec_shift_3, old_x)

    spec_shift_4 = spectrum[3::4]
    spec_shift_4 = interpolate(spec_shift_4, old_x)

    return [spec_shift_1, spec_shift_2, spec_shift_3, spec_shift_4]


def augment_smooth(
    spectrum: np.ndarray, sigmas: Optional[List[float]] = None
) -> List[np.ndarray]:
    if sigmas is None:
        sigmas = [0.75, 1.25]
    smoothed_spectra = list()
    for sigma in sigmas:
        smooth_spectrum = gaussian_filter1d(spectrum, sigma)
        smoothed_spectra.append(smooth_spectrum)

    return smoothed_spectra


def augment(row, augment_names):

    spectra = row["IR"][0]
    augmented_spectra = [spectra]
    for augment_type in augment_names:
        if augment_type == "horizontal":
            augmented_spectra.extend(horizontal_shift_augment(spectra))
        elif augment_type == "horizontal3":
            augmented_spectra.extend(horizontal_shift_augment3(spectra))
        elif augment_type == "horizontal4":
            augmented_spectra.extend(horizontal_shift_augment4(spectra))
        elif augment_type == "smooth":
            augmented_spectra.extend(augment_smooth(spectra))
        elif augment_type == "smooth075":
            augmented_spectra.extend(augment_smooth(spectra, sigmas=[0.75]))
        elif augment_type == "smooth100":
            augmented_spectra.extend(augment_smooth(spectra, sigmas=[1.0]))
        elif augment_type == "smooth125":
            augmented_spectra.extend(augment_smooth(spectra, sigmas=[1.25]))
        elif augment_type == "smooth150":
            augmented_spectra.extend(augment_smooth(spectra, sigmas=[1.5]))
        elif augment_type == "smooth175":
            augmented_spectra.extend(augment_smooth(spectra, sigmas=[1.75]))
        else:
            raise ValueError(f"Unknown augmentation {augment_type}")

    return {"Smiles": row["Smiles"] * len(augmented_spectra), "IR": augmented_spectra}


def remove_stereo_flip_spec(row):
    mol = Chem.MolFromSmiles(row["smiles"])
    Chem.RemoveStereochemistry(mol)
    row["smiles"] = Chem.MolToSmiles(mol)
    row["ir_spectra"] = np.flip(np.array(row["ir_spectra"]))
    return row


def augment_model(dataset: Dataset, augment_model_config: Dict[str, Any]):
    real_augmenter = RealSpecAugmentation(
        model_path=Path(augment_model_config["augment_model_path"]),
        mol_info=augment_model_config["augment_model_mol_info"],
        needs_init=augment_model_config["augment_model_init"],
    )

    if (
        augment_model_config["augment_fraction"] < 1.0
        and not augment_model_config["augment_replace_orig"]
    ):
        augment_set = dataset.train_test_split(
            test_size=augment_model_config["augment_fraction"], seed=DEFAULT_SEED
        )["test"]
    else:
        augment_set = dataset

    augment_set = augment_set.map(
        real_augmenter.process_batch,
        batched=True,
        batch_size=512,
        load_from_cache_file=False,
    )

    if augment_model_config["augment_replace_orig"]:
        return augment_set
    else:
        dataset = dataset.map(remove_stereo_flip_spec, num_proc=7)
        return concatenate_datasets([augment_set, dataset])


def build_dataset_multimodal(
    data_config: Dict[str, Any],
    data_path: Path,
    cv_split: int,
    test_only: bool = False,
    func_group_split: bool = False,
    smiles_split: bool = False,
    augment_names: Optional[str] = None,
    augment_path: Optional[str] = None,
    augment_fraction: float = 0.0,
    augment_model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Union[str, int, bool]], DatasetDict]:
    if not os.path.isdir(data_path):
        raise ValueError(
            "Data path must specify path to directory containing the dataset files as parqet."
        )

    dataset_dict = load_dataset("parquet", data_dir=data_path)

    if test_only:
        datasets = list(dataset_dict.values())
        combined_dataset = concatenate_datasets(datasets)
        dataset_dict = DatasetDict({"test": combined_dataset})
    elif func_group_split:
        dataset_dict = func_split(data_path, cv_split=cv_split, seed=DEFAULT_SEED)
    elif smiles_split:
        target_column = ""
        for modality_config in data_config.values():
            if modality_config["target"]:
                target_column = modality_config["column"]
                break
        dataset_dict = smiles_split_fn(
            data_path, target_column, cv_split=cv_split, seed=DEFAULT_SEED
        )
    elif len(dataset_dict) == 1:
        dataset = list(dataset_dict.values())[0]

        if augment_model_config is not None and augment_model_config["apply"]:
            dataset = augment_model(dataset, augment_model_config)

        dataset_dict = split(dataset, cv_split)
    # If there are three datasets in the data dict check if they have the right keys.
    elif len(dataset_dict) == 3 and set(dataset_dict.keys()) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            f"Expected ['train', 'validation', 'test'] in dataset but found {list(dataset_dict.keys())}."
        )
    elif len(dataset_dict) != 3:
        raise ValueError(
            f"Excpected to find three datasets with name ['train', 'validation', 'test'] but found {len(dataset_dict)} with names {list(dataset_dict.keys())}."
        )

    if augment_path is not None:
        augment_data = load_from_disk(augment_path)
        sample_indices = np.random.choice(
            range(len(augment_data)), int(len(augment_data) * augment_fraction)
        )
        sampled_augment_data = augment_data.select(sample_indices)

        dataset_dict["train"] = concatenate_datasets(
            [dataset_dict["train"], sampled_augment_data]
        )
        dataset_dict["train"] = dataset_dict["train"].shuffle()

    relevant_columns = set()
    rename_columns = dict()

    extracted_config = dict()
    for modality in data_config.keys():
        if isinstance(data_config[modality]["column"], str):
            relevant_columns.add(data_config[modality]["column"])
            rename_columns[data_config[modality]["column"]] = modality

            extracted_config[data_config[modality]["column"]] = data_config[modality]
            extracted_config[data_config[modality]["column"]].pop("column")

        elif isinstance(data_config[modality]["column"], list):
            relevant_columns.update(data_config[modality]["column"])
            extracted_config[modality] = data_config[modality]

        else:
            raise ValueError(
                f"Expected column to be either list or str for modality: {modality}"
            )

    existing_columns = set(dataset_dict[list(dataset_dict.keys())[0]].column_names)
    columns_to_drop = existing_columns.difference(relevant_columns)

    processed_dataset_dict = DatasetDict()
    for dataset_key in dataset_dict.keys():
        selected_dataset = dataset_dict[dataset_key]
        processed_dataset = selected_dataset.remove_columns(columns_to_drop)
        processed_dataset = selected_dataset.rename_columns(rename_columns)

        processed_dataset_dict[dataset_key] = processed_dataset

    if augment_names is not None and not test_only:
        augment_names = augment_names.split("_")  # type: ignore
        augment_fn = partial(augment, augment_names=augment_names)
        augmented_train_set = processed_dataset_dict["train"].map(
            augment_fn,
            batched=True,
            batch_size=1,
            remove_columns=processed_dataset_dict["train"].column_names,
            num_proc=7,
        )
        processed_dataset_dict["train"] = augmented_train_set

    return data_config, processed_dataset_dict
