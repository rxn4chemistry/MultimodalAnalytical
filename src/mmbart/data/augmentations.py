from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import (
    Dataset,
    concatenate_datasets,
)
from rdkit import Chem  # RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from torch import nn



class MLP_Bottleneck(nn.Module): # noqa: N801
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


class CNN_1D(nn.Module): # noqa: N801

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

    def process_batch(self, batch, remove_stereo: bool = True): # noqa: ARG002

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
