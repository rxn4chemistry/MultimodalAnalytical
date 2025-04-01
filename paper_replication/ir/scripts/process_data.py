from pathlib import Path

import click
import numpy as np
import pandas as pd
import tqdm.auto as tqdm


def load_mm_dataset(data_path: Path) -> pd.DataFrame:
    """Loads and processes all IR spectra contained in the multimodal dataset."""

    all_chunks = list()
    for chunk_path in tqdm.tqdm(data_path.glob("*.parquet"), total=245):
        chunk = pd.read_parquet(chunk_path, columns=['molecular_formula', 'smiles', 'ir_spectra'])
        chunk['ir_spectra'] = chunk['ir_spectra'].map(lambda spec : spec[:1791].astype(np.float32))
        all_chunks.append(chunk)
    
    return pd.concat(all_chunks)


def load_synth_ir(data_path: Path) -> pd.DataFrame:
    """Loads and processes IR spectra."""

    synth_ir = pd.read_pickle(data_path)
    synth_ir = synth_ir.rename(columns={"formula": "molecular_formula", "spectra": "ir_spectra"})
    synth_ir['ir_spectra'] = synth_ir['ir_spectra'].map(lambda spec : spec.astype(np.float32))
    return synth_ir


@click.command()
@click.option("--data_folder", type=Path, required=True)
def main(data_folder: Path):

    print("Loading and processing IR spectra from Multimodal Dataset.")
    mm_ir_spectra = load_mm_dataset(data_folder / "raw_data" / "multimodal_spectroscopic_dataset")

    print("Loading and processing IR spectra from Synthetic IR dataset")
    synth_ir_spectra = load_synth_ir(data_folder / "raw_data" / "IRtoMol" / "data" / "ir_data.pkl")

    all_ir_data = pd.concat([mm_ir_spectra, synth_ir_spectra])
    all_ir_data = all_ir_data.drop_duplicates(subset='smiles')
    all_ir_data = all_ir_data.sample(frac=1, random_state=3245)

    print("Saving Data")
    pretraining_data_path = data_folder / "pretraining"
    pretraining_data_path.mkdir(parents=True, exist_ok=True)
    all_ir_data.to_parquet(pretraining_data_path / "pretrain_data.parquet")



if __name__ == '__main__':
    main()
