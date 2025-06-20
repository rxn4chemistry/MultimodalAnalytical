#!/usr/bin/env python
import os

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spectra(record):
    freq = np.array(record["Frequency(cm^-1)"])
    ir_spectra = np.array(record["ir_spectra"])
    fact = np.max(np.abs(ir_spectra))
    plt.figure(figsize=(10, 6))
    plt.plot(
        freq, 1 / fact * ir_spectra, label=record["smiles"], color="blue", linewidth=1
    )
    plt.xlabel("Frequency (cm^-1)", fontsize=14)
    plt.ylabel("IR Spectra [arb. units]", fontsize=14)
    plt.xlim(100, 4000)
    plt.legend()
    plt.grid(True)
    output_filename = "ir_spectra_id_" + str(record["id"]) + ".png"
    plt.savefig(output_filename, dpi=400, bbox_inches="tight")
    plt.close()
    return


def load_df(path_to_parquet_folder):
    base_path = Path(path_to_parquet_folder)
    files = [
        os.path.join(base_path / "./IR_data_chunk001_of_009.parquet"),
        os.path.join(base_path / "./IR_data_chunk002_of_009.parquet"),
        os.path.join(base_path / "./IR_data_chunk003_of_009.parquet"),
        os.path.join(base_path / "./IR_data_chunk004_of_009.parquet"),
        os.path.join(base_path / "./IR_data_chunk005_of_009.parquet"),
        os.path.join(base_path / "./IR_data_chunk006_of_009.parquet"),
        os.path.join(base_path / "./IR_data_chunk007_of_009.parquet"),
        os.path.join(base_path / "./IR_data_chunk008_of_009.parquet"),
        os.path.join(base_path / "./IR_data_chunk009_of_009.parquet"),
    ]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


if __name__ == "__main__":
    # set the `path_to_parquet_folder`
    path_to_parquet_folder = (
        "../zenodo_data/"
    )

    # load the dataset
    df_ir = load_df(path_to_parquet_folder)

    record = df_ir.iloc[0]  # select the first record
    plot_spectra(record)
