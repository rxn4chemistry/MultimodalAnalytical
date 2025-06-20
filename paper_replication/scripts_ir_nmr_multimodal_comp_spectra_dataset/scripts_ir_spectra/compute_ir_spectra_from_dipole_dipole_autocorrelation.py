#!/usr/bin/env python
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack, signal


def gen_spectrum_me1(filedipole, dipole_np, save_dir, md_timestep_in_fs) -> None:
    # Inputs
    # T = 300.0  # K
    output_autocorrelation_file = Path(save_dir) / "autocorr_orig_"+filedipole+".txt"

    # Constants
    # boltz = 1.38064852e-23  # m^2 kg s^-2 K^-1
    # lightspeed = 299792458.0  # m s^-1
    # reduced_planck = 1.05457180013e-34  # kg m^2 s^-1

    n_points = dipole_np.shape[0]
    nq = n_points
    data_points = int(nq / 2) - 1

    ###############################################################################
    # Get autocorrelation function
    ###############################################################################
    # Calculate autocorrelation function
    # Load data
    time = [ i*md_timestep_in_fs for i in range(n_points)]
    dipole_x = dipole_np[:,0]
    dipole_y = dipole_np[:,1]
    dipole_z = dipole_np[:,2]

    # Do calculation
    # Note that this method of calculating an autocorrelation function is very fast, but it can be difficult to follow.
    # For readability, I've presented a more straightforward (but much, much slower) method in the commented block below.
    # Shift the array
    if len(time) % 2 == 0:
        dipole_x_shifted = np.zeros(len(time) * 2)
        dipole_y_shifted = np.zeros(len(time) * 2)
        dipole_z_shifted = np.zeros(len(time) * 2)
    else:
        dipole_x_shifted = np.zeros(len(time) * 2 - 1)
        dipole_y_shifted = np.zeros(len(time) * 2 - 1)
        dipole_z_shifted = np.zeros(len(time) * 2 - 1)
    dipole_x_shifted[len(time) // 2 : len(time) // 2 + len(time)] = dipole_x
    dipole_y_shifted[len(time) // 2 : len(time) // 2 + len(time)] = dipole_y
    dipole_z_shifted[len(time) // 2 : len(time) // 2 + len(time)] = dipole_z
    # Convolute the shifted array with the flipped array, which is equivalent to performing a correlation
    autocorr_x_full = signal.fftconvolve(
        dipole_x_shifted, dipole_x[::-1], mode="same"
    )[(-len(time)) :] / np.arange(len(time), 0, -1)
    autocorr_y_full = signal.fftconvolve(
        dipole_y_shifted, dipole_y[::-1], mode="same"
    )[(-len(time)) :] / np.arange(len(time), 0, -1)
    autocorr_z_full = signal.fftconvolve(
        dipole_z_shifted, dipole_z[::-1], mode="same"
    )[(-len(time)) :] / np.arange(len(time), 0, -1)
    autocorr_full = autocorr_x_full + autocorr_y_full + autocorr_z_full
    # Truncate the autocorrelation array
    autocorr = autocorr_full[: int(data_points)]

    # Save data
    np.savetxt(
        output_autocorrelation_file,
        np.column_stack((time[: len(autocorr)], autocorr)),
        header="Time(fs) Autocorrelation(e*Ang)",
    )
       
    return np.column_stack((time[: len(autocorr)], autocorr))


def gen_spectrum_me2(save_path, autocorr_xy, md_timestep_in_fs) -> None:
    # Inputs
    # autocorrelation_option = 2  # 1 to calculate it, 2 to load a pre-calculated one
    temp = 300.0  # K

    # Constants
    boltz = 1.38064852e-23  # m^2 kg s^-2 K^-1
    lightspeed = 299792458.0  # m s^-1
    reduced_planck = 1.05457180013e-34  # kg m^2 s^-1

    # n_points = autocorr_xy.shape[0]
    # nq = n_points
    # data_points = int(nq / 2) - 1

    # time = autocorr_xy[:,0]
    autocorr = autocorr_xy[:,1]

    timestep = (md_timestep_in_fs) * 1.0e-15  # converts time from femtoseconds to seconds

    ###############################################################################
    # Calculate spectra
    # Note that intensities are relative, and so can be multiplied by a constant to compare to experiment.
    ###############################################################################
    # Calculate the FFTs of autocorrelation functions
    lineshape = fftpack.dct(autocorr, type=1)[1:]
    lineshape_frequencies = np.linspace(0, 0.5 / timestep, len(autocorr))[1:]
    lineshape_frequencies_wn = lineshape_frequencies / (
        100.0 * lightspeed
    )  # converts to wavenumbers (cm^-1)

    # Calculate spectra
    field_description = lineshape_frequencies * (
        1.0 - np.exp(-reduced_planck * lineshape_frequencies / (boltz * temp))
    )
    quantum_correction = lineshape_frequencies / (
        1.0 - np.exp(-reduced_planck * lineshape_frequencies / (boltz * temp))
    )
    # quantum correction per doi.org/10.1021/jp034788u. Other options are possible, see doi.org/10.1063/1.441739 and doi.org/10.1080/00268978500102801.
    spectra = lineshape * field_description
    spectra_qm = spectra * quantum_correction

    # Save data
    np.savetxt(
        save_path,
        np.column_stack(
            (
                lineshape_frequencies_wn,
                lineshape,
                field_description,
                quantum_correction,
                spectra,
                spectra_qm,
            )
        ),
        header="Frequency(cm^-1), Lineshape, Field_description, Quantum_correction, Spectra, Spectra_qm",
        delimiter=",",
    )
    return


# Blackman window function
def blackman_window(nn):
    n = np.arange(nn)
    return 0.42 - 0.5 * np.cos(2 * np.pi * n / (nn - 1)) + 0.08 * np.cos(4 * np.pi * n / (nn - 1))


# Function to apply damping to the last part of the signal
def damp_last_part(x, y, damping_fraction=0.2):
    n = len(x)
    window = blackman_window(n)
    damped_window = np.ones(n)
    damped_window[int((1 - damping_fraction) * n):] = window[int((1 - damping_fraction) * n):]
    y_damped = y * damped_window
    # return y_damped
    return np.column_stack((x, y_damped))


def plot_auto(filedipole, autocorr, autocorr_damp, save_dir):
    x = autocorr[:,0]
    y = autocorr[:,1]
    y_damped_last_part = autocorr_damp[:,1]
    header_ = "# Time(fs) Autocorrelation(e*Ang)"

    output_filename = Path(save_dir) / "autocorr_damp_"+filedipole+".txt"
    np.savetxt(output_filename, np.column_stack((x, y_damped_last_part)), header=header_, comments='', delimiter=' ')
    plot_filename = Path(save_dir) / "plot_autocorr_"+filedipole+".png"
    
    # Plotting the original vs. damped signal
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Original Signal", linestyle='--', color='blue')
    plt.plot(x, y_damped_last_part, label="Signal with damping", color='green')
    plt.title("Signal with Blackman window applied")
    plt.xlabel("Time (fs)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_filename, dpi=400, bbox_inches='tight')
    # plt.show()
    plt.close()
    return


if __name__ == "__main__":

    # `filedipole` contains the dipole x,y,z
    # shape is n_points x 3
    # the timestep in fs need to be given in input

    if len(sys.argv) < 3:
        print("Usage: python script_ir_dipole_me...py <dipole.npy> <md_timestep_in_fs>")
        sys.exit(1)

    filedipole = sys.argv[1]
    save_dir = Path(filedipole)
    md_timestep_in_fs = float(sys.argv[2])

    dipole_np = np.load(filedipole)
    n_points = dipole_np.shape[0]
    filedipole = Path(filedipole)
    filedipole = Path(filedipole.replace(".npy", ""))

    print("reading from", filedipole, "which contains", n_points, "lines and md_timestep_in_fs = ", md_timestep_in_fs)

    output_autocorrelation_file = Path(save_dir) / "autocorr_orig_"+filedipole+".txt"
    save_path = Path(save_dir) / "IR-data_"+filedipole+"_auto_damped.csv"
    csv_path = save_path
    print("save_path:", save_path)
    output_filename = Path(save_dir) / "autocorr_damp_"+filedipole+".txt"
    plot_filename = Path(save_dir) / "plot_autocorr_"+filedipole+".png"
    autocorr = gen_spectrum_me1(filedipole, dipole_np, save_dir, md_timestep_in_fs)
    autocorr_damp = damp_last_part(autocorr[:,0], autocorr[:,1], damping_fraction=0.5)
    gen_spectrum_me2(csv_path, autocorr_damp, md_timestep_in_fs)
    plot_auto(filedipole, autocorr, autocorr_damp, save_dir)

