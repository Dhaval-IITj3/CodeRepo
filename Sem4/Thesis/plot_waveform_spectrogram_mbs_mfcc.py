import csv

import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil
import librosa
import re

# Safety: Prevent accidental shadowing of built-ins
import builtins
for name in ['bin', 'dir', 'id', 'max', 'min', 'sum', 'list', 'dict']:
    if name in globals():
        raise NameError(f"STOP! You used '{name}' as a variable name. It's a built-in function!")

"""
✅ Rule of thumb:
Use Librosa when your main goal is signal analysis, feature extraction, and visualization.
Use Torchaudio when you’re building a deep learning model in PyTorch and want an efficient, end-to-end GPU pipeline.
"""

# Frequency bin configuration
SR = 22050              # sampling rate
N_FFT = 1024            # FFT window size
N_BINS = 20              # number of frequency bins (can tune)
HOP_LENGTH = 512

N_MFCC = 20         # number of MFCCs to extract

EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR_NAME = 'Data'
BASE_DIR = Path(__file__).parent
BASE_DIR = Path(BASE_DIR, ".").resolve()
DATA_DIR = Path.joinpath(BASE_DIR, DATA_DIR_NAME).absolute()
ENGINE_SOUND_DIR = Path.joinpath(BASE_DIR, Path(DATA_DIR_NAME), 'EngineSoundsGradualChanges')

# Extract engine names from file names
engines = []
engine_dir_path = []


def compute_mbe(filepath, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bins=N_BINS):
    """Compute Multiband Energy (histogram of power in frequency bins)."""
    # Load
    y, _ = librosa.load(filepath, sr=sr, mono=True)

    # STFT → magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2  # power spectrum

    # Frequency axis
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Bin edges (linear scale)
    bin_edges = np.logspace(np.log10(max(20, sr/1000)), np.log10(sr//2), n_bins + 1)

    energies = []
    bin_centers = []
    for i in range(n_bins):
        f_low = bin_edges[i]
        f_high = bin_edges[i + 1]
        mask = (freqs >= f_low) & (freqs < f_high)
        if np.any(mask):
            energy = np.sum(S[mask, :])
        else:
            energy = 1e-10  # avoid zero
        energies.append(energy)
        bin_centers.append(np.sqrt(f_low * f_high))  # geometric mean

    return np.array(energies), np.array(bin_centers), bin_edges


def create_plot_grid(total_plots):
    num_cols = 1
    while num_cols ** 2 < total_plots:
        num_cols += 1

    num_rows = (total_plots + num_cols - 1) // num_cols

    return num_cols, num_rows


def natural_sort_key(s):
    """
    Key function for natural sorting.
    Use with sorted(..., key=natural_sort_key)
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(s))]


def safe_bar_width(bin_edges):
    """
    Returns a safe width value/array for matplotlib bar plot on log scale.
    Works even if only 1 bin exists.
    """
    if len(bin_edges) < 2:
        return 0.8  # fallback
    diffs = np.diff(bin_edges)
    if len(diffs) == 0:
        return 0.8
    # Use 80% of the geometric distance between edges
    return diffs * 0.8


def get_ax(axs, row, col, num_rows, num_cols):
    """Safely access subplot axis regardless of 1D/2D structure"""
    if num_rows == 1 and num_cols == 1:
        return axs
    elif num_rows == 1:
        return axs[col]
    elif num_cols == 1:
        return axs[row]
    else:
        return axs[row, col]


if __name__ == "__main__":
    # Iterate through all files in Data/EngineSoundsGradualChanges
    for subdir, dirs, files in os.walk(ENGINE_SOUND_DIR):
        # Skip empty directories
        if Path(subdir).absolute() == ENGINE_SOUND_DIR:
            continue

        if len(subdir) == 0:
            print("Skipping empty directory at " + subdir)
            continue

        print("Processing directory " + subdir)
        engines.append(subdir.split(os.sep)[-1].split('_')[0])

        plot_dir_name = subdir.replace(str(DATA_DIR_NAME), 'Plots')
        os.makedirs(plot_dir_name, exist_ok=True)
        os.makedirs(os.path.join(plot_dir_name, 'waveform'), exist_ok=True)
        os.makedirs(os.path.join(plot_dir_name, 'spectrogram'), exist_ok=True)

        # Clear previous plots
        shutil.rmtree(os.path.join(plot_dir_name, 'waveform'))
        shutil.rmtree(os.path.join(plot_dir_name, 'spectrogram'))

        engine_sounds = []
        for sd, d, soundFiles in os.walk(subdir):
            ogg_files = [f for f in soundFiles if f.endswith('.ogg')]
            engine_sounds = sorted(ogg_files, key=natural_sort_key)

        if len(engine_sounds) == 0:
            print("Skipping empty directory at " + subdir)
            continue

        print("Processing directory " + subdir)

        # Create grid of plots
        num_cols, num_rows = create_plot_grid(len(engine_sounds))
        waveform_fig, waveform_axs = plt.subplots(num_rows, num_cols, figsize=(num_rows*6, num_cols*3))

        # Create grid of plots for spectrogram
        spec_fig, spec_axs = plt.subplots(num_rows, num_cols, figsize=(num_rows*6, num_cols*3))

        # Create grid of plots for mbe
        mbe_fig, mbe_axs = plt.subplots(num_rows, num_cols, figsize=(num_rows*6, num_cols*3))

        # Create grid of plots for mfcc
        mfcc_fig, mfcc_axs = plt.subplots(num_rows, num_cols, figsize=(num_rows*6, num_cols*3))

        # Plot each sound file in the grid
        col_cnt = 0
        row_cnt = 0
        mbe_data_for_engine = []  # To collect all recordings for this engine

        engine_name=None
        bin_centers = None
        for sound_file in engine_sounds:
            engine_name = sound_file.split('_')[0]

            # Load the sound file
            sound_file_path = os.path.join(subdir, sound_file)
            waveform, sample_rate = torchaudio.load(sound_file_path)
            recording_name = sound_file.split("_")[1].split(".")[0]  # e.g., "1000km", "5000km"

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.squeeze().numpy()  # Convert to numpy for plotting
            time_axis = np.arange(len(waveform)) / sample_rate  # Time axis in seconds

            # add the waveform to the grid
            waveform_axs[row_cnt, col_cnt].plot(time_axis, waveform, color='blue')
            waveform_axs[row_cnt, col_cnt].set_title(f'{sound_file.split("_")[1].split(".")[0]}')
            waveform_axs[row_cnt, col_cnt].set_xlabel('Time (seconds)')
            waveform_axs[row_cnt, col_cnt].set_ylabel('Amplitude')
            waveform_axs[row_cnt, col_cnt].set_xticks([])
            waveform_axs[row_cnt, col_cnt].set_yticks([])

            # Plot spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=64,  # Number of Mel bands
                n_fft=N_FFT,  # FFT window size
                hop_length=HOP_LENGTH  # Hop length for time resolution
            )

            mel_spectrogram = mel_transform(torch.from_numpy(waveform).float())
            mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)  # Convert to dB

            # Add the spectrogram to the grid
            spec_axs[row_cnt, col_cnt].imshow(mel_spectrogram_db, aspect='auto', origin='lower', extent=[0, time_axis[-1], 0, sample_rate/2])
            spec_axs[row_cnt, col_cnt].set_title(f'{sound_file.split("_")[1].split(".")[0]}')
            spec_axs[row_cnt, col_cnt].set_xlabel('Time (seconds)')
            spec_axs[row_cnt, col_cnt].set_ylabel('Frequency (Hz)')
            spec_axs[row_cnt, col_cnt].set_xticks([])

            mbe_energies, bin_centers, bin_edges = compute_mbe(sound_file_path)

            # Store data for CSV
            mbe_data_for_engine.append({
                'recording': sound_file,
                'label': recording_name,
                'energies': mbe_energies.tolist(),
                'bin_centers': bin_centers.tolist()
            })

            # bin_centers = 0.5 * (bins[:-1] + bins[1:])
            # width = (bins[1] - bins[0]) * 0.4  # bar width
            # mbe_axs[row_cnt, col_cnt].bar(bin_centers, mbe, width, color='blue')
            # mbe_axs[row_cnt, col_cnt].set_title(f'{sound_file.split("_")[1].split(".")[0]}')
            # mbe_axs[row_cnt, col_cnt].set_xlabel('Frequency (Hz)')
            # mbe_axs[row_cnt, col_cnt].set_ylabel('Power')
            # mbe_axs[row_cnt, col_cnt].set_xticks([])

            # Plotting
            ax = get_ax(mbe_axs, row_cnt, col_cnt, num_rows, num_cols)

            # === BAR PLOT WITH SAFE WIDTH ===
            width = safe_bar_width(bin_edges)

            ax.bar(bin_centers, mbe_energies,
                   width=width,
                   color='steelblue',
                   edgecolor='black',
                   alpha=0.85,
                   align='center',
                   log=False)  # log=False because x is already log-scaled

            ax.set_title(recording_name, fontsize=10, pad=10)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Energy')
            ax.set_xscale('log')

            # Nice log-scale ticks
            ax.set_xticks([50, 100, 200, 500, 1000, 2000, 4000, 8000])
            ax.set_xticklabels(['50', '100', '200', '500', '1k', '2k', '4k', '8k'])
            ax.grid(True, axis='y', alpha=0.3, linewidth=0.7)
            ax.grid(True, axis='x', alpha=0.2, which='minor')

            # Load the sound file for MFCC
            y, sr = librosa.load(sound_file_path, sr=SR, mono=True)
            mfcc_before = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)

            # Add the MFCC to the grid
            mfcc_axs[row_cnt, col_cnt].imshow(mfcc_before, aspect='auto', origin='lower', extent=[0, time_axis[-1], 0, N_MFCC])
            mfcc_axs[row_cnt, col_cnt].set_title(f'{sound_file.split("_")[1].split(".")[0]}')
            mfcc_axs[row_cnt, col_cnt].set_xlabel('Time (seconds)')
            mfcc_axs[row_cnt, col_cnt].set_ylabel('MFCC Coeffs')
            mfcc_axs[row_cnt, col_cnt].set_xticks([])

            col_cnt += 1
            if col_cnt >= num_cols:
                col_cnt = 0
                row_cnt += 1

        # === Hide unused subplots ===
        while row_cnt < num_rows:
            while col_cnt < num_cols:
                ax = get_ax(mbe_axs, row_cnt, col_cnt, num_rows, num_cols)
                ax.axis('off')
                col_cnt += 1
            col_cnt = 0
            row_cnt += 1

        # Save the plots
        waveform_fig.tight_layout()
        waveform_fig.savefig(f'{plot_dir_name}/{engine_name}_waveform.png')
        plt.close(waveform_fig)

        spec_fig.tight_layout()
        spec_fig.savefig(f'{plot_dir_name}/{engine_name}_spectrogram.png')
        plt.close(spec_fig)

        mbe_fig.tight_layout(h_pad=2)
        mbe_fig.savefig(f'{plot_dir_name}/{engine_name}_mbe.png', dpi=150, bbox_inches='tight')
        plt.close(mbe_fig)

        # === Save MBE numerical data to CSV ===
        engine_mbe_csv = f"{plot_dir_name}/{engine_name}_mbe_data.csv"
        with open(engine_mbe_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Recording', 'Label'] + [f'Band_{i + 1}_Hz({bin_centers[i]:.1f}Hz)' for i in range(N_BINS)]
            writer.writerow(header)

            for item in mbe_data_for_engine:
                row = [item['recording'], item['label']] + item['energies']
                writer.writerow(row)

        print(f"MBE numerical data saved: {engine_mbe_csv}")

        # Optional: Print summary
        print(f"Engine: {engine_name} | Recordings: {len(mbe_data_for_engine)} | Bands: {N_BINS}")
        print("-" * 60)

        mfcc_fig.tight_layout()
        mfcc_fig.savefig(f'{plot_dir_name}/{engine_name}_mfcc.png')
        plt.close(mfcc_fig)

