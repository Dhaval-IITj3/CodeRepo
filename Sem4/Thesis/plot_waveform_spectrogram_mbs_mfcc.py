import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil
import librosa
import re

from matplotlib import gridspec

"""
✅ Rule of thumb:
Use Librosa when your main goal is signal analysis, feature extraction, and visualization.
Use Torchaudio when you’re building a deep learning model in PyTorch and want an efficient, end-to-end GPU pipeline.
"""

# Frequency bin configuration
SR = 16000              # sampling rate
N_FFT = 1024            # FFT window size
N_BINS = 16              # number of frequency bins (can tune)
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
    bin_edges = np.linspace(0, freqs[-1], n_bins+1)

    energies = []
    for i in range(n_bins):
        fmin, fmax = bin_edges[i], bin_edges[i+1]
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if len(idx) > 0:
            band_energy = S[idx, :].mean()   # average power across frames
        else:
            band_energy = 0
        energies.append(band_energy)

    return np.array(energies), bin_edges


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

        for sound_file in engine_sounds:
            engine_name = sound_file.split('_')[0]

            # Load the sound file
            sound_file_path = os.path.join(subdir, sound_file)
            waveform, sample_rate = torchaudio.load(sound_file_path)

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

            mbe, bins = compute_mbe(sound_file_path)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            width = (bins[1] - bins[0]) * 0.4  # bar width
            mbe_axs[row_cnt, col_cnt].bar(bin_centers, mbe, width, color='blue')
            mbe_axs[row_cnt, col_cnt].set_title(f'{sound_file.split("_")[1].split(".")[0]}')
            mbe_axs[row_cnt, col_cnt].set_xlabel('Frequency (Hz)')
            mbe_axs[row_cnt, col_cnt].set_ylabel('Power')
            mbe_axs[row_cnt, col_cnt].set_xticks([])

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

        # Save the plots
        waveform_fig.tight_layout()
        waveform_fig.savefig(f'{plot_dir_name}/{engine_name}_waveform.png')
        plt.close(waveform_fig)

        spec_fig.tight_layout()
        spec_fig.savefig(f'{plot_dir_name}/{engine_name}_spectrogram.png')
        plt.close(spec_fig)

        mbe_fig.tight_layout()
        mbe_fig.savefig(f'{plot_dir_name}/{engine_name}_mbe.png')
        plt.close(mbe_fig)

        mfcc_fig.tight_layout()
        mfcc_fig.savefig(f'{plot_dir_name}/{engine_name}_mfcc.png')
        plt.close(mfcc_fig)

