import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Paths
DATA_DIR_NAME = 'Resources'
BASE_DIR = Path(__file__).parent
BASE_DIR = Path(BASE_DIR, "../").resolve()
DATA_DIR = Path.joinpath(BASE_DIR, DATA_DIR_NAME)

BEFORE_DIR = Path(DATA_DIR, 'Before')
AFTER_DIR = Path(DATA_DIR, 'After')


# Frequency bin configuration
SR = 16000              # sampling rate
N_FFT = 1024            # FFT window size
N_BINS = 16              # number of frequency bins (can tune)
HOP_LENGTH = 512


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


if __name__ == "__main__":
    before_list = os.listdir(BEFORE_DIR)
    after_list = os.listdir(AFTER_DIR)

    # Extract engine names from file names
    engines = []

    for f in before_list:
        if f.endswith('.ogg'):
            engines.append(f.split('_')[0])

    for f in after_list:
        if f.endswith('.ogg'):
            engines.append(f.split('_')[0])

    engines = list(set(engines))

    os.makedirs('Plots', exist_ok=True)

    if os.path.isdir('Plots/mbe') and os.listdir('Plots/mbe'):
        shutil.rmtree(os.path.join('Plots', 'mbe'), ignore_errors=True)

    os.makedirs('Plots/mbe', exist_ok=True)

    # Loop over engines
    for engine in engines:
        before_file_name = os.path.join(DATA_DIR, 'Before', f'{engine}_Before.ogg')
        after_file_name = os.path.join(DATA_DIR, 'After', f'{engine}_After.ogg')

        if not os.path.exists(before_file_name) or not os.path.exists(after_file_name):
            print(f"Skipping {engine} because one of the files does not exist.")
            continue

        # Compute MBE
        mbe_before, bins = compute_mbe(before_file_name)
        mbe_after,  _    = compute_mbe(after_file_name)

        # Plot histogram comparison
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        width = (bins[1]-bins[0]) * 0.4  # bar width

        plt.figure(figsize=(10,5))
        plt.bar(bin_centers - width/2, mbe_before, width=width, label="Before Oil Change", alpha=0.7)
        plt.bar(bin_centers + width/2, mbe_after, width=width, label="After Oil Change", alpha=0.7)

        plt.title(f"Multiband Energy Comparison for {engine}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Average Energy")
        plt.legend()
        # plt.tight_layout()
        plt.savefig(os.path.join('Plots', 'mbe', f'{engine}_comparison_mbe_plot.png'))
        plt.close()
