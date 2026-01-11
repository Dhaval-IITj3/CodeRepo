import os
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import numpy as np

# --------------------------- Paths ---------------------------
DATA_DIR_NAME = 'Resources\\EngineSoundsNewBikes'
BASE_DIR = Path(__file__).parent
BASE_DIR = BASE_DIR.parent.resolve()  # Go one level up
PLOT_DIR = BASE_DIR / 'EngineSoundAnalysis' / 'Plots' / 'NewBikeFeatureExtraction'
DATA_DIR = BASE_DIR / DATA_DIR_NAME

PLOT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    print(f"Error: Data directory {DATA_DIR} does not exist.")
    exit()

def compute_mel_spectogram(filepath):
    scale,sr = librosa.load(filepath)
    mel_spectogram=librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=2048, n_mels=20, hop_length=128)
    mel_spectogram_db = librosa.power_to_db(mel_spectogram, ref=np.max)
    return mel_spectogram_db

def get_spectogram():
    # store mel for each bike in a list
    mel_array = []
    labels = []
    bike_names = []

    # Try folder-per-bike first
    subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if subdirs:
        bike_names = sorted([d.name for d in subdirs])
        for idx, folder in enumerate(subdirs):
            for file in folder.glob('*.ogg'):
                mel = compute_mel_spectogram(str(file))

                # Save the spectrogram if it's not None
                if mel is not None:
                    mel_array.append(mel)
                    labels.append(idx)
    else:
        # Fallback: all .ogg files in root are individual bikes
        ogg_files = sorted(DATA_DIR.glob('*.ogg'))
        bike_names = [f.stem for f in ogg_files]
        for idx, file in enumerate(ogg_files):
            mel = compute_mel_spectogram(str(file))
            if mel is not None:
                mel_array.append(mel)
                labels.append(idx)

    return mel_array, np.array(labels), bike_names

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


def create_plot_grid(total_plots):
    num_cols = 1
    while num_cols ** 2 < total_plots:
        num_cols += 1

    num_rows = (total_plots + num_cols - 1) // num_cols

    return num_cols, num_rows

def plot_bike_mel_grid(mel_array, labels, bike_names, sr=22050):
    """
    Plots a grid of mel-spectrograms, one per bike.
    """
    unique_labels = np.unique(labels)
    num_bikes = len(unique_labels)

    cols, rows = create_plot_grid(num_bikes)

    fig, axes = plt.subplots(rows, cols, figsize=(25, 10 * rows))
    axes = axes.flatten()

    for i, label in enumerate(unique_labels):
        ax = axes[i]

        # pick first sample for this bike
        idx = np.where(labels == label)[0][0]
        mel = mel_array[idx]

        img = librosa.display.specshow(
            mel,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            ax=ax
        )
        ax.set_title(bike_names[label])
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(PLOT_DIR, 'mel_spectogram_new_bikes.png'))


if __name__ == "__main__":
    mel_array, labels, bike_names = get_spectogram()
    plot_bike_mel_grid(mel_array, labels, bike_names)




