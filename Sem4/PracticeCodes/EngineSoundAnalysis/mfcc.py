# plot_mfcc_engine.py
# pip install librosa matplotlib soundfile

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

"""
✅ Rule of thumb:
Use Librosa when your main goal is signal analysis, feature extraction, and visualization.
Use Torchaudio when you’re building a deep learning model in PyTorch and want an efficient, end-to-end GPU pipeline.
"""


# ---------- CONFIG ----------
SR = 16000          # sampling rate for all audio
N_MFCC = 20         # number of MFCCs to extract

DATA_DIR_NAME = 'Resources'
BASE_DIR = Path(__file__).parent
BASE_DIR = Path(BASE_DIR, "../").resolve()
DATA_DIR = Path.joinpath(BASE_DIR, DATA_DIR_NAME)

BEFORE_DIR = Path(DATA_DIR, 'Before')
AFTER_DIR = Path(DATA_DIR, 'After')
# ----------------------------


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

    # Clear any existing plots
    if os.path.exists(os.path.join('Plots', 'mfcc')) and os.listdir(os.path.join('Plots', 'mfcc')):
        shutil.rmtree(os.path.join('Plots', 'mfcc'), ignore_errors=True)

    os.makedirs('Plots/mfcc', exist_ok=True)

    # Plot MFCC comparison per engine
    for engine in engines:
        before_file_name = os.path.join(DATA_DIR, 'Before', f'{engine}_Before.ogg')
        after_file_name = os.path.join(DATA_DIR, 'After', f'{engine}_After.ogg')

        if not os.path.exists(before_file_name) or not os.path.exists(after_file_name):
            print(f"Skipping {engine} because one of the files does not exist.")
            continue

        # Load audio
        y_before, sr_before = librosa.load(before_file_name, sr=SR, mono=True)
        y_after, sr_after = librosa.load(after_file_name, sr=SR, mono=True)

        # Extract MFCC
        mfcc_before = librosa.feature.mfcc(y=y_before, sr=SR, n_mfcc=N_MFCC)
        mfcc_after = librosa.feature.mfcc(y=y_after, sr=SR, n_mfcc=N_MFCC)

        # Plot
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        librosa.display.specshow(mfcc_before, x_axis="time", sr=SR, ax=axs[0])
        axs[0].set(title=f"{engine} - BEFORE Oil Change", ylabel="MFCC Coeffs")
        axs[0].label_outer()

        librosa.display.specshow(mfcc_after, x_axis="time", sr=SR, ax=axs[1])
        axs[1].set(title=f"{engine} - AFTER Oil Change", ylabel="MFCC Coeffs", xlabel="Time")

        plt.suptitle(f"MFCC Comparison for {engine}", fontsize=14)
        plt.colorbar(axs[0].collections[0], ax=axs, location="right")
        # plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot
        plt.savefig(os.path.join('Plots', 'mfcc', f'{engine}_comparison_mfcc_plot.png'))
        plt.close()
