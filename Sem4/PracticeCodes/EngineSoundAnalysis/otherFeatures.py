import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from scipy.signal import stft, windows
import pygame
import pygame.mixer
import pygame.sndarray
import math

# Paths
DATA_DIR_NAME = 'Resources'
BASE_DIR = Path(__file__).parent
BASE_DIR = Path(BASE_DIR, "../").resolve()
DATA_DIR = Path.joinpath(BASE_DIR, DATA_DIR_NAME)

PLOT_DIR = BASE_DIR / 'EngineSoundAnalysis' / 'Plots'
PLOT_FEATURE_DIR = PLOT_DIR / 'featurePlots'

BEFORE_DIR = Path(DATA_DIR, 'Before')
AFTER_DIR = Path(DATA_DIR, 'After')


# Frequency bin configuration
SR = 22050              # sampling rate
N_FFT = 1024            # FFT window size
N_BINS = 16              # number of frequency bins (can tune)
HOP_LENGTH = 512

# ------------------- Feature Names (Fixed 12-element vector) -------------------
FEATURE_NAMES = [
    'Centroid', 'Crest', 'Decrease', 'Entropy', 'Flatness', 'Flux',
    'Kurtosis', 'Rolloff', 'Skewness', 'Slope', 'Spread', 'Pitch'
]

# Make sure this matches the length of final_vector
assert len(FEATURE_NAMES) == 12, "Feature name count must match vector length"

# ------------------- Spectral Feature Functions -------------------
def spectral_centroid(S, freqs):
    total = np.sum(S) + 1e-8
    return np.sum(freqs * S) / total

def spectral_crest(S):
    return np.max(S) / (np.mean(S) + 1e-8)

def spectral_decrease(S):
    if len(S) < 2:
        return 0.0
    weights = np.arange(1, len(S)) + 1
    return np.sum((S[1:] - S[0]) / weights) / (np.sum(S[1:]) + 1e-8)

def spectral_entropy(S):
    ps = S / (np.sum(S) + 1e-8)
    ps = ps[ps > 0]
    return -np.sum(ps * np.log2(ps))

def spectral_flatness(S):
    S = S[S > 0]
    if len(S) == 0:
        return 0.0
    geo_mean = np.exp(np.mean(np.log(S)))
    arith_mean = np.mean(S)
    return geo_mean / arith_mean

def spectral_flux(S_prev, S_curr):
    return np.sqrt(np.sum((S_curr - S_prev) ** 2)) + 1e-8  # L2 norm (more common)

def spectral_spread(S, freqs, centroid):
    total = np.sum(S) + 1e-8
    return np.sqrt(np.sum(((freqs - centroid) ** 2) * S) / total)

def spectral_skewness(S, freqs, centroid, spread):
    if spread == 0:
        return 0.0
    total = np.sum(S) + 1e-8
    third_moment = np.sum(((freqs - centroid) ** 3) * S) / total
    return third_moment / (spread ** 3)

def spectral_kurtosis(S, freqs, centroid, spread):
    if spread == 0:
        return 0.0
    total = np.sum(S) + 1e-8
    fourth_moment = np.sum(((freqs - centroid) ** 4) * S) / total
    return fourth_moment / (spread ** 4)

def spectral_rolloff(S, freqs, percentile=0.85):
    total_energy = np.sum(S)
    if total_energy == 0:
        return 0.0
    cum_energy = np.cumsum(S)
    idx = np.argmax(cum_energy >= percentile * total_energy)
    return freqs[idx]

def spectral_slope(S, freqs):
    if len(freqs) < 2:
        return 0.0
    return np.polyfit(freqs, S, 1)[0]

# Improved pitch estimation using autocorrelation with bounds
def estimate_pitch(y_frame, sr):
    if len(y_frame) < 200:
        return 0.0
    # Focus on reasonable pitch range: ~80 Hz to 500 Hz → period 0.002 to 0.0125 s
    min_lag = int(sr / 500)
    max_lag = int(sr / 80)
    autocorr = np.correlate(y_frame, y_frame, mode='full')[len(y_frame):]
    autocorr = autocorr[min_lag:max_lag+1]
    if len(autocorr) == 0:
        return 0.0
    peak_idx = np.argmax(autocorr) + min_lag
    return sr / peak_idx if peak_idx > 0 else 0.0


# ------------------- Load Audio Functions -------------------
def load_audio(filename, sr=22050):
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    try:
        sound = pygame.mixer.Sound(filename)
        array = pygame.sndarray.array(sound)
        if array.ndim == 2:
            array = np.mean(array, axis=1)
        sr = pygame.mixer.get_init()[0]
        return array.astype(np.float32) / 32768.0, sr
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None


def extract_features(filename):
    y, sr = load_audio(filename)
    if y is None:
        return None, None, None, None

    window_size = int(0.015 * sr)  # 15 ms window
    hop_size = int(0.005 * sr)     # 5 ms hop
    if window_size < 64:
        window_size = 512
        hop_size = 128

    # Using power spectrogram (magnitude squared)
    S = librosa.stft(y, n_fft=window_size, hop_length=hop_size)

    # For original spectral features we use magnitude
    mag = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=window_size)

    frame_feats = []
    pitches = []
    fluxes = []
    prev_mag = None

    for i in range(mag.shape[1]):
        frame = mag[:, i] + 1e-8  # Avoid zero division
        centroid = spectral_centroid(frame, freqs)

        # Now compute spread (needed for skewness and kurtosis)
        spread = spectral_spread(frame, freqs, centroid)

        # Then skewness and kurtosis using all required params
        skewness = spectral_skewness(frame, freqs, centroid, spread)
        kurtosis = spectral_kurtosis(frame, freqs, centroid, spread)

        frame_feats.append([
            centroid,
            spectral_crest(frame),
            spectral_decrease(frame),
            spectral_entropy(frame),
            spectral_flatness(frame),
            # flux inserted later as mean
            kurtosis,  # index 5 in base
            spectral_rolloff(frame, freqs),
            skewness,  # index 7
            spectral_slope(frame, freqs),
            spread
        ])

        # Pitch
        start = i * hop_size
        end = start + window_size
        y_frame = y[start:end]
        if len(y_frame) < window_size:
            y_frame = np.pad(y_frame, (0, window_size - len(y_frame)))
        y_frame *= windows.hamming(window_size)
        pitches.append(estimate_pitch(y_frame, sr))

        # Flux
        if prev_mag is not None:
            fluxes.append(spectral_flux(prev_mag, frame))
        prev_mag = frame.copy()

    # Compute means
    mean_base = np.mean(np.array(frame_feats), axis=0)
    mean_flux = np.mean(fluxes) if fluxes else 0.0
    mean_pitch = np.mean(pitches)

    # Build final 12-element vector: insert flux after flatness (index 5)
    final_vector = np.insert(mean_base, 5, mean_flux)
    final_vector = np.append(final_vector, mean_pitch)

    # === MFCC & Delta MFCC (time-series kept for plotting) ===
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048,
                                hop_length=512, window='hamming', n_mels=64, fmax=8000)
    delta_mfcc = librosa.feature.delta(mfcc)

    return final_vector, mfcc, delta_mfcc, S


# ────────────────────────────────────────────────
#  Plotting function - one feature at a time
# ────────────────────────────────────────────────
def plot_feature_comparison(before_val, after_val, feature_name, engine, save_dir):
    plt.figure(figsize=(5, 3.2))

    x = ["Before", "After"]
    y = [before_val, after_val]

    bars = plt.bar(x, y, color=['#1f77b4', '#ff7f0e'], alpha=0.85, width=0.6)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(y),
                 f'{height:.4g}',
                 ha='center', va='bottom', fontsize=10)

    plt.title(f"{feature_name}\n{engine}", fontsize=13, pad=10)
    plt.ylabel("Value")
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Optional: connect points with line
    plt.plot(x, y, color='gray', alpha=0.5, marker='o', zorder=1)

    plt.tight_layout()
    safe_name = feature_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(save_dir / f"{safe_name}.png", dpi=120)
    plt.close()


def create_plot_grid(total_plots):
    num_cols = 1
    while num_cols ** 2 < total_plots:
        num_cols += 1

    num_rows = (total_plots + num_cols - 1) // num_cols

    return num_cols, num_rows

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

    os.makedirs(PLOT_DIR, exist_ok=True)

    if os.path.isdir(PLOT_FEATURE_DIR) and os.listdir(PLOT_FEATURE_DIR):
        shutil.rmtree(PLOT_FEATURE_DIR, ignore_errors=True)

    os.makedirs(PLOT_FEATURE_DIR, exist_ok=True)

    # We'll store results here: engine → (before_vector, after_vector)
    results = {}

    # Loop over engines
    for engine in engines:
        before_file_name = os.path.join(DATA_DIR, 'Before', f'{engine}_Before.ogg')
        after_file_name = os.path.join(DATA_DIR, 'After', f'{engine}_After.ogg')

        if not os.path.exists(before_file_name) or not os.path.exists(after_file_name):
            print(f"Skipping {engine} because one of the files does not exist.")
            continue

        print(f"Processing {engine}")

        # Extract features
        before_vector, _, _, _ = extract_features(str(before_file_name))
        after_vector, _, _, _ = extract_features(str(after_file_name))

        if before_vector is None or after_vector is None:
            print(f"  → failed to extract features for {engine}")
            continue

        results[engine] = (before_vector, after_vector)
        print("ok")

    if not results:
        print("No valid engine pairs found. Exiting.")
        exit()

    engines = sorted(results.keys())  # final ordered list
    n_engines = len(engines)

    print(f"\nFound {n_engines} engines with valid before/after pairs.")

    # ────────────────────────────────────────────────
    #  One figure per feature → grid of subplots
    # ────────────────────────────────────────────────
    plot_feature_path = Path(PLOT_FEATURE_DIR)
    if plot_feature_path.exists():
        shutil.rmtree(plot_feature_path)
    plot_feature_path.mkdir(parents=True, exist_ok=True)

    cols, rows = create_plot_grid(len(engines))

    for feat_idx, feature_name in enumerate(FEATURE_NAMES):
        fig, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=(cols * 4.2, rows * 3.0),
            sharey='row',  # optional: same y-scale per row (can be confusing)
            squeeze=False
        )
        fig.suptitle(f"{feature_name} — Before vs After Oil Change", fontsize=16, y=0.98)

        for i, engine in enumerate(engines):
            row = i // cols
            col = i % cols
            ax = axes[row, col]

            before_val = results[engine][0][feat_idx]
            after_val = results[engine][1][feat_idx]

            x = ["Before", "After"]
            y = [before_val, after_val]

            ax.bar(x, y, color=['#1f77b4', '#ff7f0e'], alpha=0.85, width=0.62)

            # Value labels
            for j, val in enumerate(y):
                ax.text(j, val + 0.015 * max(1e-6, abs(max(y) - min(y))),
                        f"{val:.4g}", ha='center', va='bottom', fontsize=9)

            # Optional connecting line
            ax.plot(x, y, color='gray', lw=1.2, marker='o', markersize=6, alpha=0.6)

            ax.set_title(engine, fontsize=11, pad=6)
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(axis='y', alpha=0.3, ls='--')

            # Hide x-labels except bottom row
            if row < rows - 1:
                ax.set_xticklabels([])

        # Hide empty subplots
        for j in range(i + 1, rows * cols):
            row = j // cols
            col = j % cols
            axes[row, col].set_visible(False)

        # Common y-label
        fig.text(0.07, 0.5, "Feature Value", va='center', rotation='vertical', fontsize=12)

        plt.tight_layout(rect=[0.08, 0.03, 0.98, 0.94])  # leave space for suptitle
        safe_name = feature_name.lower().replace(" ", "_").replace("_(mean)", "").replace("(", "").replace(")", "")
        save_path = PLOT_FEATURE_DIR / f"{safe_name}_grid.png"
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved: {save_path.name}")

    print(f"\nAll feature grid plots saved in:\n  {PLOT_FEATURE_DIR.resolve()}")
