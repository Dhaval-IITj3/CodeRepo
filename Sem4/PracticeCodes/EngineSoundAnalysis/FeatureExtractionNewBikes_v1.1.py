import os
import numpy as np
from scipy.signal import stft, windows
import pygame
import pygame.mixer
import pygame.sndarray
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import librosa
import math


# --------------------------- Paths ---------------------------
DATA_DIR_NAME = 'Resources\\EngineSoundsNewBikes'
BASE_DIR = Path(__file__).parent
BASE_DIR = BASE_DIR.parent.resolve()  # Go one level up
PLOT_DIR = BASE_DIR / 'EngineSoundAnalysis' / 'Plots' / 'NewBikeFeatureExtraction'
RESOURCE_DIR = BASE_DIR / DATA_DIR_NAME

PLOT_DIR.mkdir(parents=True, exist_ok=True)

if not RESOURCE_DIR.exists():
    print(f"Error: Data directory {RESOURCE_DIR} does not exist.")
    exit()

# ------------------- Feature Names (Fixed 12-element vector) -------------------
FEATURE_NAMES = [
    'Centroid', 'Crest', 'Decrease', 'Entropy', 'Flatness', 'Flux',
    'Kurtosis', 'Rolloff', 'Skewness', 'Slope', 'Spread', 'Pitch'
]

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

# ------------------- Audio Loading -------------------
def load_audio(filename):
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

# ------------------- Feature Extraction (Any Length Audio) -------------------
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

# ------------------- Load Dataset (Scalable to Any Number of Bikes) -------------------
def load_dataset():
    features_list = []
    mfccs_list = []
    delta_mfccs_list = []
    spectrograms_list = []
    labels = []
    bikenames = []

    subdirs = [d for d in RESOURCE_DIR.iterdir() if d.is_dir()]
    if subdirs:
        for idx, folder in enumerate(subdirs):
            for file in folder.glob('*.ogg'):
                bikename = os.path.basename(file).split('.')[0]  # better than split('.')[0]
                feat, mfcc, delta, spec = extract_features(str(file))
                if feat is not None:
                    features_list.append(feat)
                    mfccs_list.append(mfcc)
                    delta_mfccs_list.append(delta)
                    spectrograms_list.append(spec)
                    bikenames.append(bikename)
                    labels.append(idx)
    else:
        ogg_files = sorted(RESOURCE_DIR.glob('*.ogg'))
        for idx, file in enumerate(ogg_files):
            bikename = file.stem
            feat, mfcc, delta, spec = extract_features(str(file))
            if feat is not None:
                features_list.append(feat)
                mfccs_list.append(mfcc)
                delta_mfccs_list.append(delta)
                spectrograms_list.append(spec)
                bikenames.append(bikename)
                labels.append(idx)

    return (np.array(features_list),
            np.array(labels),
            mfccs_list,  # keep as list (variable shape)
            delta_mfccs_list,
            bikenames,
            spectrograms_list)


# ------------------- PyTorch Classifier (Dynamic Output Size) -------------------
class BikeClassifier(nn.Module):
    def __init__(self, input_size=12, num_classes=12):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ------------------- Plotting: Bar Charts for Any Number of Bikes -------------------
def plot_feature_comparisons(X, y, bike_names, feature_names):
    unique_labels = np.unique(y)
    mean_features = [np.mean(X[y == lbl], axis=0) for lbl in unique_labels]
    mean_features = np.array(mean_features)

    bike_labels = [name.replace('_', ' ') for name in bike_names]

    for i, name in enumerate(feature_names):
        plt.figure(figsize=(max(10, len(bike_names) * 0.8), 6))
        values = mean_features[:, i]
        bars = plt.bar(range(len(bike_labels)), values, color='teal', alpha=0.8, edgecolor='black')
        plt.scatter(range(len(bike_labels)), values, color='orange', s=80, zorder=5)

        plt.title(f'{name} Across Different Bike Engines', fontsize=16)
        plt.xlabel('Bike Engine')
        plt.ylabel(name)
        plt.xticks(range(len(bike_labels)), bike_labels, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + np.max(values) * 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{name.lower()}_comparison.png", dpi=150)
        plt.close()


# ------------------- Plotting: Heatmap for Any Number of Bikes -------------------
def plot_mfcc_comparison(mfccs_list, bike_names, feature_type="MFCC"):
    n_bikes = len(bike_names)
    fig, axes = plt.subplots(n_bikes, 1, figsize=(14, 3*n_bikes), sharex=True)
    if n_bikes == 1:
        axes = [axes]

    for idx, (mfcc, ax) in enumerate(zip(mfccs_list, axes)):
        librosa.display.specshow(mfcc, x_axis='time', ax=ax, cmap='viridis')
        ax.set_title(f'{feature_type} - {bike_names[idx]}')
        ax.set_ylabel('MFCC Coefficients')

    fig.suptitle(f'{feature_type} Heatmap Comparison Across Engines', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_DIR / f"{feature_type.lower()}_comparison_all.png", dpi=150)
    plt.close()


# ------------------- Feature Extraction: Band Energy Ratio -------------------
def calculate_split_frequency_bin(split_frequency, sample_rate, num_frequency_bins):
    """
    Calculate the FFT bin index corresponding to a given split frequency.
    """
    frequency_range = sample_rate / 2.0
    frequency_delta_per_bin = frequency_range / (num_frequency_bins - 1)  # more precise
    split_bin = math.floor(split_frequency / frequency_delta_per_bin)
    # Clamp to valid range
    split_bin = max(1, min(split_bin, num_frequency_bins - 1))
    return int(split_bin)


def band_energy_ratio(spectrogram, split_frequency, sample_rate):
    """
    Calculate Band Energy Ratio (low / high) for each time frame.

    Parameters:
    - spectrogram_power: numpy array of shape (freq_bins, time_frames) - already power (|S|**2)
    - split_frequency: frequency in Hz to split low vs high band
    - sample_rate: audio sampling rate

    Returns:
    - ber: numpy array of shape (time_frames,) with BER values per frame
    """
    split_frequency_bin = calculate_split_frequency_bin(split_frequency, sample_rate, len(spectrogram[0]))
    band_energy_ratio = []

    # calculate power spectrogram
    power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T

    # calculate BER value for each frame
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(band_energy_ratio_current_frame)

    return np.array(band_energy_ratio)


def plot_ber_time_series_grid(spectrograms_list, bikename_arry, y, sr=44100, hop_length=512):
    """
    Plot Band Energy Ratio time series in a grid of individual subplots,
    one subplot per bike engine.
    """
    # Split frequency to use (you can change or make it a list)
    split_frequency = 500  # Hz
    bike_ber_data = []  # list of (bike_name, time, ber)

    for i, bike in enumerate(bikename_arry):
        ber = band_energy_ratio(spectrograms_list[i], split_frequency, sr)
        frames = range(len(ber))
        t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        bike_ber_data.append((bike, t, ber))

    if not bike_ber_data:
        print("No data available for BER plotting.")
        return

    n_bikes = len(bike_ber_data)

    # Determine grid size (e.g. 3×4, 4×3, etc.)
    cols = min(4, n_bikes)  # max 4 columns
    rows = math.ceil(n_bikes / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows),
                             sharex=True, sharey=False)

    # Flatten axes for easy iteration
    axes_flat = axes.ravel() if n_bikes > 1 else [axes]

    for i, (bike_name, t, ber) in enumerate(bike_ber_data):
        ax = axes_flat[i]

        ax.plot(t, ber, linewidth=1.5)
        ax.set_title(bike_name, fontsize=11, pad=8)
        ax.set_xlabel('Time (s)' if i >= (rows - 1) * cols else '')
        ax.set_ylabel('BER (Low/High)')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Optional: log scale if values span many orders of magnitude
        # ax.set_yscale('log')

    # Hide unused subplots
    for j in range(n_bikes, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'Band Energy Ratio (Low < {split_frequency} Hz / High) Time Evolution\nPer Bike Engine',
                 fontsize=14, y=0.98)

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = PLOT_DIR / f"ber_time_series_grid_{split_frequency}Hz.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved BER time-series comparison plot: {save_path}")

# ------------------- Main Execution -------------------
if __name__ == '__main__':
    print("Loading and processing bike engine sounds...")
    X, y, mfcc_ary, delta_mfcc_ary, bike_names, specgrm = load_dataset()

    n_bikes = len(bike_names)
    n_samples = len(X)

    print(f"Successfully processed {n_samples} recordings from {n_bikes} unique bike engines.")
    print("Bike engines detected:", bike_names)

    if n_bikes < 2:
        print("Warning: Need at least 2 bikes to train a classifier.")
    else:
        # Save feature matrix for future use
        np.save(PLOT_DIR / 'feature_vectors.npy', X)
        np.save(PLOT_DIR / 'labels.npy', y)
        np.save(PLOT_DIR / 'bike_names.npy', bike_names)

        # Generate comparison plots
        plot_feature_comparisons(X, y, bike_names, FEATURE_NAMES)

        print("Generating MFCC comparison plots...")
        plot_mfcc_comparison(mfcc_ary, bike_names, "MFCC")

        print("Generating Delta-MFCC comparison plots...")
        plot_mfcc_comparison(delta_mfcc_ary, bike_names, "Delta-MFCC")

        print("Generating Band Energy comparison plots...")
        plot_ber_time_series_grid(specgrm, bike_names, y)

        # Train classifier
        print("\nTraining classifier...")
        model = BikeClassifier(num_classes=n_bikes)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(300):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/300 - Loss: {loss.item():.6f}")

        # Save model
        torch.save(model.state_dict(), PLOT_DIR / 'bike_engine_classifier.pth')
        print(f"\nModel trained and saved for {n_bikes} bike engines.")
        print("You can add more bikes later — just drop new folders/files and re-run!")