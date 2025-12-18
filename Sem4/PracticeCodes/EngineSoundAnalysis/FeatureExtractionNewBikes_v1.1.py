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
        return None

    window_size = int(0.015 * sr)  # 15 ms window
    hop_size = int(0.005 * sr)     # 5 ms hop
    if window_size < 64:
        window_size = 512
        hop_size = 128

    f, t, Zxx = stft(y, fs=sr, window='hamming', nperseg=window_size, noverlap=window_size - hop_size)
    mag = np.abs(Zxx)
    freqs = f

    frame_feats = []
    pitches = []
    fluxes = []
    prev_mag = None

    for i in range(mag.shape[1]):
        S = mag[:, i] + 1e-8  # Avoid zero division
        centroid = spectral_centroid(S, freqs)

        # Now compute spread (needed for skewness and kurtosis)
        spread = spectral_spread(S, freqs, centroid)

        # Then skewness and kurtosis using all required params
        skewness = spectral_skewness(S, freqs, centroid, spread)
        kurtosis = spectral_kurtosis(S, freqs, centroid, spread)

        frame_feats.append([
            centroid,
            spectral_crest(S),
            spectral_decrease(S),
            spectral_entropy(S),
            spectral_flatness(S),
            # flux inserted later as mean
            kurtosis,  # index 5 in base
            spectral_rolloff(S, freqs),
            skewness,  # index 7
            spectral_slope(S, freqs),
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
            fluxes.append(spectral_flux(prev_mag, S))
        prev_mag = S.copy()

    # Compute means
    mean_base = np.mean(frame_feats, axis=0)
    mean_flux = np.mean(fluxes) if fluxes else 0.0
    mean_pitch = np.mean(pitches)

    # Build final 12-element vector: insert flux after flatness (index 5)
    final_vector = np.insert(mean_base, 5, mean_flux)
    final_vector = np.append(final_vector, mean_pitch)

    return final_vector

# ------------------- Load Dataset (Scalable to Any Number of Bikes) -------------------
def load_dataset():
    features = []
    labels = []
    bike_names = []

    # Try folder-per-bike first
    subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if subdirs:
        bike_names = sorted([d.name for d in subdirs])
        for idx, folder in enumerate(subdirs):
            for file in folder.glob('*.ogg'):
                feat = extract_features(str(file))
                if feat is not None:
                    features.append(feat)
                    labels.append(idx)
    else:
        # Fallback: all .ogg files in root are individual bikes
        ogg_files = sorted(DATA_DIR.glob('*.ogg'))
        bike_names = [f.stem for f in ogg_files]
        for idx, file in enumerate(ogg_files):
            feat = extract_features(str(file))
            if feat is not None:
                features.append(feat)
                labels.append(idx)

    return np.array(features), np.array(labels), bike_names

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
def plot_feature_comparisons(X, bike_names):
    # Average per bike if multiple recordings
    unique_labels, counts = np.unique(y, return_counts=True)
    mean_features = []
    for label in unique_labels:
        bike_data = X[y == label]
        mean_features.append(np.mean(bike_data, axis=0))
    mean_features = np.array(mean_features)  # Shape: (n_bikes, 12)

    bike_labels = [name.replace('_', ' ') for name in bike_names]

    for i, name in enumerate(FEATURE_NAMES):
        plt.figure(figsize=(max(10, len(bike_names) * 0.8), 6))
        values = mean_features[:, i]

        bars = plt.bar(range(len(bike_labels)), values, color='teal', alpha=0.8, edgecolor='black')
        plt.scatter(range(len(bike_labels)), values, color='orange', s=80, zorder=5)

        plt.title(f'{name} Across Different Bike Engines', fontsize=16, pad=20)
        plt.xlabel('Bike Engine')
        plt.ylabel(name)
        plt.xticks(range(len(bike_labels)), bike_labels, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        # Annotate values
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + np.max(values)*0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = PLOT_DIR / f"{name.lower()}_comparison.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

# ------------------- Main Execution -------------------
if __name__ == '__main__':
    print("Loading and processing bike engine sounds...")
    X, y, bike_names = load_dataset()

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
        plot_feature_comparisons(X, bike_names)

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