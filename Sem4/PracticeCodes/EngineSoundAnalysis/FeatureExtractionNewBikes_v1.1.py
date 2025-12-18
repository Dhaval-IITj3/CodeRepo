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


DATA_DIR_NAME = 'Resources\\EngineSoundsNewBikes'
BASE_DIR = Path(__file__).parent
BASE_DIR = Path(BASE_DIR, "../").resolve()
PLOT_DIR = Path.joinpath(BASE_DIR, 'EngineSoundAnalysis', 'Plots', 'NewBikeFeatureExtraction')
data_dir = str(Path.joinpath(BASE_DIR, DATA_DIR_NAME))

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

if not os.path.exists(data_dir):
    print(f"Error: {data_dir} does not exist.")
    exit()

# Section 1: Feature Extraction Functions
# Define functions for each of the 12 spectral features based on magnitude spectrum S and frequencies freqs.
def spectral_centroid(S, freqs):
    return np.sum(freqs * S) / (np.sum(S) + 1e-8)


def spectral_crest(S):
    return np.max(S) / (np.mean(S) + 1e-8)


def spectral_decrease(S):
    if len(S) < 2:
        return 0.0
    dec = np.sum((S[1:] - S[0]) / (np.arange(1, len(S)) + 1e-8))
    return dec / (np.sum(S[1:]) + 1e-8)


def spectral_entropy(S):
    ps = S / (np.sum(S) + 1e-8)
    return -np.sum(ps * np.log2(ps + 1e-8))


def spectral_flatness(S):
    log_S = np.log(S + 1e-8)
    geo_mean = np.exp(np.mean(log_S))
    arith_mean = np.mean(S)
    return geo_mean / (arith_mean + 1e-8)


def spectral_flux(S_prev, S_curr):
    return np.sum((np.log(S_curr + 1e-8) - np.log(S_prev + 1e-8)) ** 2)


def spectral_kurtosis(S):
    mean_S = np.mean(S)
    std_S = np.std(S) + 1e-8
    return np.mean(((S - mean_S) / std_S) ** 4)


def spectral_rolloff(S, freqs, percentile=0.85):
    total_energy = np.sum(S)
    cum_energy = np.cumsum(S)
    idx = np.where(cum_energy >= percentile * total_energy)[0][0]
    return freqs[idx]


def spectral_skewness(S):
    mean_S = np.mean(S)
    std_S = np.std(S) + 1e-8
    return np.mean(((S - mean_S) / std_S) ** 3)


def spectral_slope(S, freqs):
    if len(freqs) < 2:
        return 0.0
    return np.polyfit(freqs, S, 1)[0]


def spectral_spread(S, freqs, centroid):
    return np.sqrt(np.sum(((freqs - centroid) ** 2 * S) / (np.sum(S) + 1e-8)))


def estimate_pitch(y_frame, sr):
    # Simple autocorrelation-based pitch estimation for a frame.
    autocorr = np.correlate(y_frame, y_frame, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    peaks = np.diff(np.sign(np.diff(autocorr))) < 0
    if np.sum(peaks) == 0:
        return 0.0
    peak_idx = np.where(peaks)[0][0] + 1  # First peak after zero
    return sr / peak_idx if peak_idx > 0 else 0.0


# Section 2: Load and Process Audio File
def load_audio(filename):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(filename)
    array = pygame.sndarray.array(sound)
    if array.ndim == 2:  # Stereo to mono
        array = np.mean(array, axis=1)
    sr = pygame.mixer.get_init()[0]  # Sample rate
    return array.astype(np.float32) / 32768.0, sr  # Normalize to [-1,1]


def extract_features(filename):
    y, sr = load_audio(filename)
    window_size = int(0.015 * sr)  # 15 ms
    hop_size = int(0.005 * sr)  # 5 ms hop
    noverlap = window_size - hop_size  # 10 ms overlap

    f, t, Zxx = stft(y, fs=sr, window=windows.hamming(window_size), nperseg=window_size, noverlap=noverlap)
    mag = np.abs(Zxx)

    features = []
    pitches = []
    prev_S = None
    fluxes = []

    for i in range(mag.shape[1]):
        S = mag[:, i]
        centroid = spectral_centroid(S, f)
        crest = spectral_crest(S)
        decrease = spectral_decrease(S)
        entropy = spectral_entropy(S)
        flatness = spectral_flatness(S)
        kurtosis = spectral_kurtosis(S)
        rolloff = spectral_rolloff(S, f)
        skewness = spectral_skewness(S)
        slope = spectral_slope(S, f)
        spread = spectral_spread(S, f, centroid)

        # Pitch from time-domain frame
        start = i * hop_size
        end = start + window_size
        y_frame = y[start:end] * windows.hamming(len(y[start:end]))
        pitch = estimate_pitch(y_frame, sr)
        pitches.append(pitch)

        # Flux requires previous frame
        if prev_S is not None:
            flux = spectral_flux(prev_S, S)
            fluxes.append(flux)
        prev_S = S

        # Collect per-frame features (exclude flux for now)
        frame_feats = [centroid, crest, decrease, entropy, flatness, kurtosis, rolloff, skewness, slope, spread]
        features.append(frame_feats)

    # Average over frames
    mean_features = np.mean(features, axis=0)
    mean_pitch = np.mean(pitches)
    mean_flux = np.mean(fluxes) if fluxes else 0.0

    # Insert flux after flatness (based on order in paper)
    final_vector = np.insert(mean_features, 5, mean_flux)  # Insert at index 5
    final_vector = np.append(final_vector, mean_pitch)  # Append pitch
    return final_vector


# Section 3: Data Loading and Preparation
def load_dataset(dir_path=f"{data_dir}"):
    features = []
    labels = []

    classes = os.listdir(data_dir)

    for cls in classes:
        path = os.path.join(data_dir, cls)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.ogg'):
                    path = os.path.join(path, file)
                    try:
                        feat = extract_features(path)
                        features.append(feat)
                        labels.append(cls)
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
        elif os.path.isfile(path) and path.endswith('.ogg'):
            try:
                feat = extract_features(path)
                features.append(feat)
                labels.append(cls)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    return np.array(features), np.array(labels)


# Section 4: Define and Train PyTorch Model
class BikeClassifier(nn.Module):
    def __init__(self, input_size=12):
        super(BikeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)  # 2 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_model(X, y, epochs=100, lr=0.01):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = BikeClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model


# Section 5: Plotting Features
def plot_features(X, y, feature_names):
    group1 = X[y == 0]
    group2 = X[y == 1]

    for i, name in enumerate(feature_names):
        fig, ax = plt.subplots()
        ax.boxplot([group1[:, i], group2[:, i]], labels=['Group 1', 'Group 2'])
        ax.set_title(f'{name} for Engine Groups')
        ax.set_ylabel(name)
        plt.savefig(f'{name.lower().replace(" ", "_")}_plot.png')
        plt.close()

# Main Execution
if __name__ == '__main__':
    X, y = load_dataset()
    feature_names = ['Centroid', 'Crest', 'Decrease', 'Entropy', 'Flatness', 'Flux', 'Kurtosis', 'Rolloff', 'Skewness',
                     'Slope', 'Spread', 'Pitch']

    # Plot features
    plot_features(X, y, feature_names)

    # Train model
    model = train_model(X, y)
    print('Model trained. To predict on new data, use model(torch.tensor(new_feat)).argmax()')