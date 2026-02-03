import librosa
import numpy as np
import torch

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH

# ------------------ Utility ------------------
def safe_load_audio(path):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if len(y) < SAMPLE_RATE:
            raise ValueError("Audio too short")
        return y
    except Exception as e:
        print(f"[WARNING] Failed to load {path}: {e}")
        return None

# ------------------ Spectral Feature Stack ------------------
def extract_features(y):
    """
    Extracts frame-level acoustic features.
    Output shape: (num_features, time_frames)
    """

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))

    features = []

    # Spectral descriptors
    features.append(librosa.feature.spectral_centroid(S=S))
    features.append(librosa.feature.spectral_bandwidth(S=S))
    features.append(librosa.feature.spectral_flatness(S=S))
    features.append(librosa.feature.spectral_rolloff(S=S))
    features.append(librosa.feature.spectral_contrast(S=S))
    features.append(librosa.feature.spectral_flux(S=S))
    features.append(librosa.feature.zero_crossing_rate(y))

    # MFCCs + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features.extend([mfcc, delta, delta2])

    # Stack all
    feat = np.vstack(features)

    # Normalization (per recording)
    feat = (feat - np.mean(feat, axis=1, keepdims=True)) / \
           (np.std(feat, axis=1, keepdims=True) + 1e-8)

    return torch.tensor(feat, dtype=torch.float32)
