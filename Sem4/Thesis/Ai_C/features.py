import librosa
import numpy as np
import torch
from scipy.stats import skew, kurtosis

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, WINDOW, FIXED_SAMPLES

import librosa
import numpy as np


def spectral_entropy(S, eps=1e-10):
    """Spectral entropy per frame"""
    ps = S / (np.sum(S, axis=0, keepdims=True) + eps)
    return -np.sum(ps * np.log2(ps + eps), axis=0, keepdims=True)


def spectral_crest(S, eps=1e-10):
    """Spectral crest factor"""
    return np.max(S, axis=0, keepdims=True) / (np.mean(S, axis=0, keepdims=True) + eps)


def spectral_decrease(S, eps=1e-10):
    """Spectral decrease"""
    k = np.arange(1, S.shape[0] + 1).reshape(-1, 1)
    return np.sum((S[1:] - S[0:-1]) / (k[1:] + eps), axis=0, keepdims=True)


def extract_features(audio_path: str) -> np.ndarray:
    """
    Extracts frame-level acoustic features and returns
    a tensor of shape (T, F).
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        raise RuntimeError(f"Could not read audio: {audio_path}")

    if len(y) < FIXED_SAMPLES:
        print(f"Audio too short. Audio length {len(y)}. Padding with zeros.")
        y = np.pad(y, (0, FIXED_SAMPLES - len(y)))
    else:
        y = y[:FIXED_SAMPLES]

    # STFT magnitude
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW))

    # ---------- Spectral Features ----------
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(S=S)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    spectral_spread = librosa.feature.spectral_bandwidth(S=S)
    spectral_flux = librosa.onset.onset_strength(S=S)
    spectral_flux = spectral_flux.reshape(1, -1)

    spectral_entropy_val = spectral_entropy(S)
    spectral_crest_val = spectral_crest(S)
    spectral_decrease_val = spectral_decrease(S)

    # Higher-order statistics
    spectral_skewness = skew(S, axis=0, keepdims=True)
    spectral_kurtosis = kurtosis(S, axis=0, keepdims=True)

    # Spectral slope
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT).reshape(-1, 1)
    spectral_slope = np.sum(freqs * S, axis=0, keepdims=True) / (np.sum(S, axis=0, keepdims=True) + 1e-10)

    # ---------- Pitch ----------
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch = np.sum(pitches * mags, axis=0) / (np.sum(mags, axis=0) + 1e-10)
    pitch = pitch.reshape(1, -1)

    # ---------- MFCCs ----------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # ---------- Stack all features ----------
    feature_stack = np.vstack([
        spectral_centroid,
        spectral_crest_val,
        spectral_decrease_val,
        spectral_entropy_val,
        spectral_flatness,
        spectral_flux,
        spectral_kurtosis,
        spectral_rolloff,
        spectral_skewness,
        spectral_slope,
        spectral_spread,
        pitch,
        mfcc,
        mfcc_delta,
        mfcc_delta2
    ])

    # ---------- Normalize (per feature) ----------
    feature_stack = (feature_stack - np.mean(feature_stack, axis=1, keepdims=True)) / \
                    (np.std(feature_stack, axis=1, keepdims=True) + 1e-6)

    # Return (T, F)
    return feature_stack.T