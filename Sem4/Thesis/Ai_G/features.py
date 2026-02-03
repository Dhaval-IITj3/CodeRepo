# features.py
import numpy as np
import librosa
import torch


def extract_features(waveform, sr=22050, frame_length=1024, hop_length=512, n_mfcc=13):
    """
    Extracts frame-level acoustic features from waveform.

    Args:
        waveform (np.array): Audio waveform.
        sr (int): Sample rate.
        frame_length (int): Frame size for STFT.
        hop_length (int): Hop size.
        n_mfcc (int): Number of MFCCs.

    Returns:
        np.array: (num_features, num_frames)
    """
    if len(waveform) < frame_length:
        print("Warning: Audio too short. Padding with zeros.")
        waveform = np.pad(waveform, (0, frame_length - len(waveform)))

    # Compute STFT once for efficiency
    stft = librosa.stft(waveform, n_fft=frame_length, hop_length=hop_length, window='hann')

    # Spectral features
    spec_centroid = librosa.feature.spectral_centroid(S=np.abs(stft), sr=sr)
    spec_crest = np.max(np.abs(stft), axis=0) / (np.sum(np.abs(stft), axis=0) + 1e-10)  # Custom crest
    spec_decrease = librosa.feature.spectral_rolloff(S=np.abs(stft), sr=sr,
                                                     roll_percent=0.5) - librosa.feature.spectral_rolloff(
        S=np.abs(stft), sr=sr, roll_percent=0.95)
    spec_entropy = -np.sum(np.abs(stft) ** 2 * np.log(np.abs(stft) ** 2 + 1e-10), axis=0) / np.log(
        np.abs(stft).shape[0])
    spec_flatness = librosa.feature.spectral_flatness(S=np.abs(stft))
    spec_flux = librosa.onset.onset_strength(S=np.abs(stft))
    spec_kurtosis = librosa.feature.spectral_bandwidth(S=np.abs(stft), sr=sr, p=4)  # Approx kurtosis via higher moment
    spec_rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft), sr=sr)
    spec_skewness = librosa.feature.spectral_bandwidth(S=np.abs(stft), sr=sr, p=3)  # Approx skewness
    spec_slope = np.gradient(np.mean(np.abs(stft), axis=0))
    spec_spread = librosa.feature.spectral_bandwidth(S=np.abs(stft), sr=sr)

    # Pitch (fundamental frequency)
    pitch, _, _ = librosa.pyin(waveform, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr,
                               frame_length=frame_length, hop_length=hop_length)
    pitch = np.nan_to_num(pitch)  # Handle unvoiced

    # MFCCs with deltas
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Stack all features
    features = np.vstack([
        spec_centroid, spec_crest, spec_decrease, spec_entropy, spec_flatness,
        spec_flux, spec_kurtosis, spec_rolloff, spec_skewness, spec_slope[:spec_centroid.shape[1]],
        spec_spread, pitch.reshape(1, -1),
        mfcc, delta_mfcc, delta2_mfcc
    ])

    # Normalize features (z-score)
    features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-10)

    # Validate shape
    assert features.ndim == 2, "Feature extraction failed: Invalid shape"

    return features