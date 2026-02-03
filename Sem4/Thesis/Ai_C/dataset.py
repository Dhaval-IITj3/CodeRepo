import os
import shutil
import torch
import torchaudio
from torch.utils.data import Dataset
from features import extract_features
from utils import ensure_dir
import numpy as np
from natsort import natsorted

from config import LABELS, FRESH_RATIO, MODERATE_RATIO



def assign_labels(files):
    """Assign labels based on temporal progression."""
    n = len(files)
    labels = {}
    for i, f in enumerate(natsorted(files)):
        if i < n * FRESH_RATIO:
            labels[f] = LABELS["Fresh"]
        elif i < n * (MODERATE_RATIO):
            labels[f] = LABELS["Moderate"]
        else:
            labels[f] = LABELS["Degraded"]
    return labels


def create_splits(src_root, dst_root, split=(0.7, 0.15, 0.15)):
    ensure_dir(dst_root)

    all_samples = []
    for bike in os.listdir(src_root):
        bike_path = os.path.join(src_root, bike)
        if not os.path.isdir(bike_path):
            continue

        files = [f for f in os.listdir(bike_path) if f.endswith(".ogg")]
        label_map = assign_labels(files)

        for f in files:
            all_samples.append((os.path.join(bike_path, f), label_map[f]))

    np.random.shuffle(all_samples)

    n = len(all_samples)
    splits = {
        "train": all_samples[:int(split[0] * n)],
        "val": all_samples[int(split[0] * n):int(sum(split[:2]) * n)],
        "test": all_samples[int(sum(split[:2]) * n):]
    }

    for split_name, samples in splits.items():
        split_dir = os.path.join(dst_root, split_name)
        ensure_dir(split_dir)
        for src, label in samples:
            dst = os.path.join(split_dir, f"{label}_{os.path.basename(src)}")
            shutil.copy(src, dst)

class EngineSoundDataset(Dataset):
    def __init__(self, root_dir, augment=False):

        if not os.path.exists(root_dir):
            raise ValueError(f"Root directory {root_dir} does not exist.")

        self.samples = []
        self.augment = augment

        for bike in os.listdir(root_dir):
            bike_path = os.path.join(root_dir, bike)
            if not os.path.isdir(bike_path):
                continue

            files = natsorted([
                f for f in os.listdir(bike_path)
                if f.lower().endswith(".ogg")
            ])

            # Assign labels
            label_map = assign_labels(files)

            for f in files:
                self.samples.append((os.path.join(bike_path, f), label_map[f]))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        print(f"Loading Path: {path} with label: {label}")

        features = extract_features(path)

        if self.augment:
            snr_db = np.random.uniform(10, 20)
            signal_power = np.mean(features**2)
            noise_power = signal_power / np.power(10, (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), features.shape)
            features = features + noise

        # (1, F, T) for CNN
        features = torch.tensor(features, dtype=torch.float32).transpose(0, 1).unsqueeze(0)

        return features, label
