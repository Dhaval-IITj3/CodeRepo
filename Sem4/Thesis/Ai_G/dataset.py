# dataset.py
import os
import shutil
import random
import torch
import torchaudio
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from features import extract_features  # Import from features.py
from natsort import natsorted

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class AudioDataset(Dataset):
    """
    Custom PyTorch Dataset for loading audio files, extracting features, and assigning labels.
    Supports optional augmentation for test set.
    """

    def __init__(self, file_paths, labels, augment=False, target_sr=22050, max_duration_sec=9.0):
        """
        Args:
            file_paths (list): List of audio file paths.
            labels (list): Corresponding labels (0: Fresh, 1: Moderate, 2: Degraded).
            augment (bool): Whether to apply Gaussian noise augmentation (for test set only).
            target_sr (int): Target sample rate.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.target_sr = target_sr
        self.max_samples = int(max_duration_sec * target_sr)  # e.g. 9 * 22050 = 198450 samples

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio with validation
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.size(0) == 0 or waveform.numel() == 0:
                raise ValueError("Empty waveform")

            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            if waveform.size(0) > 1:  # Convert to mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.squeeze()  # Ensure 1D
            # === IMPORTANT CHANGE: Keep only first 9 seconds ===
            waveform = waveform[:self.max_samples]
            if waveform.numel() == 0:
                raise ValueError("Audio became empty after truncation")

        except Exception as e:
            print(f"Error loading {path}: {e}. Skipping.")
            return None, None  # Will be handled in collate

        # Apply augmentation if enabled (Gaussian noise with SNR control)
        if self.augment:
            noise = torch.randn_like(waveform) * (waveform.std() / 10)  # SNR ~20dB
            waveform = waveform + noise

        # Extract features
        features = extract_features(waveform.numpy(), self.target_sr)
        features = torch.tensor(features, dtype=torch.float32)  # (num_features, num_frames)

        return features, label


def collate_fn(batch):
    """
    Custom collate function to pad variable-length feature sequences.
    Handles None items from corrupted files.
    """
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    features = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    return features_padded, labels


def prepare_datasets(data_root='Data/TrainData', fresh_thresh=0.33, mod_thresh=0.75,
                     splits=(0.7, 0.15, 0.15)):
    """
    Collects all audio files, assigns labels based on sequence position per bike folder,
    splits into train/val/test, and creates physical copies.

    Args:
        data_root (str): Root directory of the dataset.
        fresh_thresh (float): Threshold for 'Fresh' label (normalized index).
        mod_thresh (float): Threshold for 'Moderate' label.
        splits (tuple): Train/val/test split ratios.

    Returns:
        train_loader, val_loader, test_loader
    """
    all_paths = []
    all_labels = []

    # Collect and label files
    for subdir in os.listdir(data_root):
        subpath = os.path.join(data_root, subdir)
        if not os.path.isdir(subpath):
            continue
        files = natsorted([f for f in os.listdir(subpath) if f.endswith('.ogg')])
        if not files:
            continue

        # Parse recording numbers and sort
        sorted_files = natsorted(files)

        print(f"Processing {subdir}, total files: {len(sorted_files)}")
        print(f"Files: {sorted_files}")
        num_files = len(sorted_files)

        for i, file in enumerate(sorted_files):
            norm_i = i / (num_files - 1) if num_files > 1 else 0
            if norm_i < fresh_thresh:
                label = 0  # Fresh
            elif norm_i < mod_thresh:
                label = 1  # Moderate
            else:
                label = 2  # Degraded
            all_paths.append(os.path.join(subpath, file))
            all_labels.append(label)

    # Validate label consistency (at least one per class, etc.)
    if len(set(all_labels)) < 3:
        print("Warning: Not all classes are represented in the dataset.")

    # Split datasets (stratified by label)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=splits[1] + splits[2], stratify=all_labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=splits[2] / (splits[1] + splits[2]), stratify=temp_labels, random_state=42
    )

    # Create directories and copy files
    for split, paths in [('train', train_paths), ('val', val_paths), ('test', test_paths)]:
        os.makedirs(split, exist_ok=True)
        for path in paths:
            dest = os.path.join(split, os.path.basename(path))
            shutil.copy(path, dest)

    # Create datasets and loaders
    train_dataset = AudioDataset(train_paths, train_labels, augment=False, max_duration_sec=9.0)
    val_dataset = AudioDataset(val_paths, val_labels, augment=False, max_duration_sec=9.0)
    test_dataset = AudioDataset(test_paths, test_labels, augment=True, max_duration_sec=9.0)  # Augmentation on test

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader