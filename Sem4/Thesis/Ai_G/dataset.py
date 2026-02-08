# dataset.py  (updated prepare_datasets + helpers)

import os
import shutil
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from features import extract_features
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from natsort import natsorted

# Seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class BikeSequenceDataset(Dataset):
    """
    Returns one bike at a time as a sequence of recordings.
    Each item = (sequence of feature tensors, label)
    Label = degradation state based on the LAST recording in the sequence.
    """

    """
        Returns one bike → sequence of feature tensors + one label (based on last recording)
        """

    def __init__(self, bike_to_files, max_duration_sec=9.0, target_sr=22050, augment=False):
        self.bike_to_files = bike_to_files  # dict: bike_name → list of absolute file paths
        self.bike_names = list(bike_to_files.keys())
        self.max_samples = int(max_duration_sec * target_sr)
        self.target_sr = target_sr
        self.augment = augment

    def __len__(self):
        return len(self.bike_names)

    def __getitem__(self, idx):
        bike_id = self.bike_names[idx]
        file_paths = self.bike_to_files[bike_id]

        # Sort by recording number extracted from filename
        # Sort files by recording number
        def get_rec_num(path):
            fname = os.path.basename(path)
            try:
                num = fname.split('_')[-1].replace('.ogg', '')
                return int(num)
            except:
                return 9999

        file_paths = sorted(file_paths, key=get_rec_num)

        sequence = []
        for path in file_paths:
            try:
                waveform, sr = torchaudio.load(path)
                if waveform.numel() == 0:
                    continue

                if sr != self.target_sr:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                    waveform = resampler(waveform)

                if waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                waveform = waveform.squeeze(0)[:self.max_samples]

                if waveform.numel() == 0:
                    continue

                if self.augment:
                    noise = torch.randn_like(waveform) * (waveform.std() * 0.07)
                    waveform += noise

                feats = extract_features(waveform.numpy(), self.target_sr)
                feats = torch.tensor(feats, dtype=torch.float32)
                sequence.append(feats)

            except Exception as e:
                print(f"Skip {path}: {e}")
                continue

        if not sequence:
            dummy = torch.zeros(51, 100)  # fallback - adjust if your num_features differs
            return dummy.unsqueeze(0), 2  # assume degraded

        sequence = torch.stack(sequence)  # (num_rec, num_features, num_frames)

        # Label based on LAST recording's position
        num_rec = len(file_paths)
        last_norm = (num_rec - 1) / (num_rec - 1) if num_rec > 1 else 0.0
        if last_norm < 0.33:
            label = 0
        elif last_norm < 0.75:
            label = 1
        else:
            label = 2

        return sequence, label


def collate_sequences(batch):
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    if not sequences:
        return None, None, None

    seq_lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long)
    padded_seq = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    return padded_seq, labels, seq_lengths


# ──────────────────────────────────────────────────────────────
#  Custom split function (no sklearn)
# ──────────────────────────────────────────────────────────────

def custom_stratified_split(files_with_labels, val_ratio=0.15, test_ratio=0.15):
    """
    Input: list of (filepath, label, bike_name)
    Returns: train_files, val_files, test_files  (each is list of (path, label, bike))
             All lists are guaranteed non-empty when possible
    """
    if not files_with_labels:
        return [], [], []

    # Group by label
    label_to_files = defaultdict(list)
    for item in files_with_labels:
        label_to_files[item[1]].append(item)

    train, val, test = [], [], []

    # Snapshot to avoid "dictionary changed size" error
    label_items_pairs = list(label_to_files.items())
    for label, items in label_items_pairs:
        train.extend(items[:])
        random.shuffle(items)  # deterministic via seed

        n = len(items)
        if n == 0:
            continue
        elif n <= 2:
            train.append(items[0])
            train.append(items[1])
            val.append(items[0])   # prefer val over test when very small
            val.append(items[1])
        else:
            n_val  = max(1, round(n * val_ratio))
            n_test = max(1, round(n * test_ratio))
            n_train = n - n_val - n_test

            # Ensure non-empty when possible
            n_train = max(0, n_train)

            val.extend(items[n_train:n_train + n_val])
            test.extend(items[n_train + n_val:])

    # Fallback: if any set is empty, move one item from the largest set
    all_sets = [train, val, test]
    set_names = ["train", "val", "test"]

    for i, s in enumerate(all_sets):
        if not s:
            # Find largest non-empty set
            largest_idx = max(range(3), key=lambda j: len(all_sets[j]) if j != i else -1)
            if len(all_sets[largest_idx]) > 1:
                item = all_sets[largest_idx].pop()
                all_sets[i].append(item)
                print(f"Moved one sample to empty {set_names[i]} set")

    return train, val, test


def prepare_datasets(
    root_dir="Data/TrainData",
    val_ratio=0.15,
    test_ratio=0.15,
    is_training=True,
    batch_size=4,
    max_duration_sec=9.0,
    target_sr=22050
):
    """
    Main entry point.
    - Copies files to ValidationData/ and TestData/ if needed
    - Returns train/val/test loaders (train_loader is None if not is_training)
    """
    root_dir = os.path.abspath(root_dir)
    val_dir  = os.path.abspath("Data/ValidationData")
    test_dir = os.path.abspath("Data/TestData")

    os.makedirs(val_dir,  exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # ── Collect all files ────────────────────────────────────────
    all_files = []  # (abs_path, label, bike_name)
    bike_to_files = defaultdict(list)

    for bike_folder in os.listdir(root_dir):
        bike_path = os.path.join(root_dir, bike_folder)
        if not os.path.isdir(bike_path):
            continue

        files = [f for f in os.listdir(bike_path) if f.lower().endswith('.ogg')]
        if not files:
            continue

        files = natsorted(files)  # natural sort by recording number
        abs_files = []
        for fname in files:
            abspath = os.path.join(bike_path, fname)
            try:
                num = int(fname.split('_')[-1].replace('.ogg', ''))
            except:
                num = 9999

            norm_i = (num - 1) / max(1, len(files) - 1) if len(files) > 1 else 0.0
            label = 0 if norm_i < 0.25 else (1 if norm_i < 0.75 else 2)

            all_files.append((abspath, label, bike_folder))
            abs_files.append(abspath)

        bike_to_files[bike_folder] = natsorted(abs_files)

    if not all_files:
        raise ValueError("No .ogg files found.")

    print(f"Found {len(all_files)} recordings across {len(bike_to_files)} bikes.")
    print("Label distribution:", Counter(l for _, l, _ in all_files))

    # ── Custom split ─────────────────────────────────────────────
    train_items, val_items, test_items = custom_stratified_split(
        all_files,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    # ── Copy files to val/test folders ───────────────────────────
    def copy_to_split(items, target_dir):
        copied_paths = []
        for orig_path, label, bike in items:
            rel = os.path.relpath(orig_path, root_dir)
            dest = os.path.join(target_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                shutil.copy2(orig_path, dest)
            copied_paths.append(dest)
        return copied_paths

    val_copied  = copy_to_split(val_items,  val_dir)
    test_copied = copy_to_split(test_items, test_dir)

    # ── Build bike → files mapping for each split ────────────────
    def build_bike_mapping(copied_paths):
        mapping = defaultdict(list)
        for p in copied_paths:
            bike = os.path.basename(os.path.dirname(p))
            mapping[bike].append(p)
        return mapping

    train_bike_files = bike_to_files if is_training else {}
    val_bike_files   = build_bike_mapping(val_copied)
    test_bike_files  = build_bike_mapping(test_copied)

    # ── Create datasets ──────────────────────────────────────────
    train_ds = BikeSequenceDataset(
        train_bike_files,
        max_duration_sec=max_duration_sec,
        target_sr=target_sr,
        augment=True
    ) if is_training and train_bike_files else None

    val_ds = BikeSequenceDataset(
        val_bike_files,
        max_duration_sec=max_duration_sec,
        target_sr=target_sr,
        augment=False
    )

    test_ds = BikeSequenceDataset(
        test_bike_files,
        max_duration_sec=max_duration_sec,
        target_sr=target_sr,
        augment=False
    )

    # ── DataLoaders ──────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sequences
    ) if train_ds is not None else None

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sequences
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sequences
    )

    print(f"→ Train bikes: {len(train_ds.bike_names) if train_ds else 0}")
    print(f"→ Val   bikes: {len(val_ds.bike_names)}")
    print(f"→ Test  bikes: {len(test_ds.bike_names)}")

    return train_loader, val_loader, test_loader
