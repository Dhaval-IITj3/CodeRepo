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

    def __init__(self, bike_to_files, abs_data_root, max_duration_sec=9.0, target_sr=22050, augment=False):
        self.bike_to_files = bike_to_files  # dict: bike_id → list of absolute paths
        self.abs_data_root = abs_data_root
        self.bike_ids = list(bike_to_files.keys())
        self.max_samples = int(max_duration_sec * target_sr)
        self.target_sr = target_sr
        self.augment = augment

    def __len__(self):
        return len(self.bike_ids)

    def __getitem__(self, idx):
        bike_id = self.bike_ids[idx]
        file_paths = self.bike_to_files[bike_id]

        # Sort by recording number extracted from filename
        def get_rec_num(p):
            fname = os.path.basename(p)
            try:
                return int(fname.split('_')[-1].replace('.ogg', ''))
            except:
                return 0

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


def prepare_datasets(
        data_root='Data/TrainData',
        batch_size=4,
        splits=(0.70, 0.15, 0.15),
        max_duration_sec=9.0,
        target_sr=22050
):
    """
    - Collects absolute paths of all .ogg files
    - Groups by bike folder
    - Splits individual recordings (not bikes) into train/val/test
    - Makes physical copies into train/val/test folders
    - Returns loaders (bike-sequence style)
    """
    abs_data_root = os.path.abspath(data_root)

    # 1. Collect all absolute paths + labels + bike grouping
    all_files = []
    bike_to_files = defaultdict(list)
    label_list = []  # for stratified split

    for subdir in os.listdir(abs_data_root):
        subpath = os.path.join(abs_data_root, subdir)
        if not os.path.isdir(subpath):
            continue
        files = [f for f in os.listdir(subpath) if f.lower().endswith('.ogg')]
        if not files:
            continue

        abs_files = [os.path.join(subpath, f) for f in files]

        # Sort by recording number
        def get_num(fname):
            try:
                return int(os.path.basename(fname).split('_')[-1].replace('.ogg', ''))
            except:
                return 9999

        sorted_abs_files = natsorted(abs_files, key=get_num)
        num_files = len(sorted_abs_files)

        for i, abspath in enumerate(sorted_abs_files):
            norm_i = i / (num_files - 1) if num_files > 1 else 0.0
            label = 0 if norm_i < 0.33 else (1 if norm_i < 0.66 else 2)

            all_files.append((abspath, label, subdir))  # path, label, bike_id
            bike_to_files[subdir].append(abspath)
            label_list.append(label)

    if not all_files:
        raise ValueError("No .ogg files found in the data directory.")

    print(f"Found {len(all_files)} recordings across {len(bike_to_files)} bikes.")
    print("Class distribution:", Counter(label_list))

    # 2. Stratified split of recordings (not bikes)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, label_list,
        test_size=splits[1] + splits[2],
        stratify=label_list,
        random_state=42
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels,
        test_size=splits[2] / (splits[1] + splits[2]),
        stratify=temp_labels,
        random_state=42
    )

    # 3. Create directories and copy files (absolute paths preserved in structure)
    split_dirs = {'train': train_files, 'val': val_files, 'test': test_files}

    for split_name, file_list in split_dirs.items():
        os.makedirs(split_name, exist_ok=True)
        for abspath, _, bike_id in file_list:
            rel_path = os.path.relpath(abspath, abs_data_root)
            dest_path = os.path.join(split_name, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(abspath, dest_path)

    # 4. Build bike → list of files (now from copied locations)
    train_bike_files = defaultdict(list)
    val_bike_files = defaultdict(list)
    test_bike_files = defaultdict(list)

    for split_name, file_list in split_dirs.items():
        target_dict = {'train': train_bike_files, 'val': val_bike_files, 'test': test_bike_files}[split_name]
        base = os.path.abspath(split_name)
        for abspath, _, bike_id in file_list:
            target_dict[bike_id].append(os.path.join(base, os.path.relpath(abspath, abs_data_root)))

    # 5. Create datasets
    train_ds = BikeSequenceDataset(train_bike_files, abs_data_root=abs_data_root, augment=True,
                                   max_duration_sec=max_duration_sec, target_sr=target_sr)
    val_ds = BikeSequenceDataset(val_bike_files, abs_data_root=abs_data_root, augment=False,
                                 max_duration_sec=max_duration_sec, target_sr=target_sr)
    test_ds = BikeSequenceDataset(test_bike_files, abs_data_root=abs_data_root, augment=False,
                                  max_duration_sec=max_duration_sec, target_sr=target_sr)

    # 6. DataLoaders — batch_size is configurable
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)

    print(f"→ Train bikes: {len(train_ds)}, Val bikes: {len(val_ds)}, Test bikes: {len(test_ds)}")

    return train_loader, val_loader, test_loader