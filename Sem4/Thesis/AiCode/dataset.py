import os
import shutil
from torch.utils.data import Dataset
from features import extract_features, safe_load_audio
from config import CLASS_NAMES

class EngineOilDataset(Dataset):
    """
    CNN-LSTM Dataset:
    Returns tensor of shape (1, features, time)
    """

    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]

        y = safe_load_audio(path)
        if y is None:
            raise RuntimeError(f"Corrupt file: {path}")

        features = extract_features(y)
        features = features.unsqueeze(0)  # (1, F, T)

        return features, label

# ------------------ Label Assignment ------------------
def assign_labels(files):
    """
    Assign labels based on temporal ordering.
    """
    n = len(files)
    labels = []

    for i in range(n):
        ratio = i / n
        if ratio < 0.33:
            labels.append(CLASS_NAMES.index("Fresh"))
        elif ratio < 0.66:
            labels.append(CLASS_NAMES.index("Moderate"))
        else:
            labels.append(CLASS_NAMES.index("Degraded"))

    return labels
