import torch
import random
import numpy as np
from pathlib import Path


# ---------------- File paths ---------------
DATA_DIR_NAME = 'Data'
BASE_DIR = Path(__file__).parent
BASE_DIR = Path(BASE_DIR, ".").resolve()
DATA_DIR = Path.joinpath(BASE_DIR, DATA_DIR_NAME).absolute()
TRAIN_DATA_DIR = Path.joinpath(BASE_DIR, Path(DATA_DIR_NAME), 'TrainData')
VALIDATION_DATA_DIR = Path.joinpath(BASE_DIR, Path(DATA_DIR_NAME), 'ValidationData')

# ------------------ Audio ------------------
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
WINDOW = "hann"
MAX_DURATION = 9.0  # seconds
FIXED_SAMPLES = int(SAMPLE_RATE * MAX_DURATION)

# ------------------ Labels ------------------
LABELS = {"Fresh": 0, "Moderate": 1, "Degraded": 2}
NUM_CLASSES = len(LABELS.keys())

# Proportion-based label split within each bike folder
FRESH_RATIO = 0.33
MODERATE_RATIO = 0.75
# remaining → Degraded

# ------------------ Training ------------------
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
