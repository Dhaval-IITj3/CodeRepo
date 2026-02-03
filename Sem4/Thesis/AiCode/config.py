import torch
import random
import numpy as np

# ------------------ Reproducibility ------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------ Audio ------------------
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
WINDOW = "hann"
MAX_DURATION = 10.0  # seconds

# ------------------ Labels ------------------
CLASS_NAMES = ["Fresh", "Moderate", "Degraded"]
NUM_CLASSES = len(CLASS_NAMES)

# Proportion-based label split within each bike folder
FRESH_RATIO = 0.33
MODERATE_RATIO = 0.33
# remaining → Degraded

# ------------------ Training ------------------
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
