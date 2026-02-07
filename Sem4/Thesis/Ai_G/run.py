# run.py
from dataset import prepare_datasets
from train import train_model
from evaluate import evaluate_model
from model import BikeLSTMClassifier
from pathlib import Path

DATA_DIR_NAME = "Data"
BASE_DIR = Path(__file__).parent
BASE_DIR = Path(BASE_DIR, ".").resolve()
DATA_DIR = Path.joinpath(BASE_DIR, DATA_DIR_NAME).absolute()
ENGINE_SOUND_DIR = Path.joinpath(BASE_DIR, Path(DATA_DIR_NAME), 'TrainData')


if __name__ == "__main__":
    train_loader, val_loader, test_loader = prepare_datasets(
        data_root=ENGINE_SOUND_DIR,
        batch_size=4,  # ← change freely, e.g. 2, 8, etc.
        splits=(0.7, 0.15, 0.15)
    )

    # Detect num_features from first valid batch
    for batch in train_loader:
        if batch[0] is not None:
            features = batch[0]
            num_features = features.shape[2]  # (batch, max_rec, num_features, frames)
            break
    else:
        raise ValueError("No valid data in train loader")

    model = train_model(
        train_loader, val_loader,
        model_class=BikeLSTMClassifier,
        num_features=num_features
    )

    evaluate_model(model, test_loader)
    print("Experiment completed.")