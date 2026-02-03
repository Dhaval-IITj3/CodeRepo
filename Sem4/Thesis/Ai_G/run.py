# run.py
from dataset import prepare_datasets
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    print("Starting oil degradation classification pipeline...")
    train_loader, val_loader, test_loader = prepare_datasets()

    sample_batch, _ = next(iter(train_loader))
    num_features = sample_batch.shape[-2]

    print(f"Detected {num_features} input features")

    model = train_model(train_loader, val_loader, num_features)
    evaluate_model(model, test_loader)
    print("Experiment finished.")