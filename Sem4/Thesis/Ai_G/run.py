# run.py
import argparse
import os
import torch
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


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the bike-level oil degradation model")
    parser.add_argument('--model-path', '-m', type=str, default=None,
                        help="Path to pre-trained .pth model file. If valid, skips training and evaluates only.")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Batch size for data loaders")
    parser.add_argument('--epochs', type=int, default=80,
                        help="Maximum training epochs (only used if training)")
    parser.add_argument('--patience', type=int, default=12,
                        help="Early stopping patience (only used if training)")

    args = parser.parse_args()

    # Decide whether we need training mode
    train_mode = (args.model_path is None)  # If model-path given → assume eval-only (will check existence later)

    print("Preparing datasets...")
    train_loader, val_loader, test_loader = prepare_datasets(
        root_dir=str(ENGINE_SOUND_DIR),
        batch_size=args.batch_size,
        is_training=train_mode,  # ← key change: conditional train loader creation
        val_ratio=0.4,
        test_ratio=0.4
    )

    # Detect number of features — try from val_loader or test_loader if train_loader is None
    num_features = None
    loaders_to_check = [train_loader, val_loader, test_loader]
    for loader in loaders_to_check:
        if loader is not None:
            for batch in loader:
                if batch[0] is not None:
                    features = batch[0]  # (batch, max_rec, num_features, frames)
                    num_features = features.shape[2]
                    break
            if num_features is not None:
                break

    if num_features is None:
        raise ValueError("Could not determine num_features from any loader. Check data.")

    print(f"Detected {num_features} acoustic features per recording")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BikeLSTMClassifier(num_features=num_features).to(device)

    # ──────────────────────────────────────────────────────────────
    # Decide: Load existing model OR Train new one
    # ──────────────────────────────────────────────────────────────
    model_loaded = False
    model_path_to_use = None

    if args.model_path:
        # Handle relative vs absolute path
        if os.path.isabs(args.model_path):
            candidate_path = args.model_path
        else:
            candidate_path = os.path.join(BASE_DIR, args.model_path)

        candidate_path = os.path.normpath(os.path.abspath(candidate_path))

        if os.path.isfile(candidate_path) and candidate_path.lower().endswith('.pth'):
            print(f"Loading pre-trained model from: {candidate_path}")
            model.load_state_dict(torch.load(candidate_path, map_location=device))
            model.eval()
            model_loaded = True
            model_path_to_use = candidate_path
        else:
            print(f"Warning: Model path '{args.model_path}' not found or invalid.")
            print(f"Checked: {candidate_path}")
            print("Falling back to training a new model...")
            model_loaded = False

    if not model_loaded:
        if train_loader is None:
            print("Error: No training data available, but no valid pre-trained model was loaded.")
            print("Cannot proceed. Provide a valid --model-path or ensure training data exists.")
            return

        print("Training new model...")
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model_class=BikeLSTMClassifier,
            num_features=num_features,
            num_epochs=args.epochs,
            patience=args.patience,
            device=device
        )
        # Assume train_model saves the best model here:
        model_path_to_use = "best_model.pth"
        print(f"Training completed. Best model saved as: {model_path_to_use}")

    # ──────────────────────────────────────────────────────────────
    # Evaluation (always run if we have test_loader)
    # ──────────────────────────────────────────────────────────────
    if test_loader is None:
        print("Warning: No test loader created. Skipping evaluation.")
    else:
        print("\n" + "=" * 60)
        if model_loaded:
            print(f"Evaluating pre-loaded model from: {model_path_to_use}")
        else:
            print("Evaluating freshly trained model")
        print("=" * 60 + "\n")

        evaluate_model(model, test_loader, device=device)

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
