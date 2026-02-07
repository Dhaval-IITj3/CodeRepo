# evaluate.py
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from model import BikeLSTMClassifier  # Import from model.py


def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluates the model on the test set and prints metrics.

    Args:
        model: Trained CNNLSTM model.
        test_loader: Test DataLoader.
        device (str): 'cuda' or 'cpu'.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if batch[0] is None:
                continue
            features, labels, lengths = batch  # ← unpack 3 values
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(features, lengths)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision (macro): {prec:.4f}")
    print(f"Test Recall (macro): {rec:.4f}")
    print(f"Test F1-Score (macro): {f1:.4f}")
    print("\nClassification Report:\n")

    # Print classification report
    print(f"True labels in test: {sorted(set(all_labels))}")
    print(f"Predicted labels in test: {sorted(set(all_preds))}")

    print(classification_report(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        target_names=['Fresh', 'Moderate', 'Degraded'],
        zero_division=0,
        digits=4
    ))

    print("True class counts:", Counter(all_labels))
    print("Predicted class counts:", Counter(all_preds))
