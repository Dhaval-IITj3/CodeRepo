# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import BikeLSTMClassifier  # Import from model.py
from datetime import datetime


def train_model(train_loader, val_loader, model_class, num_features, num_epochs=80, patience=12, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model_class(num_features=num_features, hidden_size=128, num_layers=2).to(device)
    """
    Trains the CNN-LSTM model with early stopping.

    Args:
        train_loader, val_loader: DataLoaders.
        num_features (int): Input feature dimension.
        num_epochs (int): Max epochs.
        patience (int): Early stopping patience.
        device (str): 'cuda' or 'cpu'.

    Returns:
        model: Trained model.
    """
    model = model_class(num_features=num_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    best_val_loss = float('inf')
    early_stop_count = 0

    for epoch in range(num_epochs):
        model.train()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            if batch[0] is None:  # safety skip
                continue
            features, labels, lengths = batch  # ← NOW unpack 3 values
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)  # or .cpu() if needed for pack

            optimizer.zero_grad()
            outputs = model(features, lengths)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch[0] is None:
                    continue
                features, labels, lengths = batch
                features = features.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)

                outputs = model(features, lengths)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        timestamp = datetime.now().strftime("%d%m%Y %H:%M:%S:%f")[:-3]  # up to milliseconds
        print(f"{timestamp} Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model