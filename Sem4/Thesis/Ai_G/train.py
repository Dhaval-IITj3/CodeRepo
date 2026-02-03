# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import CNNLSTM  # Import from model.py


def train_model(train_loader, val_loader, num_features, num_epochs=50, patience=5,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
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
    model = CNNLSTM(num_features=num_features).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    best_val_loss = float('inf')
    early_stop_count = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0, 0
        for features, labels in train_loader:
            if features is None:
                continue
            features, labels = features.to(device), labels.to(device)
            lengths = torch.tensor([f.size(1) for f in features])  # Original lengths
            outputs = model(features, lengths)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                if features is None:
                    continue
                features, labels = features.to(device), labels.to(device)
                lengths = torch.tensor([f.size(1) for f in features])
                outputs = model(features, lengths)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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