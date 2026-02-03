import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import EngineSoundDataset
from model import CNNLSTM
from utils import set_seed
from tqdm import tqdm
from config import TRAIN_DATA_DIR, VALIDATION_DATA_DIR

set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = EngineSoundDataset(TRAIN_DATA_DIR)
val_ds = EngineSoundDataset(VALIDATION_DATA_DIR)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

model = CNNLSTM().to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, patience=3)

best_loss = float("inf")
patience, counter = 5, 0

for epoch in range(50):
    model.train()
    train_loss = 0

    for x, y in tqdm(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_loss += criterion(model(x), y).item()

    scheduler.step(val_loss)

    print(f"Epoch {epoch} | Train {train_loss:.3f} | Val {val_loss:.3f}")

    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        counter += 1
        if counter >= patience:
            break
