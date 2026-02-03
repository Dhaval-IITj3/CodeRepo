import torch
from torch.utils.data import DataLoader
from dataset import EngineSoundDataset
from model import CNNLSTM
from sklearn.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_ds = EngineSoundDataset("data/test", augment=True)
test_loader = DataLoader(test_ds, batch_size=8)

model = CNNLSTM().to(DEVICE)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        preds = torch.argmax(model(x), dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

print(classification_report(y_true, y_pred, target_names=["Fresh", "Moderate", "Degraded"]))
