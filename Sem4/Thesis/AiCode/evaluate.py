import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from model import CNNLSTM
from config import *

def evaluate(test_ds):
    loader = DataLoader(test_ds, batch_size=1)
    model = CNNLSTM(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            y_pred.append(out.argmax(1).item())
            y_true.append(y.item())

    print(classification_report(
        y_true, y_pred, target_names=CLASS_NAMES
    ))
