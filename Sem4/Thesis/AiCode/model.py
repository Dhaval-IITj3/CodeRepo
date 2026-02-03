import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # -------- CNN --------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # -------- LSTM --------
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.cnn(x)

        # reshape → (B, T, C)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)

        _, (hn, _) = self.lstm(x)
        out = hn[-1]

        return self.fc(out)
