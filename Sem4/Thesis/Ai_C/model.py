import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

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

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.cnn(x)            # (B, C, F', T')
        x = x.mean(dim=2)          # (B, C, T')
        x = x.permute(0, 2, 1)     # (B, T', C)

        out, _ = self.lstm(x)
        out = out[:, -1, :]        # Last timestep
        return self.fc(out)
