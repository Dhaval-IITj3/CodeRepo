# model.py   ← modified to first embed each recording → then LSTM over bike sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecordingEmbedder(nn.Module):
    """CNN that turns one recording feature map → fixed-size embedding"""

    def __init__(self, num_features):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))  # global average pooling → fixed size
        )
        self.flat_dim = 128

    def forward(self, x):
        # x: (batch * num_rec, 1, num_feat, num_frames)
        x = self.cnn(x)  # → (..., 128, 1, 1)
        x = x.view(x.size(0), -1)  # → (..., 128)
        return x


class BikeLSTMClassifier(nn.Module):
    """
    Model that:
    1. Embeds each recording independently with CNN
    2. Feeds sequence of embeddings to LSTM
    3. Predicts one label per bike
    """

    def __init__(self, num_features, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        self.embedder = RecordingEmbedder(num_features)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # x: (batch, max_num_rec, num_features, num_frames)
        b, max_rec, f, t = x.shape

        # Flatten to process each recording independently
        x_flat = x.view(b * max_rec, 1, f, t)  # (b*max_rec, 1, f, t)
        embeds = self.embedder(x_flat)  # (b*max_rec, 128)
        embeds = embeds.view(b, max_rec, -1)  # (b, max_rec, 128)

        # Pack variable-length sequences
        packed = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hn, _) = self.lstm(packed)

        # Take last hidden state
        out = hn[-1]  # (batch, hidden_size)
        logits = self.fc(out)
        return logits