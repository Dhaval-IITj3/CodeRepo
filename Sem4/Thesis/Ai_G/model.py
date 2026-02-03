# model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for audio classification.
    """

    def __init__(self, num_features, num_classes=3, hidden_size=128, num_layers=2):
        """
        Args:
            num_features (int): Number of input features (height of input).
            num_classes (int): Number of output classes.
            hidden_size (int): LSTM hidden size.
            num_layers (int): Number of LSTM layers.
        """
        super(CNNLSTM, self).__init__()

        # CNN front-end
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        # LSTM back-end
        self.lstm = nn.LSTM(input_size=128 * (num_features // 8),  # After pooling: height // 8
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=False)

        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        """
        Args:
            x (torch.Tensor): Input features (batch, num_features, num_frames)
            lengths (torch.Tensor): Original sequence lengths for packing.

        Returns:
            torch.Tensor: Logits (batch, num_classes)
        """
        # Add channel dim: (batch, 1, num_features, num_frames)
        x = x.unsqueeze(1)

        # CNN
        x = self.cnn(x)  # (batch, 128, num_features//8, num_frames//8)

        # Reshape for LSTM: (batch, num_frames//8, 128 * num_features//8)
        batch, channels, height, time = x.shape
        x = x.view(batch, time, channels * height)

        # Pack for LSTM (handle variable lengths)
        if lengths is not None:
            lengths = lengths // 8  # Adjust for pooling
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM
        lstm_out, (hn, cn) = self.lstm(x)

        # Get last hidden state
        if lengths is not None:
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        hn = hn[-1]  # Last layer's hidden

        # FC
        out = self.fc(hn)

        return out