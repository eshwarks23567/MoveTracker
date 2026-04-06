"""
1D Convolutional Neural Network for HAR.

Architecture:
    Input (window_size × channels)
    → Conv1D(64) → BN → ReLU → MaxPool
    → Conv1D(128) → BN → ReLU → MaxPool
    → Conv1D(256) → BN → ReLU
    → GlobalAvgPool → Dropout → FC(128) → FC(num_classes)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


class ConvBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU → optional MaxPool."""

    def __init__(self, in_channels, out_channels, kernel_size=5,
                 stride=1, pool_size=2, use_pool=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_size) if use_pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNN1D(nn.Module):
    """
    1D CNN for time-series activity recognition.

    Parameters
    ----------
    num_channels : int
        Number of input sensor channels.
    num_classes : int
        Number of activity classes.
    dropout : float
        Dropout rate before final classifier.
    """

    def __init__(self, num_channels=None, num_classes=None, dropout=None):
        super().__init__()
        num_channels = num_channels or config.NUM_CHANNELS
        num_classes = num_classes or config.NUM_CLASSES
        dropout = dropout or config.DROPOUT_RATE

        # Input: (batch, channels, window_size) after permute
        self.features = nn.Sequential(
            ConvBlock(num_channels, 64, kernel_size=7, pool_size=2),
            ConvBlock(64, 128, kernel_size=5, pool_size=2),
            ConvBlock(128, 256, kernel_size=3, use_pool=False),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, window_size, num_channels)

        Returns
        -------
        logits : torch.Tensor, shape (batch, num_classes)
        """
        # Conv1D expects (batch, channels, length)
        x = x.permute(0, 2, 1)

        x = self.features(x)          # (batch, 256, T')
        x = self.global_pool(x)       # (batch, 256, 1)
        x = x.squeeze(-1)             # (batch, 256)
        x = self.dropout(x)
        x = self.classifier(x)        # (batch, num_classes)

        return x


def get_model(**kwargs):
    """Factory function for CNN1D."""
    return CNN1D(**kwargs)


if __name__ == "__main__":
    model = CNN1D()
    x = torch.randn(4, config.WINDOW_SIZE, config.NUM_CHANNELS)
    out = model(x)
    print(f"CNN1D: Input {x.shape} → Output {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
