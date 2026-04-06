"""
Recurrent Neural Networks (LSTM / GRU) for HAR.

Architecture:
    Input (window_size × channels)
    → Bidirectional LSTM/GRU (2 layers, hidden_size=128)
    → Take last hidden state
    → Dropout → FC(128) → FC(num_classes)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


class RNNModel(nn.Module):
    """
    Bidirectional LSTM or GRU for time-series activity recognition.

    Parameters
    ----------
    num_channels : int
        Number of input sensor channels.
    num_classes : int
        Number of activity classes.
    hidden_size : int
        Hidden units per direction in the RNN.
    num_layers : int
        Number of stacked RNN layers.
    rnn_type : str
        'lstm' or 'gru'.
    dropout : float
        Dropout rate.
    bidirectional : bool
        Whether to use bidirectional RNN.
    """

    def __init__(self, num_channels=None, num_classes=None, hidden_size=128,
                 num_layers=2, rnn_type='lstm', dropout=None, bidirectional=True):
        super().__init__()

        num_channels = num_channels or config.NUM_CHANNELS
        num_classes = num_classes or config.NUM_CLASSES
        dropout = dropout or config.DROPOUT_RATE

        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        rnn_cls = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU

        self.rnn = rnn_cls(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        fc_input_size = hidden_size * self.num_directions

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_size, 128),
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
        # x is already (batch, seq_len, input_size)
        rnn_out, _ = self.rnn(x)     # (batch, seq_len, hidden*directions)

        # Use the last time step's output
        last_output = rnn_out[:, -1, :]  # (batch, hidden*directions)

        x = self.dropout(last_output)
        x = self.classifier(x)

        return x


class LSTMModel(RNNModel):
    """Bidirectional LSTM convenience wrapper."""
    def __init__(self, **kwargs):
        kwargs['rnn_type'] = 'lstm'
        super().__init__(**kwargs)


class GRUModel(RNNModel):
    """Bidirectional GRU convenience wrapper."""
    def __init__(self, **kwargs):
        kwargs['rnn_type'] = 'gru'
        super().__init__(**kwargs)


def get_model(rnn_type='lstm', **kwargs):
    """Factory function for RNN models."""
    return RNNModel(rnn_type=rnn_type, **kwargs)


if __name__ == "__main__":
    for rnn_type in ['lstm', 'gru']:
        model = RNNModel(rnn_type=rnn_type)
        x = torch.randn(4, config.WINDOW_SIZE, config.NUM_CHANNELS)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{rnn_type.upper()}: Input {x.shape} → Output {out.shape}, "
              f"Params: {n_params:,}")
