"""
Hybrid CNN-LSTM model for HAR.

Architecture:
    Input (window_size × channels)
    → Conv1D(64) → BN → ReLU → MaxPool
    → Conv1D(128) → BN → ReLU → MaxPool
    → Bidirectional LSTM(128)
    → Attention pooling (optional)
    → Dropout → FC(128) → FC(num_classes)

The CNN layers extract local temporal features, while the LSTM
captures longer-range temporal dependencies. This combination
typically outperforms either architecture alone for HAR tasks.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


class TemporalAttention(nn.Module):
    """
    Simple temporal attention mechanism.
    Learns a weighted sum over LSTM time steps.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, rnn_output):
        """
        Parameters
        ----------
        rnn_output : torch.Tensor, shape (batch, seq_len, hidden_size)

        Returns
        -------
        context : torch.Tensor, shape (batch, hidden_size)
        weights : torch.Tensor, shape (batch, seq_len)
        """
        attn_scores = self.attention(rnn_output)     # (batch, seq, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (batch, seq, 1)
        context = torch.sum(rnn_output * attn_weights, dim=1)  # (batch, hidden)
        return context, attn_weights.squeeze(-1)


class HybridCNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid model for time-series activity recognition.

    Parameters
    ----------
    num_channels : int
        Number of input sensor channels.
    num_classes : int
        Number of activity classes.
    cnn_filters : list of int
        Number of filters in each CNN layer.
    lstm_hidden : int
        Hidden size for the LSTM.
    lstm_layers : int
        Number of LSTM layers.
    use_attention : bool
        Whether to use temporal attention over LSTM outputs.
    dropout : float
        Dropout rate.
    """

    def __init__(self, num_channels=None, num_classes=None,
                 cnn_filters=None, lstm_hidden=128, lstm_layers=2,
                 use_attention=True, dropout=None):
        super().__init__()

        num_channels = num_channels or config.NUM_CHANNELS
        num_classes = num_classes or config.NUM_CLASSES
        dropout = dropout or config.DROPOUT_RATE
        cnn_filters = cnn_filters or [64, 128]

        # ─── CNN Feature Extractor ─────────────────────────────────
        cnn_layers = []
        in_ch = num_channels
        for out_ch in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # ─── LSTM Temporal Encoder ─────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_output_size = lstm_hidden * 2  # bidirectional

        # ─── Attention ─────────────────────────────────────────────
        self.use_attention = use_attention
        if use_attention:
            self.attention = TemporalAttention(lstm_output_size)

        # ─── Classifier ───────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
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
        # CNN expects (batch, channels, length)
        x = x.permute(0, 2, 1)        # (batch, channels, window_size)
        x = self.cnn(x)                # (batch, cnn_filters[-1], T')

        # LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)        # (batch, T', cnn_filters[-1])
        rnn_out, _ = self.lstm(x)      # (batch, T', lstm_hidden*2)

        # Pooling over time
        if self.use_attention:
            x, _ = self.attention(rnn_out)  # (batch, lstm_hidden*2)
        else:
            x = rnn_out[:, -1, :]           # last time step

        x = self.dropout(x)
        x = self.classifier(x)

        return x


def get_model(**kwargs):
    """Factory function for HybridCNNLSTM."""
    return HybridCNNLSTM(**kwargs)


if __name__ == "__main__":
    for attn in [True, False]:
        model = HybridCNNLSTM(use_attention=attn)
        x = torch.randn(4, config.WINDOW_SIZE, config.NUM_CHANNELS)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        label = "with attention" if attn else "no attention"
        print(f"CNN-LSTM ({label}): Input {x.shape} → Output {out.shape}, "
              f"Params: {n_params:,}")
