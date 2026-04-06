"""
Unit tests for model architectures.
"""

import sys
import torch
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


BATCH = 4
WINDOW = config.WINDOW_SIZE
CHANNELS = config.NUM_CHANNELS
CLASSES = config.NUM_CLASSES


class TestCNN1D:
    def test_output_shape(self):
        from src.models.cnn import CNN1D
        model = CNN1D()
        x = torch.randn(BATCH, WINDOW, CHANNELS)
        out = model(x)
        assert out.shape == (BATCH, CLASSES)

    def test_gradients_flow(self):
        from src.models.cnn import CNN1D
        model = CNN1D()
        x = torch.randn(BATCH, WINDOW, CHANNELS, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestLSTM:
    def test_output_shape(self):
        from src.models.rnn import LSTMModel
        model = LSTMModel()
        x = torch.randn(BATCH, WINDOW, CHANNELS)
        out = model(x)
        assert out.shape == (BATCH, CLASSES)

    def test_gru_output_shape(self):
        from src.models.rnn import GRUModel
        model = GRUModel()
        x = torch.randn(BATCH, WINDOW, CHANNELS)
        out = model(x)
        assert out.shape == (BATCH, CLASSES)


class TestHybrid:
    def test_output_shape_with_attention(self):
        from src.models.hybrid import HybridCNNLSTM
        model = HybridCNNLSTM(use_attention=True)
        x = torch.randn(BATCH, WINDOW, CHANNELS)
        out = model(x)
        assert out.shape == (BATCH, CLASSES)

    def test_output_shape_without_attention(self):
        from src.models.hybrid import HybridCNNLSTM
        model = HybridCNNLSTM(use_attention=False)
        x = torch.randn(BATCH, WINDOW, CHANNELS)
        out = model(x)
        assert out.shape == (BATCH, CLASSES)

    def test_parameter_count(self):
        from src.models.hybrid import HybridCNNLSTM
        model = HybridCNNLSTM()
        n_params = sum(p.numel() for p in model.parameters())
        # Should be reasonable for mobile deployment
        assert n_params < 10_000_000, f"Model too large: {n_params:,} params"


class TestModelComparison:
    """Ensure all models produce valid probability distributions."""

    @pytest.mark.parametrize("model_name", ['cnn', 'lstm', 'gru', 'hybrid'])
    def test_softmax_output(self, model_name):
        from src.training.train import get_model_by_name

        model = get_model_by_name(model_name)
        model.eval()

        x = torch.randn(BATCH, WINDOW, CHANNELS)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

        # Probabilities should sum to 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

        # All probabilities should be non-negative
        assert (probs >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
