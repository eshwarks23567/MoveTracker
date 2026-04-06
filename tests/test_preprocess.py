"""
Unit tests for the preprocessing pipeline.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.data.preprocess import (
    handle_missing_values,
    apply_lowpass_filter,
    create_windows,
    compute_normalization_stats,
    normalize,
    map_labels,
)


class TestMissingValues:
    def test_short_gaps_interpolated(self):
        """Short NaN gaps should be filled via interpolation."""
        import pandas as pd

        data = {'col_3': [1.0, np.nan, np.nan, 4.0, 5.0]}
        df = pd.DataFrame(data)
        result = handle_missing_values(df, [3])
        assert not result['col_3'].isna().any()

    def test_long_gaps_dropped(self):
        """Gaps longer than MAX_INTERP_GAP should result in dropped rows."""
        import pandas as pd

        values = [1.0] + [np.nan] * 20 + [2.0]
        df = pd.DataFrame({'col_3': values})
        result = handle_missing_values(df, [3])
        assert len(result) < len(df)


class TestLowpassFilter:
    def test_output_shape(self):
        data = np.random.randn(1000, 12).astype(np.float32)
        filtered = apply_lowpass_filter(data)
        assert filtered.shape == data.shape

    def test_reduces_high_freq(self):
        """Filtered signal should have less high-frequency content."""
        t = np.linspace(0, 1, 1000)
        # Signal with 5 Hz + 40 Hz components
        signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 40 * t)
        data = signal.reshape(-1, 1).astype(np.float32)

        filtered = apply_lowpass_filter(data, cutoff=20, fs=100)

        # Check that high-freq component is attenuated
        fft_orig = np.abs(np.fft.fft(data[:, 0]))
        fft_filt = np.abs(np.fft.fft(filtered[:, 0]))

        # Energy above 30 Hz should be much lower after filtering
        high_freq_orig = np.sum(fft_orig[300:])
        high_freq_filt = np.sum(fft_filt[300:])
        assert high_freq_filt < high_freq_orig * 0.1


class TestWindowing:
    def test_correct_shape(self):
        T, C = 1000, 12
        data = np.random.randn(T, C).astype(np.float32)
        labels = np.ones(T, dtype=np.int64)

        X, y = create_windows(data, labels, window_size=256, step=128)

        expected_windows = (T - 256) // 128 + 1
        assert X.shape == (expected_windows, 256, C)
        assert y.shape == (expected_windows,)

    def test_majority_vote_label(self):
        """Window label should be the majority class in the window."""
        T = 256
        labels = np.array([1] * 200 + [2] * 56, dtype=np.int64)
        data = np.random.randn(T, 6).astype(np.float32)

        X, y = create_windows(data, labels, window_size=256, step=256)
        assert y[0] == 1  # Majority is class 1

    def test_empty_input(self):
        data = np.random.randn(100, 6).astype(np.float32)
        labels = np.ones(100, dtype=np.int64)
        X, y = create_windows(data, labels, window_size=256, step=128)
        assert X.shape[0] == 0  # Not enough data


class TestNormalization:
    def test_zero_mean_unit_var(self):
        X = np.random.randn(100, 256, 12).astype(np.float32) * 10 + 5
        means, stds = compute_normalization_stats(X)
        X_norm = normalize(X, means, stds)

        flat = X_norm.reshape(-1, 12)
        np.testing.assert_allclose(flat.mean(axis=0), 0, atol=0.01)
        np.testing.assert_allclose(flat.std(axis=0), 1, atol=0.01)


class TestLabelMapping:
    def test_contiguous_mapping(self):
        labels = np.array([1, 2, 3, 4, 5, 6, 12])
        mapped, valid = map_labels(labels)

        assert mapped.min() == 0
        assert mapped.max() == config.NUM_CLASSES - 1
        assert valid.all()

    def test_unknown_labels_invalid(self):
        labels = np.array([1, 99, 2])  # 99 is not a valid activity
        mapped, valid = map_labels(labels)
        assert not valid[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
