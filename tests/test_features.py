"""
Unit tests for feature extraction.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.features.extract import (
    time_domain_features,
    frequency_domain_features,
    statistical_features,
    cross_channel_features,
    extract_window_features,
    extract_all_features,
    get_feature_names,
)


class TestTimeDomainFeatures:
    def test_returns_expected_keys(self):
        channel = np.random.randn(256).astype(np.float32)
        features = time_domain_features(channel)
        expected_keys = ['mean', 'std', 'min', 'max', 'median', 'rms',
                         'range', 'zero_crossing_rate', 'mean_abs_deviation',
                         'signal_magnitude_area']
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"

    def test_constant_signal(self):
        channel = np.ones(256).astype(np.float32) * 5.0
        features = time_domain_features(channel)
        assert features['mean'] == pytest.approx(5.0)
        assert features['std'] == pytest.approx(0.0)
        assert features['range'] == pytest.approx(0.0)


class TestFrequencyDomainFeatures:
    def test_returns_expected_keys(self):
        channel = np.random.randn(256).astype(np.float32)
        features = frequency_domain_features(channel)
        expected_keys = ['fft_energy', 'spectral_entropy', 'dominant_freq',
                         'spectral_centroid', 'spectral_bandwidth']
        for key in expected_keys:
            assert key in features

    def test_pure_sine(self):
        """Dominant frequency should match the sine wave frequency."""
        fs = 100
        t = np.arange(256) / fs
        freq = 10.0  # 10 Hz
        channel = np.sin(2 * np.pi * freq * t).astype(np.float32)
        features = frequency_domain_features(channel, fs=fs)
        # Dominant freq should be close to 10 Hz
        assert abs(features['dominant_freq'] - freq) < 1.0


class TestStatisticalFeatures:
    def test_returns_expected_keys(self):
        channel = np.random.randn(256).astype(np.float32)
        features = statistical_features(channel)
        expected_keys = ['skewness', 'kurtosis', 'iqr',
                         'percentile_25', 'percentile_75']
        for key in expected_keys:
            assert key in features


class TestWindowFeatures:
    def test_feature_vector_length(self):
        window = np.random.randn(config.WINDOW_SIZE, config.NUM_CHANNELS).astype(np.float32)
        features = extract_window_features(window)
        names = get_feature_names()
        assert len(features) == len(names)

    def test_no_nans(self):
        window = np.random.randn(config.WINDOW_SIZE, config.NUM_CHANNELS).astype(np.float32)
        features = extract_window_features(window)
        assert not np.any(np.isnan(features))


class TestBatchExtraction:
    def test_batch_shape(self):
        X = np.random.randn(5, config.WINDOW_SIZE, config.NUM_CHANNELS).astype(np.float32)
        features = extract_all_features(X)
        names = get_feature_names()
        assert features.shape == (5, len(names))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
