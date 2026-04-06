"""
Feature extraction for classical machine learning.

Extracts time-domain, frequency-domain, and statistical features
from each channel of each sliding window.
"""

import sys
import numpy as np
from scipy import stats as sp_stats
from scipy.fft import fft, fftfreq
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


# ─── Time-Domain Features ────────────────────────────────────────────────────

def time_domain_features(channel: np.ndarray) -> dict:
    """
    Extract time-domain features from a single channel window.

    Parameters
    ----------
    channel : np.ndarray, shape (window_size,)

    Returns
    -------
    dict of feature_name → value
    """
    return {
        'mean': np.mean(channel),
        'std': np.std(channel),
        'min': np.min(channel),
        'max': np.max(channel),
        'median': np.median(channel),
        'rms': np.sqrt(np.mean(channel ** 2)),
        'range': np.ptp(channel),
        'zero_crossing_rate': np.sum(np.diff(np.sign(channel)) != 0) / len(channel),
        'mean_abs_deviation': np.mean(np.abs(channel - np.mean(channel))),
        'signal_magnitude_area': np.sum(np.abs(channel)) / len(channel),
    }


# ─── Frequency-Domain Features ───────────────────────────────────────────────

def frequency_domain_features(channel: np.ndarray,
                               fs: float = None) -> dict:
    """
    Extract frequency-domain features from a single channel window.

    Parameters
    ----------
    channel : np.ndarray, shape (window_size,)
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict of feature_name → value
    """
    fs = fs or config.PAMAP2_SAMPLING_RATE
    n = len(channel)

    # FFT
    fft_vals = fft(channel)
    fft_mag = np.abs(fft_vals[:n // 2])
    freqs = fftfreq(n, d=1.0 / fs)[:n // 2]

    # Power spectrum
    power = fft_mag ** 2
    total_power = np.sum(power) + 1e-10

    # Normalized power for entropy
    power_norm = power / total_power

    features = {
        'fft_energy': total_power,
        'spectral_entropy': -np.sum(power_norm * np.log2(power_norm + 1e-12)),
        'dominant_freq': freqs[np.argmax(fft_mag)] if len(fft_mag) > 0 else 0,
        'spectral_centroid': np.sum(freqs * power) / total_power,
        'spectral_bandwidth': np.sqrt(
            np.sum(((freqs - np.sum(freqs * power) / total_power) ** 2) * power) / total_power
        ),
    }

    return features


# ─── Statistical Features ────────────────────────────────────────────────────

def statistical_features(channel: np.ndarray) -> dict:
    """
    Extract statistical features from a single channel window.

    Parameters
    ----------
    channel : np.ndarray, shape (window_size,)

    Returns
    -------
    dict of feature_name → value
    """
    return {
        'skewness': float(sp_stats.skew(channel)),
        'kurtosis': float(sp_stats.kurtosis(channel)),
        'iqr': float(sp_stats.iqr(channel)),
        'percentile_25': np.percentile(channel, 25),
        'percentile_75': np.percentile(channel, 75),
    }


# ─── Cross-Channel Features ──────────────────────────────────────────────────

def cross_channel_features(window: np.ndarray) -> dict:
    """
    Extract features that relate multiple channels to each other.

    Parameters
    ----------
    window : np.ndarray, shape (window_size, num_channels)

    Returns
    -------
    dict of feature_name → value
    """
    features = {}

    # Correlation between accel axes (hand: channels 0,1,2)
    for i in range(min(3, window.shape[1])):
        for j in range(i + 1, min(3, window.shape[1])):
            corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
            features[f'corr_hand_accel_{i}_{j}'] = corr if not np.isnan(corr) else 0.0

    # Magnitude of acceleration (hand)
    if window.shape[1] >= 3:
        accel_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
        features['hand_accel_magnitude_mean'] = np.mean(accel_mag)
        features['hand_accel_magnitude_std'] = np.std(accel_mag)

    # Magnitude of acceleration (ankle) — channels 6,7,8
    if window.shape[1] >= 9:
        ankle_mag = np.sqrt(np.sum(window[:, 6:9] ** 2, axis=1))
        features['ankle_accel_magnitude_mean'] = np.mean(ankle_mag)
        features['ankle_accel_magnitude_std'] = np.std(ankle_mag)

    return features


# ─── Full Feature Extraction ─────────────────────────────────────────────────

def extract_window_features(window: np.ndarray, fs: float = None) -> np.ndarray:
    """
    Extract all features from a single window.

    Parameters
    ----------
    window : np.ndarray, shape (window_size, num_channels)
    fs : float, optional

    Returns
    -------
    np.ndarray, shape (num_features,)
    """
    all_features = {}

    # Per-channel features
    for ch in range(window.shape[1]):
        channel_data = window[:, ch]
        prefix = f"ch{ch}_"

        td = time_domain_features(channel_data)
        fd = frequency_domain_features(channel_data, fs)
        st = statistical_features(channel_data)

        for feat_dict in [td, fd, st]:
            for k, v in feat_dict.items():
                all_features[prefix + k] = v

    # Cross-channel features
    cc = cross_channel_features(window)
    all_features.update(cc)

    return np.array(list(all_features.values()), dtype=np.float32)


def extract_all_features(X: np.ndarray, fs: float = None) -> np.ndarray:
    """
    Extract features from all windows.

    Parameters
    ----------
    X : np.ndarray, shape (N, window_size, num_channels)

    Returns
    -------
    features : np.ndarray, shape (N, num_features)
    """
    from tqdm import tqdm

    features_list = []
    for i in tqdm(range(len(X)), desc="Extracting features"):
        feat = extract_window_features(X[i], fs)
        features_list.append(feat)

    return np.stack(features_list, axis=0)


def get_feature_names(num_channels: int = None) -> list:
    """Return the ordered list of feature names."""
    num_channels = num_channels or config.NUM_CHANNELS
    names = []

    # Per-channel features
    per_channel = (
        list(time_domain_features(np.zeros(10)).keys()) +
        list(frequency_domain_features(np.zeros(10)).keys()) +
        list(statistical_features(np.zeros(10)).keys())
    )

    for ch in range(num_channels):
        for feat_name in per_channel:
            names.append(f"ch{ch}_{feat_name}")

    # Cross-channel features
    cc = cross_channel_features(np.zeros((10, num_channels)))
    names.extend(cc.keys())

    return names


if __name__ == "__main__":
    # Quick test
    print("Testing feature extraction...")
    dummy = np.random.randn(config.WINDOW_SIZE, config.NUM_CHANNELS).astype(np.float32)
    features = extract_window_features(dummy)
    names = get_feature_names()
    print(f"  Features per window: {len(features)}")
    print(f"  Feature names count: {len(names)}")
    print(f"  Sample features: {names[:5]}")
