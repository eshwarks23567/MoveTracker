"""
Domain adaptation techniques for bridging PAMAP2 → smartphone.

Handles differences between PAMAP2 IMUs (body-worn, 100 Hz) and
typical smartphone sensors (in-hand/pocket, 50 Hz) including:
- Resampling (100 Hz → 50 Hz)
- Channel subsetting (dual IMU → single device)
- Sensor axis remapping (device orientation compensation)
- Optional: Domain-Adversarial Neural Network (DANN)
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


# ─── Resampling ───────────────────────────────────────────────────────────────

def downsample_windows(X: np.ndarray, source_fs: float = None,
                       target_fs: float = None) -> np.ndarray:
    """
    Downsample windowed data to match smartphone sampling rate.

    Parameters
    ----------
    X : np.ndarray, shape (N, window_size, channels)
    source_fs : float
        Source sampling rate (default: 100 Hz).
    target_fs : float
        Target sampling rate (default: 50 Hz).

    Returns
    -------
    X_resampled : np.ndarray, shape (N, new_window_size, channels)
    """
    source_fs = source_fs or config.PAMAP2_SAMPLING_RATE
    target_fs = target_fs or config.PHONE_SAMPLING_RATE

    ratio = target_fs / source_fs
    new_length = int(X.shape[1] * ratio)

    N, T, C = X.shape
    X_resampled = np.zeros((N, new_length, C), dtype=np.float32)

    for i in range(N):
        for ch in range(C):
            X_resampled[i, :, ch] = resample(X[i, :, ch], new_length)

    return X_resampled


# ─── Channel Subsetting ──────────────────────────────────────────────────────

def subset_hand_only(X: np.ndarray) -> np.ndarray:
    """
    Keep only hand IMU channels (0-5: accel + gyro) to simulate
    phone-in-hand scenario.

    Parameters
    ----------
    X : np.ndarray, shape (N, window_size, 12)

    Returns
    -------
    X_hand : np.ndarray, shape (N, window_size, 6)
    """
    return X[:, :, :6]  # First 6 channels are hand accel + gyro


def subset_ankle_only(X: np.ndarray) -> np.ndarray:
    """
    Keep only ankle IMU channels (6-11: accel + gyro) to simulate
    phone-in-pocket scenario.

    Parameters
    ----------
    X : np.ndarray, shape (N, window_size, 12)

    Returns
    -------
    X_ankle : np.ndarray, shape (N, window_size, 6)
    """
    return X[:, :, 6:]  # Last 6 channels are ankle accel + gyro


# ─── Sensor Axis Remapping ────────────────────────────────────────────────────

def random_rotation_matrix():
    """Generate a random 3D rotation matrix (small angle)."""
    angles = np.random.randn(3) * 0.2  # Small random angles in radians

    alpha, beta, gamma = angles
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def apply_axis_remapping(X: np.ndarray, rotation_matrix=None) -> np.ndarray:
    """
    Apply a rotation to triaxial sensor data to simulate different
    device mounting orientations.

    Parameters
    ----------
    X : np.ndarray, shape (N, window_size, channels)
    rotation_matrix : np.ndarray, shape (3, 3), optional

    Returns
    -------
    X_rotated : np.ndarray, same shape as input
    """
    if rotation_matrix is None:
        rotation_matrix = random_rotation_matrix()

    X_rotated = X.copy()
    num_groups = X.shape[2] // 3

    for g in range(num_groups):
        start = g * 3
        end = start + 3
        if end <= X.shape[2]:
            # (N, T, 3) @ (3, 3) → (N, T, 3)
            X_rotated[:, :, start:end] = X[:, :, start:end] @ rotation_matrix.T

    return X_rotated


# ─── Add Realistic Noise ─────────────────────────────────────────────────────

def add_sensor_noise(X: np.ndarray, noise_level=0.02) -> np.ndarray:
    """
    Add noise that mimics smartphone sensor characteristics:
    - Gaussian measurement noise
    - Small bias drift
    """
    X_noisy = X.copy()

    # Gaussian noise
    noise = np.random.randn(*X.shape).astype(np.float32) * noise_level
    X_noisy += noise

    # Small bias drift (constant per window, per channel)
    bias = np.random.randn(X.shape[0], 1, X.shape[2]).astype(np.float32) * 0.01
    X_noisy += bias

    return X_noisy


# ─── Domain-Adversarial Neural Network (DANN) ────────────────────────────────

class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation."""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class DomainAdversarialNetwork(nn.Module):
    """
    DANN wrapper for any base HAR model.
    Adds a domain classifier with gradient reversal for
    unsupervised domain adaptation.

    Parameters
    ----------
    feature_extractor : nn.Module
        The CNN/RNN feature extractor (without final classifier).
    task_classifier : nn.Module
        Activity classification head.
    feature_dim : int
        Dimension of the feature representation.
    lambda_val : float
        Gradient reversal scaling factor.
    """

    def __init__(self, feature_extractor, task_classifier,
                 feature_dim=256, lambda_val=1.0):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier
        self.lambda_val = lambda_val

        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # source vs target domain
        )

    def forward(self, x, alpha=None):
        alpha = alpha or self.lambda_val
        features = self.feature_extractor(x)

        # Task prediction
        task_output = self.task_classifier(features)

        # Domain prediction with gradient reversal
        reversed_features = GradientReversal.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)

        return task_output, domain_output


# ─── Full Domain Adaptation Pipeline ─────────────────────────────────────────

def prepare_phone_simulation(X: np.ndarray, mode='hand',
                              add_noise=True, remap_axes=True,
                              downsample=True) -> np.ndarray:
    """
    Apply a full domain-adaptation pre-processing pipeline to simulate
    smartphone sensor data from PAMAP2 data.

    Parameters
    ----------
    X : np.ndarray, shape (N, window_size, 12)
    mode : str
        'hand' for phone in hand, 'ankle' for phone in pocket.
    add_noise : bool
    remap_axes : bool
    downsample : bool

    Returns
    -------
    X_adapted : np.ndarray
    """
    # Step 1: Subset channels
    if mode == 'hand':
        X_adapted = subset_hand_only(X)
    elif mode == 'ankle':
        X_adapted = subset_ankle_only(X)
    else:
        X_adapted = X.copy()

    # Step 2: Remap axes
    if remap_axes:
        X_adapted = apply_axis_remapping(X_adapted)

    # Step 3: Add noise
    if add_noise:
        X_adapted = add_sensor_noise(X_adapted)

    # Step 4: Downsample
    if downsample:
        X_adapted = downsample_windows(X_adapted)

    return X_adapted


if __name__ == "__main__":
    print("Testing domain adaptation pipeline...")

    # Dummy data: 10 windows, 256 samples, 12 channels
    X = np.random.randn(10, config.WINDOW_SIZE, config.NUM_CHANNELS).astype(np.float32)

    # Test individual operations
    X_hand = subset_hand_only(X)
    print(f"  Hand only: {X.shape} → {X_hand.shape}")

    X_ankle = subset_ankle_only(X)
    print(f"  Ankle only: {X.shape} → {X_ankle.shape}")

    X_resampled = downsample_windows(X)
    print(f"  Downsampled: {X.shape} → {X_resampled.shape}")

    X_rotated = apply_axis_remapping(X)
    print(f"  Axis remapped: {X.shape} → {X_rotated.shape}")

    X_noisy = add_sensor_noise(X)
    print(f"  Noisy: {X.shape} → {X_noisy.shape}")

    # Full pipeline
    X_phone = prepare_phone_simulation(X, mode='hand')
    print(f"\n  Full pipeline (hand): {X.shape} → {X_phone.shape}")

    print("\nAll domain adaptation tests passed! ✅")
