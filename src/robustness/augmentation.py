"""
Data augmentation techniques for robust HAR training.

All augmentations operate on torch tensors and can be composed
for on-the-fly augmentation during training.

Techniques:
- Jitter: Add Gaussian noise
- Scaling: Random amplitude scaling
- Rotation: Random 3D rotation of acceleration/gyroscope axes
- Time Warping: Stretch/compress temporal segments
- Permutation: Randomly shuffle temporal segments
- Magnitude Warping: Smooth sinusoidal magnitude distortion
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


class Jitter:
    """Add Gaussian noise to the signal."""

    def __init__(self, sigma=None):
        self.sigma = sigma or config.JITTER_SIGMA

    def __call__(self, x):
        """x: (window_size, channels)"""
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class Scaling:
    """Randomly scale signal amplitude."""

    def __init__(self, scale_range=None):
        self.scale_range = scale_range or config.SCALING_RANGE

    def __call__(self, x):
        factor = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1])
        return x * factor


class Rotation:
    """
    Apply random 3D rotation to triaxial sensor groups.
    Rotates groups of 3 channels (x, y, z) together to simulate
    different device orientations.
    """

    def __call__(self, x):
        """x: (window_size, channels)"""
        x = x.clone()

        # Generate random rotation matrix
        angles = torch.randn(3) * 0.3  # Small random angles
        R = self._rotation_matrix(angles[0], angles[1], angles[2])

        # Apply to each triaxial group (channels 0-2, 3-5, 6-8, 9-11)
        num_groups = x.shape[1] // 3
        for g in range(num_groups):
            start = g * 3
            end = start + 3
            if end <= x.shape[1]:
                x[:, start:end] = x[:, start:end] @ R.T

        return x

    @staticmethod
    def _rotation_matrix(alpha, beta, gamma):
        """Create a 3D rotation matrix from Euler angles."""
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        cb, sb = torch.cos(beta), torch.sin(beta)
        cg, sg = torch.cos(gamma), torch.sin(gamma)

        Rx = torch.tensor([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=torch.float32)
        Ry = torch.tensor([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=torch.float32)
        Rz = torch.tensor([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=torch.float32)

        return Rz @ Ry @ Rx


class TimeWarping:
    """
    Apply time warping by stretching/compressing random segments.
    Uses smooth cubic spline warping for natural-looking distortions.
    """

    def __init__(self, warp_ratio=None, num_knots=4):
        self.warp_ratio = warp_ratio or config.TIME_WARP_RATIO
        self.num_knots = num_knots

    def __call__(self, x):
        """x: (window_size, channels)"""
        T, C = x.shape

        # Create warping path
        orig_steps = torch.linspace(0, 1, T)

        # Random perturbation at knot points
        knot_positions = torch.linspace(0, 1, self.num_knots + 2)
        perturbations = torch.randn(self.num_knots + 2) * self.warp_ratio
        perturbations[0] = 0    # Fix start
        perturbations[-1] = 0   # Fix end

        warped_positions = knot_positions + perturbations
        warped_positions = torch.clamp(warped_positions, 0, 1)
        warped_positions, _ = torch.sort(warped_positions)

        # Interpolate to get new time indices
        warped_steps = torch.from_numpy(
            np.interp(
                orig_steps.numpy(),
                warped_positions.numpy(),
                knot_positions.numpy(),
            )
        ).float()

        # Map back to original indices
        new_indices = (warped_steps * (T - 1)).long().clamp(0, T - 1)

        return x[new_indices]


class Permutation:
    """Randomly permute temporal segments."""

    def __init__(self, num_segments=None):
        self.num_segments = num_segments or config.PERMUTATION_SEGMENTS

    def __call__(self, x):
        """x: (window_size, channels)"""
        T = x.shape[0]
        segment_len = T // self.num_segments

        # Split into segments
        segments = []
        for i in range(self.num_segments):
            start = i * segment_len
            end = start + segment_len if i < self.num_segments - 1 else T
            segments.append(x[start:end])

        # Shuffle segments
        indices = torch.randperm(len(segments))
        shuffled = [segments[i] for i in indices]

        return torch.cat(shuffled, dim=0)


class MagnitudeWarping:
    """
    Apply smooth magnitude distortion using sinusoidal function.
    Multiplies the signal by a smooth envelope to simulate
    varying signal strength.
    """

    def __init__(self, sigma=0.2, num_knots=4):
        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, x):
        """x: (window_size, channels)"""
        T = x.shape[0]

        # Create smooth warp curve
        t = torch.linspace(0, 2 * np.pi, T)
        freq = torch.empty(1).uniform_(0.5, 2.0).item()
        warp = 1.0 + torch.sin(t * freq) * self.sigma

        return x * warp.unsqueeze(1)


class ComposeAugmentations:
    """Compose multiple augmentations with individual probabilities."""

    def __init__(self, augmentations, probabilities=None):
        """
        Parameters
        ----------
        augmentations : list of callables
        probabilities : list of float, optional
            Probability of applying each augmentation. Default: 0.5 for each.
        """
        self.augmentations = augmentations
        self.probabilities = probabilities or [0.5] * len(augmentations)

    def __call__(self, x):
        for aug, prob in zip(self.augmentations, self.probabilities):
            if torch.rand(1).item() < prob:
                x = aug(x)
        return x


def get_default_augmentation():
    """Get the default augmentation pipeline for HAR training."""
    return ComposeAugmentations(
        augmentations=[
            Jitter(),
            Scaling(),
            Rotation(),
            TimeWarping(),
            Permutation(),
            MagnitudeWarping(),
        ],
        probabilities=[0.5, 0.5, 0.3, 0.3, 0.3, 0.3],
    )


if __name__ == "__main__":
    print("Testing augmentations...")
    x = torch.randn(config.WINDOW_SIZE, config.NUM_CHANNELS)

    augmentations = [
        ("Jitter", Jitter()),
        ("Scaling", Scaling()),
        ("Rotation", Rotation()),
        ("TimeWarping", TimeWarping()),
        ("Permutation", Permutation()),
        ("MagnitudeWarping", MagnitudeWarping()),
        ("Composed", get_default_augmentation()),
    ]

    for name, aug in augmentations:
        out = aug(x)
        assert out.shape == x.shape, f"{name}: shape mismatch {out.shape} vs {x.shape}"
        print(f"  ✅ {name:20s}: {x.shape} → {out.shape}")

    print("\nAll augmentations passed! ✅")
