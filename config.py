"""
Central configuration for the Human Activity Recognition project.
All paths, hyperparameters, sensor mappings, and label definitions live here.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
PAMAP2_SAMPLING_RATE = 100  # Hz
NUM_SUBJECTS = 9
SUBJECT_IDS = list(range(1, NUM_SUBJECTS + 1))

# Column layout in PAMAP2 .dat files (0-indexed)
# Col 0: timestamp, Col 1: activityID, Col 2: heart_rate
# Cols 3-19: IMU hand, Cols 20-36: IMU chest, Cols 37-53: IMU ankle
# Within each IMU block (17 cols):
#   0: temperature
#   1-3: accel ±16g   4-6: accel ±6g
#   7-9: gyroscope   10-12: magnetometer   13-16: orientation (invalid)

# Selected sensor columns (0-indexed in the full 54-column file)
# Hand  accel ±16g: cols 3,4,5    gyro: cols 10,11,12
# Ankle accel ±16g: cols 37,38,39  gyro: cols 44,45,46
HAND_ACCEL_COLS = [3, 4, 5]
HAND_GYRO_COLS = [10, 11, 12]
ANKLE_ACCEL_COLS = [37, 38, 39]
ANKLE_GYRO_COLS = [44, 45, 46]
SELECTED_SENSOR_COLS = HAND_ACCEL_COLS + HAND_GYRO_COLS + ANKLE_ACCEL_COLS + ANKLE_GYRO_COLS
NUM_CHANNELS = len(SELECTED_SENSOR_COLS)  # 12

TIMESTAMP_COL = 0
ACTIVITY_COL = 1
HEART_RATE_COL = 2

# ─── Activity Labels ─────────────────────────────────────────────────────────
# Full PAMAP2 activity map
FULL_ACTIVITY_MAP = {
    0: "other",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

# Subset used for this project (7 classes)
SELECTED_ACTIVITIES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    12: "ascending_stairs",
}

# Map original activity IDs → contiguous 0-based class indices
ACTIVITY_TO_IDX = {aid: idx for idx, aid in enumerate(sorted(SELECTED_ACTIVITIES.keys()))}
IDX_TO_ACTIVITY = {idx: SELECTED_ACTIVITIES[aid] for aid, idx in ACTIVITY_TO_IDX.items()}
NUM_CLASSES = len(SELECTED_ACTIVITIES)

# ─── Windowing ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 256          # samples (2.56 s at 100 Hz)
WINDOW_OVERLAP = 0.5       # 50% overlap
WINDOW_STEP = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))  # 128

# ─── Feature Engineering (for deep models) ───────────────────────────────────
# Add gravity-orientation cues to improve sitting vs standing separation.
ADD_ORIENTATION_FEATURES = True
# For each accelerometer triad (ax, ay, az): add [pitch, roll, acc_mag].
# Current setup uses 2 accel triads (hand, ankle), so +6 channels when enabled.
ORIENTATION_FEATURES_PER_TRIAD = 3

# ─── Preprocessing ────────────────────────────────────────────────────────────
MAX_INTERP_GAP = 10        # Max consecutive NaN samples to interpolate
LOWPASS_CUTOFF = 20        # Hz — Butterworth filter cutoff
LOWPASS_ORDER = 4          # Butterworth filter order
APPLY_LOWPASS = True       # Whether to apply low-pass filtering

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
DROPOUT_RATE = 0.3
RANDOM_SEED = 42

# Loss shaping for hard classes.
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 1.5
# Additional multiplier on top of inverse-frequency weights.
HARD_CLASS_WEIGHT_MULTIPLIER = {
    1: 1.10,  # sitting
    2: 1.10,  # standing
}

# Post-eval temporal smoothing by local majority vote.
MAJORITY_VOTE_WINDOW = 5

# ─── Augmentation ─────────────────────────────────────────────────────────────
JITTER_SIGMA = 0.05
SCALING_RANGE = (0.8, 1.2)
TIME_WARP_RATIO = 0.1
PERMUTATION_SEGMENTS = 5

# ─── Deployment ───────────────────────────────────────────────────────────────
TFLITE_MODEL_PATH = RESULTS_DIR / "model.tflite"
TARGET_INFERENCE_MS = 20   # Target max inference time on mobile (ms)
TARGET_MODEL_SIZE_MB = 5   # Target max model file size (MB)
PHONE_SAMPLING_RATE = 50   # Typical smartphone sensor rate (Hz)
