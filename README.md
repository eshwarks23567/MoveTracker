# Human Activity Recognition Using Deep Learning on Wearable Sensor Data

> End-to-end pipeline: PAMAP2 sensor data → Preprocessing → Classical ML & Deep Learning → Mobile Deployment via TensorFlow Lite

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![TFLite](https://img.shields.io/badge/TFLite-Deployment-FF6F00.svg)](https://www.tensorflow.org/lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

<!-- ═══════════════════════════════════════════════════════════════════════════
     IEEE MAPPING GUIDE
     Use this to quickly find which README section maps to which IEEE section:

     README Section              →  IEEE Section
     ─────────────────────────      ─────────────────────
     1. Abstract                →  Abstract
     2. Introduction            →  I. Introduction
     3. Related Work            →  II. Related Work / Literature Review
     4. Dataset                 →  III-A. Dataset Description
     5. Data Preprocessing      →  III-B. Data Preprocessing
     6. Feature Engineering     →  III-C. Feature Engineering (Classical ML)
     7. Model Architectures     →  III-D. Deep Learning Architectures
     8. Data Augmentation       →  III-E. Data Augmentation
     9. Domain Adaptation       →  III-F. Domain Adaptation
     10. Experimental Setup     →  IV. Experimental Setup
     11. Results                →  V. Results & Discussion
     12. Mobile Deployment      →  VI. Deployment
     13. Conclusion             →  VII. Conclusion & Future Work
     14. References             →  References
     ═══════════════════════════════════════════════════════════════════════ -->

## 1. Abstract

This project presents a complete Human Activity Recognition (HAR) system that classifies seven physical activities—lying, sitting, standing, walking, running, cycling, and ascending stairs—from wearable inertial measurement unit (IMU) sensor data. The system implements the full machine learning lifecycle, from raw data ingestion and signal preprocessing through to model training, evaluation using Leave-One-Subject-Out (LOSO) cross-validation, and mobile deployment via TensorFlow Lite. Both classical machine learning baselines (Random Forest, SVM, XGBoost) and deep learning architectures (1D CNN, Bidirectional LSTM, Bidirectional GRU, and a CNN-LSTM Hybrid with temporal attention) are developed and compared. The current methodology strengthens posture-sensitive classes by adding orientation-aware features, class-weighted loss, and temporal majority-vote smoothing. A domain adaptation pipeline bridges the gap between laboratory-grade body-worn sensors and consumer smartphone hardware. The best-performing model is exported to TensorFlow Lite with INT8 quantization, targeting sub-20 ms inference and under 5 MB model size on mid-range smartphones.

---

## 2. Introduction

### 2.1 Problem Statement

Human Activity Recognition aims to automatically detect and classify the physical activity a person is performing, using data from motion sensors such as accelerometers and gyroscopes. Reliable HAR systems have applications in healthcare monitoring, elderly fall detection, fitness tracking, rehabilitation assessment, and context-aware computing.

### 2.2 Objectives

The primary objectives of this work are:

1. **Build a robust preprocessing pipeline** for the PAMAP2 dataset that handles missing values, removes high-frequency noise, segments continuous data into fixed-length windows, and augments posture-sensitive accelerometer channels with orientation cues.
2. **Develop and compare multiple model families** — classical ML models using handcrafted features and deep learning models that learn representations end-to-end from raw sensor signals.
3. **Evaluate generalization** using Leave-One-Subject-Out (LOSO) cross-validation, which tests each model's ability to recognize activities for unseen users.
4. **Deploy the best model to mobile** via a PyTorch → ONNX → TensorFlow Lite conversion pipeline, with quantization for efficient on-device inference.
5. **Investigate domain adaptation** to bridge the gap between body-worn research-grade IMUs and smartphone sensors.

### 2.3 Recognized Activities

The system classifies the following seven activity classes, selected from the 18 activities in the PAMAP2 protocol:

| Class Index | Activity         | PAMAP2 ID | Description               |
|:-----------:|------------------|:---------:|---------------------------|
| 0           | Lying            | 1         | Lying down / reclining    |
| 1           | Sitting          | 2         | Seated position           |
| 2           | Standing         | 3         | Stationary upright        |
| 3           | Walking          | 4         | Normal-pace locomotion    |
| 4           | Running          | 5         | Jogging / running         |
| 5           | Cycling          | 6         | Bicycle riding            |
| 6           | Ascending Stairs | 12        | Walking up stairs         |

---

## 3. Related Work

Sensor-based HAR has been extensively studied. Reiss and Stricker [1] introduced the PAMAP2 dataset with body-worn IMUs operating at 100 Hz. Ordóñez and Roggen [3] demonstrated that deep convolutional and LSTM recurrent networks can outperform traditional handcrafted-feature pipelines for multimodal wearable activity recognition. More recently, hybrid architectures combining CNNs for local pattern extraction with recurrent layers for temporal modelling, augmented with attention mechanisms, have achieved state-of-the-art performance on benchmark HAR datasets. This project draws on these advances while adding a full deployment pathway to mobile devices.

---

## 4. Dataset

### 4.1 PAMAP2 Physical Activity Monitoring

The PAMAP2 dataset [1][2] was collected from **9 subjects** wearing **3 Colibri wireless IMU sensors** mounted on the **wrist (hand)**, **chest**, and **ankle**. Each IMU records accelerometer (±16 g), gyroscope, magnetometer, and temperature data at **100 Hz** sampling rate. The dataset covers 18 activity types; this work selects **7 representative classes** (see Section 2.3).

### 4.2 Sensor Selection

From the 54-column PAMAP2 records, **12 channels** are selected:

| Source IMU | Modality          | Columns (0-indexed) | Channels Used |
|------------|-------------------|:-------------------:|:-------------:|
| Hand       | Accelerometer ±16 g | 3, 4, 5           | 3             |
| Hand       | Gyroscope          | 10, 11, 12         | 3             |
| Ankle      | Accelerometer ±16 g | 37, 38, 39        | 3             |
| Ankle      | Gyroscope          | 44, 45, 46         | 3             |
| **Total**  |                    |                     | **12**        |

The chest IMU and magnetometer/temperature modalities are excluded as they contribute less discriminative information for the selected activities while increasing dimensionality.

---

## 5. Data Preprocessing

The preprocessing pipeline transforms raw `.dat` files into normalized, windowed tensors suitable for model input. Each step is detailed below.

### 5.1 Missing Value Handling

Sensor recordings contain gaps due to wireless packet loss. Short gaps (≤ 10 consecutive samples) are filled using **linear interpolation**. Rows with remaining NaN values in any selected sensor column are dropped.

### 5.2 Signal Filtering

A **4th-order Butterworth low-pass filter** with a cutoff frequency of **20 Hz** is applied per channel to attenuate high-frequency measurement noise exceeding the bandwidth of human motion.

### 5.3 Sliding Window Segmentation

The continuous time-series is segmented into fixed-length windows using a sliding window approach:

| Parameter       | Value           | Rationale                                          |
|-----------------|:---------------:|----------------------------------------------------|
| Window size     | 256 samples     | Corresponds to 2.56 seconds at 100 Hz              |
| Overlap         | 50%             | Step of 128 samples; balances data volume and redundancy |
| Labelling       | Majority vote   | Each window inherits the most frequent activity label within it |

### 5.4 Orientation-Aware Feature Augmentation

To improve separation of static postures such as sitting and standing, the preprocessing pipeline appends gravity-oriented features to each accelerometer triad used by the deep models:

- `pitch = arctan2(ax, sqrt(ay^2 + az^2))`
- `roll = arctan2(ay, sqrt(ax^2 + az^2))`
- `acc_mag = sqrt(ax^2 + ay^2 + az^2)`

These features expose device orientation relative to gravity, which is often the main difference between sitting and standing even when the raw motion amplitude is low. With the two selected accelerometer triads, this adds 6 extra channels, increasing deep-model input from 12 to 18 channels.

### 5.5 Normalization

Per-channel **z-score normalization** (zero mean, unit variance) is applied. Statistics are computed across all windows to standardize the scale of accelerometer and gyroscope signals.

### 5.6 Pipeline Output

The preprocessed output consists of three NumPy arrays:
- `windows_X.npy` — shape `(N, 256, 18)`: sensor windows with orientation-aware features for deep learning; classical feature extraction still operates on the full window tensor
- `windows_y.npy` — shape `(N,)`: contiguous 0-indexed activity labels
- `subject_ids.npy` — shape `(N,)`: subject identifier per window (for LOSO splits)

---

## 6. Feature Engineering (Classical ML)

For classical machine learning models, a set of **~368 handcrafted features** is extracted from each sliding window across all 18 input channels. These features are grouped into four categories.

### 6.1 Time-Domain Features (10 per channel)

- Mean, standard deviation, minimum, maximum, median
- Root mean square (RMS)
- Range (peak-to-peak)
- Zero-crossing rate
- Mean absolute deviation
- Signal magnitude area (SMA)

### 6.2 Frequency-Domain Features (5 per channel)

- FFT energy (total power)
- Spectral entropy
- Dominant frequency
- Spectral centroid
- Spectral bandwidth

### 6.3 Statistical Features (5 per channel)

- Skewness, kurtosis
- Interquartile range (IQR)
- 25th and 75th percentiles

### 6.4 Cross-Channel Features (8 total)

- Pairwise correlations between hand accelerometer axes (3 pairs)
- Hand acceleration magnitude (mean and std)
- Ankle acceleration magnitude (mean and std)

### 6.5 Feature Count Summary

| Feature Group      | Per Channel | Channels | Subtotal |
|--------------------|:-----------:|:--------:|:--------:|
| Time-domain        | 10          | 18       | 180      |
| Frequency-domain   | 5           | 18       | 90       |
| Statistical        | 5           | 18       | 90       |
| Cross-channel      | —           | —        | 8        |
| **Total**          |             |          | **~368** |

---

## 7. Model Architectures

### 7.1 Classical Machine Learning Baselines

| Model          | Implementation  | Key Hyperparameters                            |
|----------------|:--------------:|------------------------------------------------|
| Random Forest  | scikit-learn    | Ensemble of decision trees; uses all ~368 features |
| SVM            | scikit-learn    | RBF kernel; regularization via C parameter      |
| XGBoost        | XGBoost library | Gradient-boosted trees; learning rate, max depth |

### 7.2 1D Convolutional Neural Network (CNN)

The CNN operates on raw sensor windows of shape `(256, 18)` and extracts local temporal patterns through successive convolutional layers.

**Architecture:**
```
Input (256 × 18)
  → Conv1D(18→64, k=7, pad=3) → BatchNorm → ReLU → MaxPool(2)
  → Conv1D(64→128, k=5, pad=2) → BatchNorm → ReLU → MaxPool(2)
  → Conv1D(128→256, k=3, pad=1) → BatchNorm → ReLU
  → Global Average Pooling
  → Dropout(0.3) → FC(256→128) → ReLU → Dropout(0.3) → FC(128→7)
```

Global average pooling replaces fully-connected flattening, reducing parameter count and overfitting risk.

### 7.3 Bidirectional LSTM / GRU

The recurrent models process the full 256-step sequence, capturing long-range temporal dependencies in both forward and backward directions.

**Architecture:**
```
Input (256 × 18)
  → BiLSTM(input=18, hidden=128, layers=2, dropout=0.3)
    OR BiGRU(input=18, hidden=128, layers=2, dropout=0.3)
  → Last time-step output (dim = 128 × 2 = 256)
  → Dropout(0.3) → FC(256→128) → ReLU → Dropout(0.3) → FC(128→7)
```

GRU uses fewer parameters than LSTM (no separate cell state), offering faster training with comparable performance.

### 7.4 CNN-LSTM Hybrid with Temporal Attention

The hybrid model combines the local feature extraction capability of CNNs with the sequential modelling of LSTMs. A **temporal attention mechanism** learns to weight the importance of different time steps.

**Architecture:**
```
Input (256 × 18)
  → Conv1D(18→64, k=5, pad=2) → BatchNorm → ReLU → MaxPool(2)
  → Conv1D(64→128, k=5, pad=2) → BatchNorm → ReLU → MaxPool(2)
  → BiLSTM(input=128, hidden=128, layers=2, dropout=0.3)
  → Temporal Attention (softmax-weighted sum over time steps)
  → Dropout(0.3) → FC(256→128) → ReLU → Dropout(0.3) → FC(128→7)
```

**Temporal Attention Mechanism:**
- Computes attention scores via a two-layer MLP: `Linear(256→128) → Tanh → Linear(128→1)`
- Applies softmax across the temporal dimension to produce normalized attention weights
- Outputs a weighted sum of LSTM hidden states as the context vector

This architecture is expected to be the best-performing model, as the CNN layers reduce the effective sequence length for the LSTM while preserving discriminative local features, and the attention mechanism focuses on the most informative segments of the activity window.

---

## 8. Data Augmentation

Six augmentation techniques are applied stochastically during training to improve generalization and reduce overfitting. Each is applied independently with a per-sample probability.

| # | Technique         | Description                                                   | Probability | Parameters        |
|:-:|-------------------|---------------------------------------------------------------|:-----------:|-------------------|
| 1 | Jitter            | Add Gaussian noise to simulate measurement imprecision        | 0.50        | σ = 0.05          |
| 2 | Scaling           | Random amplitude scaling to model gain variations             | 0.50        | Range [0.8, 1.2]  |
| 3 | 3D Rotation       | Random Euler rotation of triaxial groups for orientation invariance | 0.30   | Angles ~ N(0, 0.3)|
| 4 | Time Warping      | Smooth temporal stretch/compression via cubic spline warping  | 0.30        | Warp ratio ±10%   |
| 5 | Permutation       | Shuffle temporal segments to break strict ordering dependency | 0.30        | 5 segments         |
| 6 | Magnitude Warping | Sinusoidal envelope distortion for smooth amplitude variation | 0.30        | σ = 0.2           |

Augmentations operate on individual windows (shape `256 × 18`) and are composed through a `ComposeAugmentations` pipeline that applies each technique independently.

---

## 9. Domain Adaptation (PAMAP2 → Smartphone)

To bridge the domain gap between the PAMAP2 body-worn IMU setup and real-world smartphone usage, a four-step adaptation pipeline is implemented.

### 9.1 Frequency Resampling

PAMAP2 data recorded at 100 Hz is downsampled to **50 Hz** using SciPy's polyphase resampling to match typical smartphone sensor sampling rates. This reduces the window length from 256 to 128 samples.

### 9.2 Channel Subsetting

The 12-channel dual-IMU input is reduced to **6 channels** (accelerometer + gyroscope from a single sensor) to simulate a single handheld/pocketed device:
- **Hand mode**: channels 0–5 (hand accel + gyro) → phone in hand
- **Ankle mode**: channels 6–11 (ankle accel + gyro) → phone in pocket/leg

### 9.3 Axis Remapping

Random 3D rotations (small Euler angles ~ N(0, 0.2) radians) are applied to triaxial sensor groups to simulate arbitrary device orientations, promoting orientation-invariant representations.

### 9.4 Domain-Adversarial Neural Network (DANN)

For unsupervised adaptation, a **Domain-Adversarial Neural Network** with gradient reversal is implemented. The DANN adds a domain classifier branch that predicts whether input data comes from the source (PAMAP2) or target (smartphone-simulated) domain, while the gradient reversal layer forces the shared feature extractor to learn domain-invariant representations.

---

## 10. Experimental Setup

### 10.1 Evaluation Protocol

All models are evaluated using **Leave-One-Subject-Out (LOSO) cross-validation** with **9 folds**. In each fold, one subject's data is held out entirely for testing while the remaining 8 subjects' data is used for training. LOSO is the gold-standard protocol for HAR, as it assesses true inter-subject generalization.

### 10.2 Training Configuration

| Hyperparameter           | Value                                       |
|--------------------------|:-------------------------------------------:|
| Optimizer                | Adam (β₁=0.9, β₂=0.999)                    |
| Learning rate            | 1 × 10⁻³                                   |
| Weight decay             | 1 × 10⁻⁴                                   |
| Batch size               | 64                                          |
| Max epochs               | 50                                          |
| Early stopping patience  | 10 epochs (min Δ = 0.001)                   |
| LR scheduler             | ReduceLROnPlateau (factor=0.5, patience=5)  |
| Gradient clipping        | Max norm = 1.0 (for RNN stability)          |
| Dropout rate             | 0.3                                         |
| Loss function            | Cross-Entropy with inverse-frequency class weights, plus optional focal loss |
| Random seed              | 42                                          |

### 10.3 Class Imbalance Handling

Class weights are computed as the inverse of per-class frequency in the training split and passed to `CrossEntropyLoss`. Hard-to-separate posture classes receive a small additional boost so the learner does not overfit only the easiest dynamic activities. Focal loss can be enabled when the goal is to emphasize hard samples further.

### 10.4 Temporal Smoothing

To reduce flicker in predictions and improve stability across adjacent windows, the evaluation pipeline applies a centered majority vote over a short prediction window. This is especially useful for stationary activities, where individual windows can be noisy even when the underlying posture is stable.

### 10.5 Evaluation Metrics

The following metrics are reported per fold and aggregated (mean ± std across 9 LOSO folds):

- **Accuracy** — overall proportion of correct predictions
- **Weighted F1-score** — F1 averaged by class support
- **Macro F1-score** — unweighted average F1 across classes (sensitive to minority class performance)
- **Per-class precision and recall**
- **Confusion matrix** — to identify systematic misclassification patterns
- **Inference latency** and **model size** — for deployment feasibility

---

## 11. Results

The following metrics were computed using Leave-One-Subject-Out (LOSO) cross-validation across 4 folds (representing a subset of subjects).

### 11.1 Model Comparison (LOSO)

| Model          | Accuracy (mean ± std) | F1 Weighted (mean ± std) | F1 Macro (mean ± std) | Params   |
|----------------|:---------------------:|:------------------------:|:---------------------:|:--------:|
| Random Forest  | 0.730 ± 0.090         | 0.695 ± 0.120            | 0.618 ± 0.135         | —        |
| SVM            | 0.841 ± 0.052         | 0.839 ± 0.056            | 0.721 ± 0.117         | —        |
| XGBoost        | 0.726 ± 0.035         | 0.686 ± 0.035            | 0.606 ± 0.097         | —        |
| 1D CNN         | 0.958 ± 0.022         | 0.958 ± 0.022            | 0.853 ± 0.109         | ~180K    |
| BiLSTM         | 0.918 ± 0.032         | 0.916 ± 0.036            | 0.801 ± 0.108         | ~574K    |
| BiGRU          | —                     | —                        | —                     | —        |
| CNN-LSTM+Attn  | 0.940 ± 0.020         | 0.941 ± 0.020            | 0.837 ± 0.126         | ~772K    |

### 11.2 Per-Class Performance (Best Model: 1D CNN)

| Activity         | Precision | Recall | F1-score |
|------------------|:---------:|:------:|:--------:|
| Lying            | 0.994     | 0.950  | 0.971    |
| Sitting          | 0.948     | 0.942  | 0.945    |
| Standing         | 0.870     | 0.963  | 0.914    |
| Walking          | 0.980     | 0.977  | 0.978    |
| Running          | 1.000     | 0.909  | 0.952    |
| Cycling          | 0.998     | 0.967  | 0.982    |
| Ascending Stairs | 0.966     | 0.936  | 0.951    |

### 11.3 Discussion

- **Best Approach**: The deep learning models, particularly the 1D CNN (95.8% accuracy), significantly outperformed the classical ML baselines (~72-84% accuracy) across the board. The CNN's ability to extract local temporal patterns proved highly effective.
- **Stationary vs. Dynamic Activities**: Stationary activities like standing and sitting showed slight confusion due to their similar sensor signatures (F1 scores ~0.91-0.94). Dynamic activities like walking, cycling, and running achieved exceptionally high precision and recall (F1 ≥ 0.95).
- **Architecture Trade-offs**: The CNN-LSTM Hybrid model (94.0% accuracy) performed comparably to the 1D CNN but required >4x the parameters (~772K vs ~180K). For mobile deployment, the standalone 1D CNN offers the best accuracy-to-size trade-off.

---

## 12. Mobile Deployment

The mobile app runs the trained HAR model on-device and does not require a backend once the assets are bundled. Real phone sensors feed a rolling window buffer, which is then passed through the local inference runtime and lightweight post-processing rules.

### 12.1 Runtime Path

- DeviceMotion data is collected from the phone accelerometer and gyroscope.
- Sensor samples are smoothed and stored in a rolling window.
- The local model runs with ONNX Runtime Web or the bundled deployment runtime.
- Prediction smoothing and anti-spoof rules suppress false walking/lying transitions caused by noisy hand motion.

### 12.2 Export Pipeline

The trained PyTorch model is exported through a multi-stage conversion path when needed:

```text
PyTorch (.pt) → ONNX (.onnx) → TensorFlow SavedModel → TFLite (.tflite)
```

The ONNX export is stored under `results/`, and the mobile bundle includes the runtime assets required for local execution.

### 12.3 Quantization Targets

Post-training quantization is used to reduce model size and latency for on-device use:

| Quantization Mode | Size Reduction | Expected Accuracy Impact |
|-------------------|:--------------:|:------------------------:|
| None (FP32)       | Baseline       | —                        |
| Dynamic range     | ~2-3x          | Minimal                  |
| Float16           | ~2x            | Negligible               |
| Full INT8         | ~4x            | < 1% accuracy drop       |

### 12.4 Deployment Targets

| Metric            | Target                              |
|-------------------|:-----------------------------------:|
| Model size        | < 5 MB                              |
| Inference latency | < 20 ms on mid-range smartphone     |
| Runtime           | Android via Capacitor and local JS runtime |

### 12.5 Mobile App

The `mobile_app/` directory contains the browser-based phone UI used for live inference and validation. It supports:

- Real sensor mode using the browser `DeviceMotion` API
- Simulation mode for desktop browsers or denied sensor permissions
- A live activity card, per-class probability bars, and a running history log
- Toasts and a sensor badge to indicate whether real sensors, waiting, or simulation are active

### 12.6 Running the Mobile App

The app must be served over HTTPS for the `DeviceMotion` API to function on modern phones. A local HTTPS server is included that auto-generates a self-signed TLS certificate.

**Requirement:**
```bash
pip install cryptography
```

**Start the server:**
```bash
cd mobile_app
python serve.py
```

The server prints two URLs:
```text
Local:    https://localhost:5500
Network:  https://10.x.x.x:5500
```

**On your phone:**
1. Ensure the phone and PC are on the same Wi-Fi network.
2. Open the Network URL in the phone's browser.
3. Accept the self-signed certificate warning (tap Advanced → Proceed).
4. Tap Start. On Android, real sensors activate immediately. On iOS, tap "Enable Sensors" in the overlay that appears.

The sensor badge in the header displays `REAL` when live sensor data is being received, `WAIT` while the listener is initialising, and `SIM` when running in simulation mode.

**Custom port:**
```bash
python serve.py --port 9000
```

---

## 13. Conclusion and Future Work

### 13.1 Summary

This project demonstrates a complete HAR pipeline from raw PAMAP2 sensor data to mobile deployment. Multiple classical and deep learning approaches are implemented and evaluated under the rigorous LOSO cross-validation protocol, with the current pipeline emphasizing posture-aware features and class-balanced training for the hardest static classes.

### 13.2 Future Work

- **Hard-negative retraining**: Add more sitting, standing, and phone-shaking samples so the model learns the toughest posture boundaries directly from data.
- **Native mobile app**: Port the web prototype to a React Native or native Android/iOS application for access to background sensor collection, persistent logging, and app store distribution.
- **Transformer architectures**: Investigate self-attention-based models (e.g., HAR Transformer) for longer-range temporal modelling.
- **Federated learning**: Enable collaborative model improvement across devices without centralised data collection.
- **Additional activities**: Extend classification to the full 18-class PAMAP2 set or integrate activities from other datasets (WISDM, UCI-HAR).
- **Continuous learning**: Enable on-device fine-tuning to adapt to individual users' movement patterns.

---

## 14. References

1. Reiss, A., & Stricker, D. (2012). *Introducing a New Benchmarked Dataset for Activity Monitoring*. In Proceedings of the 16th International Symposium on Wearable Computers (ISWC), pp. 108–109.
2. [PAMAP2 Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
3. Ordóñez, F.J., & Roggen, D. (2016). *Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*. Sensors, 16(1), 115.
4. Ganin, Y., et al. (2016). *Domain-Adversarial Training of Neural Networks*. Journal of Machine Learning Research, 17(1), 2096–2030.

---

## Appendix A: Project Structure

```
Human Activity/
├── config.py                      # Central configuration (all hyperparameters)
├── requirements.txt               # Python dependencies
├── README.md                      # This document
├── src/
│   ├── data/
│   │   ├── download.py            # PAMAP2 dataset downloader
│   │   ├── preprocess.py          # Full preprocessing pipeline (Sec. 5)
│   │   └── dataset.py             # PyTorch Dataset + LOSO splits
│   ├── features/
│   │   └── extract.py             # 368 handcrafted features (Sec. 6)
│   ├── models/
│   │   ├── classical.py           # RF, SVM, XGBoost (Sec. 7.1)
│   │   ├── cnn.py                 # 1D CNN (Sec. 7.2)
│   │   ├── rnn.py                 # BiLSTM / BiGRU (Sec. 7.3)
│   │   └── hybrid.py              # CNN-LSTM + Attention (Sec. 7.4)
│   ├── training/
│   │   ├── train.py               # LOSO training loop (Sec. 10)
│   │   └── evaluate.py            # Metrics & visualization (Sec. 10.4)
│   ├── robustness/
│   │   ├── augmentation.py        # 6 augmentation techniques (Sec. 8)
│   │   └── domain_adapt.py        # PAMAP2 → smartphone adaptation (Sec. 9)
│   └── deploy/
│       ├── export_tflite.py       # TFLite conversion + quantization (Sec. 12)
│       └── export_onnx.py         # ONNX export
├── mobile_app/                    # Progressive web app (Sec. 12.4)
│   ├── index.html                 # App shell and layout
│   ├── app.js                     # Sensor handling, classifier, UI logic
│   ├── app.css                    # Styles
│   └── serve.py                   # Local HTTPS server (self-signed cert)
├── tests/                         # Unit tests
└── results/                       # Checkpoints, figures, exported models
```

---

## Appendix B: Quick Start

### B.1 Installation

```bash
git clone https://github.com/yourusername/human-activity-recognition.git
cd human-activity-recognition

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

### B.2 Download Dataset

```bash
python -m src.data.download
```

### B.3 Preprocess Data

```bash
python -m src.data.preprocess
# Output: data/processed/windows_X.npy  (N × 256 × 12)
```

### B.4 Train Models

```bash
python -m src.training.train --model cnn --epochs 50
python -m src.training.train --model lstm --epochs 50
python -m src.training.train --model hybrid --epochs 50
```

### B.5 Export for Mobile

```bash
python -m src.deploy.export_tflite --model hybrid --quantize dynamic
```

### B.6 Run the Mobile App

```bash
pip install cryptography        # one-time, if not already installed
cd mobile_app
python serve.py
# Open the printed Network URL on your phone
```

### B.7 Run Tests

```bash
python -m pytest tests/ -v
```

---

## Appendix C: Technologies Used

| Category          | Technology                                   |
|-------------------|----------------------------------------------|
| Language          | Python 3.10+                                 |
| Deep Learning     | PyTorch 2.0+                                 |
| Classical ML      | scikit-learn, XGBoost                        |
| Signal Processing | NumPy, SciPy (Butterworth filter, FFT, resampling) |
| Data Handling     | Pandas                                       |
| Mobile Inference  | TensorFlow Lite                              |
| Cross-Framework   | ONNX / ONNX Runtime                         |
| Visualization     | Matplotlib, Seaborn                          |

---

## License

This project is licensed under the MIT License.
