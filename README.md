<div align="center">

# HRLAD
### Data-Driven Homotopic Reinforcement Learning for Time Series Anomaly Detection in Consumer Electronics

<p>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4a2b.svg" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Platform"></a>
  <a href="#"><img src="https://img.shields.io/badge/IEEE%20TCE-2026-00629b.svg" alt="IEEE TCE"></a>
</p>

<p><em>A unified framework integrating Markov-switching latent-mode modeling,
homotopy-based policy continuation, and data-driven online adaptation
for anomaly detection in consumer-electronics time series.</em></p>

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Methods](#-methods)
- [Configuration](#-configuration)
- [Evaluation Metrics](#-evaluation-metrics)
- [Reproducing Paper Results](#-reproducing-paper-results)
- [Deployment](#-deployment)
- [Citation](#-citation)
- [License](#-license)

---

## 🔭 Overview

**HRLAD** (Homotopic Reinforcement Learning for Anomaly Detection) addresses three
core challenges in consumer-electronics time-series anomaly detection:

1. **Non-stationary, multi-modal data** — devices switch among operating regimes
   (standby, active, degradation, fault), producing distribution shifts that
   defeat static detectors.
2. **Scarce anomaly labels** — anomalies are rare, and dense supervision is
   unavailable during deployment.
3. **Slow RL convergence** — standard reinforcement learning struggles with
   large state spaces and sparse rewards in the anomaly-detection setting.

HRLAD resolves these by coupling a **Markov-switching latent-mode model**
(capturing regime dynamics), a **homotopy-based policy continuation path**
(smoothly deforming a simple threshold detector into the optimal RL policy via a
$\lambda$-parameterized family of Bellman operators), and a **data-driven online
update mechanism** (incrementally refining the policy from streaming data with
forgetting-weighted LSTD recursion — no physical device model required).

---

## 🌟 Key Contributions

| # | Contribution | Module |
|---|---|---|
| 1 | **Markov-switching latent-mode model** — a discrete Markov chain with mode-dependent Gaussian emissions provides a unified framework for multi-modal time series. | `models/markov_jump.py` |
| 2 | **Homotopic RL detection framework** — a $\lambda$-parameterized path gradually transitions the policy from a threshold rule ($\lambda{=}0$) to the optimal RL strategy ($\lambda{=}1$), with provable $\gamma$-contraction and Lipschitz value continuity. | `models/homotopy_rl.py` |
| 3 | **Data-driven online update** — kernelized LSTD with a forgetting factor refines the policy from streaming data, requiring <8% label budget. | `models/homotopy_rl.py` (`online_update`) |
| 4 | **Comprehensive benchmark** — 12 methods (HRLAD + 11 baselines) evaluated on 3 real-device datasets with 10-seed statistical validation. | `run_experiment.py` |

---

## 📂 Repository Structure

```
code/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .gitignore
├── config.py                     # Central hyperparameter store (mirrors paper)
├── run_experiment.py             # Main experiment runner (train + test + stats)
│
├── configs/
│   └── default.yaml              # YAML configuration (editable hyperparameters)
│
├── models/
│   ├── __init__.py
│   ├── homotopy_rl.py            # ★ HRLAD: the proposed homotopic RL detector
│   ├── markov_jump.py            #   Markov-switching latent-mode model (EM, Viterbi)
│   ├── actor_critic.py           #   Deep actor-critic networks (DDPG-style)
│   ├── feature_extractor.py      #   Multi-domain sliding-window feature extraction
│   └── baselines.py              #   11 baseline detectors (see Methods below)
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py            # Dataset loading / NAB & Yahoo loaders / get_all_datasets
│   └── device_ports.py           # ★ Device acquisition ports (ADB / Modbus / SMBus) + mock replay
│
├── utils/
│   ├── __init__.py
│   └── metrics.py                # F1, AUROC, FAR, detection delay, Cohen's d
│
└── scripts/
    ├── train.py                  # Standalone training entry point
    └── evaluate.py               # Standalone evaluation entry point
```

---

## 🔧 Installation

### Prerequisites

- **Python** ≥ 3.10
- **pip** (or **conda**)

### Step 1 — Clone

```bash
git clone https://github.com/7WD1/Papercode.git
cd Papercode
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is optional — all baselines gracefully degrade to
> NumPy/scikit-learn fallbacks when PyTorch is unavailable. However, the full
> HRLAD actor-critic and deep baselines (LSTM-AE, VAE-AD, Anomaly Transformer,
> etc.) require PyTorch.

---

## 🚀 Quick Start

### Run the full experiment (all methods × all datasets × 10 seeds)

```bash
python run_experiment.py --output-dir results/
```

### Quick smoke test (1 seed, SPSD only)

```bash
python run_experiment.py --quick
```

### Train HRLAD only

```bash
python scripts/train.py --method HRLAD --dataset SPSD --output-dir checkpoints/
```

### Evaluate a trained model

```bash
python scripts/evaluate.py --method HRLAD --dataset BMSD --checkpoint checkpoints/hrlad_bmsd.pkl
```

---

## 📊 Datasets

HRLAD is evaluated on three real-device testbed datasets collected under
controlled fault injection. The statistics below are identical to those reported
in the paper (Section V-A and the consolidated dataset table).

### Consolidated statistics

| Item | SPSD | HPMD | BMSD |
|------|:----:|:----:|:----:|
| Domain | Smartphone sensors (accel. + gyro) | Home appliance power | Battery management (V / I / T) |
| Sampling rate | 50 Hz | 1 Hz | 10 Hz |
| #input channels | 6 | 3 | 3 |
| Devices | 3 phones | 3 appliances | 1 battery pack |
| Anomaly types | 4 | 3 | 3 |
| #anomaly events | 1,584 | 1,276 | 1,142 |
| Total samples (effective) | 4.32M | 5.18M | 3.89M |
| Train samples | 3.02M | 3.63M | 2.72M |
| Validation samples | 0.65M | 0.78M | 0.58M |
| Test samples | 0.65M | 0.78M | 0.58M |
| Anomaly ratio (test) | 3.7% | 2.9% | 4.2% |
| Monitoring duration | 30 days | 60 days | 45 days |

> **Effective samples** exclude device idle periods, scheduled maintenance
> shutdowns, and data-quality exclusions (e.g., sensor dropout intervals). The
> "duration" refers to the total monitoring campaign including such inactive
> periods. **Splitting** is chronological 70% / 15% / 15% (train / validation /
> test), preventing information leakage from future observations. All datasets
> adopt event-level annotations labeled from onset to resolution by domain
> experts (inter-rater agreement Cohen's $\kappa > 0.92$), with ambiguous
> segments resolved through consensus review.

### Dataset details

**SPSD — Smartphone Sensor Dataset.**
Collected from mainstream smartphones running Android 13–14: a **Samsung Galaxy
S23**, a **Xiaomi 13**, and a **OnePlus 11**. Records accelerometer and gyroscope
readings at 50 Hz over 30 days. Four anomaly types, generated by controlled
fault injection (e.g., deliberate free-fall, gyroscope bias calibration failure)
and verified by two domain experts (Cohen's $\kappa > 0.92$):
1. Free-fall detection failure
2. Gyroscope drift
3. Accelerometer noise surge
4. Sensor fusion anomaly

**HPMD — Home Appliance Power Monitoring Dataset.**
Monitors power consumption of three appliances in a controlled test apartment:
an **LG GR-B247 refrigerator**, a **Daikin FTXTA35 air conditioner**, and an
**LG WD-T14410 washing machine**. Records at 1 Hz over 60 days. Three anomaly
types, injected via hardware faults (compressor relay cycling, inverter PWM
misconfiguration, motor phase loss) and annotated by the engineering team:
1. Compressor start failure
2. Inverter abnormal oscillation
3. Motor stalling

**BMSD — Battery Management System Dataset.**
Monitors voltage, current, and temperature of a **4S2P lithium-polymer pack
(14.8 V, 5000 mAh)** in a temperature-controlled chamber. Records at 10 Hz over
45 days. Three anomaly types, including controlled overcharge (4.35 V/cell),
thermal runaway precursors detected by embedded thermocouples, and accelerated
aging via cycle-life testing:
1. Overcharge / over-discharge
2. Thermal runaway precursor
3. Internal resistance degradation

### Injection protocol realism

The injection protocol was designed to reproduce the characteristics of real
consumer-electronics anomalies rather than idealized step faults:
- **Gradual onset** — fault magnitude is ramped over 200–800 ms rather than
  applied instantaneously, so the onset is masked by normal workload fluctuations.
- **Sensor-noise-floor injection** — noise is added at each device's measured
  sensor noise floor, so injected anomalies are not trivially separable.
- **User-behavior confounds** — injected events are interleaved with legitimate
  mode transitions (e.g., a smartphone entering gaming mode while a sensor fault
  is active).
- **Partial observability** — one input channel is occluded during 15% of the
  injected events.

Under this protocol, static detectors (LSTM-AE, OCSVM) achieve F1 below 83% on
the same data, confirming that the detection difficulty is comparable to field
conditions while the controlled setting preserves reproducibility.

### Data acquisition interface (`data/device_ports.py`)

Samples are obtained from the **paper-declared physical devices** through a
dedicated device-port layer rather than a free-form synthetic generator. Each
port targets the real hardware interface of the corresponding testbed and falls
back to a deterministic *mock replay* when the hardware (or its SDK) is absent,
so the public repository is runnable out-of-the-box.

| Dataset | Device port class | Real interface | Paper-declared devices |
|---------|-------------------|----------------|------------------------|
| **SPSD** | `SPSDDevicePort` | Android `SensorManager` over ADB (`adb-shell`) | Samsung Galaxy S23, Xiaomi 13, OnePlus 11 |
| **HPMD** | `HPMDDevicePort` | Modbus/TCP power meters (`pymodbus`) | LG GR-B247 refrigerator, Daikin FTXTA35 AC, LG WD-T14410 washer |
| **BMSD** | `BMSDDevicePort` | SMBus/I2C Smart Battery controller (`smbus2`) | 4S2P lithium-polymer pack (14.8 V, 5000 mAh) |

Every port exposes the same interface: `connect()` / `read_sample()` /
`acquire(n_samples)` / `selected_channel(values)` / `close()`, and supports the
context-manager protocol.

```python
from data.device_ports import open_port

# Mock replay (default) — no hardware needed, deterministic, CI-friendly
with open_port('SPSD', n_samples=2000, seed=42, use_mock=True) as port:
    Y, labels = port.acquire(n_samples=2000)        # Y: (2000, 6) multivariate
    accel_x = port.selected_channel(Y)              # (2000,) paper L572 selection

# Real device — requires the SDK + hardware; raises DeviceUnavailable if absent
with open_port('BMSD', use_mock=False) as port:     # needs `pip install smbus2`
    sample = port.read_sample()                      # (3,) one live V/I/T reading
```

The heavy SDK dependencies (`adb-shell`, `pymodbus`, `smbus2`) are imported
**lazily** inside the real-acquisition path, so they are never required for
`use_mock=True`. If the SDK is missing or the device is unreachable, the port
raises `DeviceUnavailable` with an actionable message.

### Channel selection (paper Section IV-A, L572)

The detector runs on a single validation-selected scalar channel per dataset,
exactly as specified in the manuscript:

| Dataset | Multivariate stream | Selected channel |
|---------|:---:|:---:|
| SPSD | 6 ch (accel xyz + gyro xyz) @ 50 Hz | `accel_x` |
| HPMD | 3 ch (appliance power) @ 1 Hz | `refrigerator_power` (active power) |
| BMSD | 3 ch (V / I / T) @ 10 Hz | `cell_voltage` |

`get_all_datasets()` returns this selected 1-D channel for each dataset,
preserving the contract consumed by `FeatureExtractor` and `HRLADDetector`.

### Data loaders

The `data/data_loader.py` module additionally provides:
- `generate_synthetic_ce_data()` — legacy 1-D synthetic waveform generator
  (retained for backward compatibility; superseded by the device ports above).
- `load_nab_data()` / `load_yahoo_s5()` — loaders for the public NAB and Yahoo S5
  benchmarks (if local data directories are provided).
- `get_all_datasets()` — aggregator returning all configured datasets via the
  device-port interface (mock replay by default).

> **Data availability:** The three datasets (SPSD, HPMD, BMSD), the full
> preprocessing scripts, the train/validation/test split indices, and the
> hyperparameter configuration files will be released in this repository upon
> paper acceptance. **Representative data samples (10 normal and 10 anomalous
> windows per dataset, in CSV format) are provided as supplementary material in
> the [`revise/`](revise/) folder** — see [`revise/README.md`](revise/README.md)
> for the per-dataset layout and column schema.
> The fixed random seeds are `{42, 123, 456, 789, 2024, 314, 271, 1618, 999,
> 2048}`, and all reported results are averaged over these 10 independent runs.
> Preprocessing follows Z-score normalization, windowing ($w = 64$, stride 1),
> and FFT feature extraction ($m = 4$ bins).

---

## 🧪 Methods

### Proposed Method: HRLAD

The HRLAD detector (`models/homotopy_rl.py` → `HRLADDetector`) implements the
complete pipeline:

```
Raw signal y_t
    │
    ▼
┌─────────────────────┐
│  Feature Extraction │  ← Sliding window (w=64) → 4 time-domain + 4 FFT = dim 8
│  (Z-score norm)     │
└────────┬────────────┘
         │ x_t
         ▼
┌─────────────────────┐
│  Markov-Switching   │  ← EM estimates transition matrix Π and mode emissions
│  Latent-Mode Model  │     Viterbi decodes latent mode sequence s_t
└────────┬────────────┘
         │ Π̂, ŝ_t
         ▼
┌─────────────────────┐
│  Homotopic RL       │  ← λ: 0 → 1   (affine reward interpolation)
│  Detection Framework│     R_λ = (1-λ)R₀ + λR
│                     │     Policy iteration with warm-start along λ-path
│  • LSTD evaluation  │     Kernelized LSTD with RBF features
│  • Actor-Critic     │     DDPG-style deep refinement (optional)
└────────┬────────────┘
         │ π_λ
         ▼
┌─────────────────────┐
│  Online Adaptive    │  ← Forgetting-weighted LSTD (ζ=0.97, N_eff≈33)
│  Detection          │     Delayed-feedback label budget < 8% of test set
└─────────────────────┘
         │
         ▼
   Anomaly decision: {Normal, Warning, Alarm}
```

### Baselines (11 methods)

All baselines are implemented in `models/baselines.py` with a unified
`fit_predict(values, labels)` interface:

| Method | Type | Key Idea |
|--------|------|----------|
| ARIMA | Statistical | Residual-based thresholding on ARIMA forecasts |
| Isolation Forest | Classical ML | Tree-based anomaly isolation |
| LSTM-AE | Deep AE | Reconstruction error from LSTM autoencoder |
| VAE-AD | Deep AE | Variational autoencoder reconstruction |
| USAD | Deep AE | Dual-encoder adversarial autoencoder |
| Anomaly Transformer | Transformer | Prior-association discrepancy |
| TimesNet | Transformer | 2D-transformed temporal forecasting residual |
| PatchTST-AD | Transformer | Patch-level forecasting error |
| DCdetector | Transformer | Dual-attention contrastive detection |
| MAUT (MemST) | Memory | Memory-augmented U-Transformer |
| iTransformer-AD | Transformer | Inverted-transformer forecasting error |
| TranAD (CARD) | Transformer | Self-conditioning adversarial reconstruction |
| RL-AD | Reinforcement Learning | DQN-based binary detection (Q-margin) |

---

## ⚙️ Configuration

Hyperparameters are managed in two ways:

1. **`config.py`** — a Python class with all defaults (mirrors paper Table III).
2. **`configs/default.yaml`** — an editable YAML file for experiment customization.

Key hyperparameters (as reported in the paper):

| Parameter | Symbol | Value |
|-----------|:---:|:---:|
| Window length | $w$ | 64 |
| Feature dimension | $d$ | 8 |
| Discount factor | $\gamma$ | 0.95 |
| Homotopy step | $\Delta\lambda$ | 0.15 |
| Forgetting factor | $\zeta$ | 0.97 ($N_\text{eff} \approx 33$) |
| Actor learning rate | — | $3 \times 10^{-4}$ |
| Critic learning rate | — | $10^{-3}$ |
| Replay buffer size | — | 50,000 |
| Polyak coefficient | $\tau$ | 0.005 |
| ε-greedy schedule | — | 1.0 → 0.01 over 200 episodes |
| Hidden layers | — | [256, 128, 64] |
| Number of seeds | — | 10 |

---

## 📈 Evaluation Metrics

All methods are evaluated on:

| Metric | Description |
|--------|-------------|
| **F1** | Harmonic mean of precision and recall (pointwise) |
| **AUROC** | Area under the ROC curve |
| **FAR** | False alarm rate (false positives / total negatives) |
| **DD** | Detection delay (steps from event onset to first correct alarm) |
| **Cohen's *d*** | Effect size between HRLAD and each baseline |
| **Bootstrap 95% CI** | 10,000-resample confidence intervals |
| **Paired *t*-test** | Statistical significance ($\alpha = 0.05$) |
| **Wilcoxon signed-rank** | Non-parametric significance test |
| **Holm–Bonferroni** | Multiple-comparison correction |

Reported main results (F1 %):

| Method | SPSD | HPMD | BMSD |
|--------|:---:|:---:|:---:|
| **HRLAD (Proposed)** | **93.5** | **91.2** | **92.1** |
| TranAD | 88.3 | 85.9 | 87.2 |
| iTransformer-AD | 87.9 | 85.5 | 86.8 |
| RL-AD | 87.8 | 85.2 | 86.5 |

---

## 🔁 Reproducing Paper Results

### Full experiment (≈ 4 hours on RTX 4090)

```bash
python run_experiment.py \
    --seeds 42 123 456 789 2024 314 271 1618 999 2048 \
    --datasets SPSD HPMD BMSD \
    --methods HRLAD LSTM-AE VAE-AD USAD Anomaly-Transformer \
              TimesNet PatchTST-AD DCdetector MemST iTransformer-AD \
              CARD RL-AD \
    --output-dir results/
```

### Generate all figures

The figure-generation scripts are located in the `experiment/` directory of the
main project workspace. To reproduce all figures, run from the project root:

```bash
python experiment/generate_all_figures.py
```

---

## 🖥️ Deployment

HRLAD is designed for resource-constrained consumer-electronics devices.
Reported deployment costs:

| Platform | Latency (ms/sample) | Memory (MB) | Energy (mJ/1k samples) |
|----------|:---:|:---:|:---:|
| RTX 4090 (GPU) | 0.048 | 1,800 | — |
| i9-13900K (CPU) | 1.92 | 642 | — |
| Jetson Nano (edge, FP16) | 1.85 | 410 | 1,860 |
| Raspberry Pi 4 (edge) | 6.74 | 386 | 5,230 |

> All platforms satisfy the real-time budget at their respective data rates
> (50 Hz / 10 Hz / 1 Hz). Model size: **245K parameters**.

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{jiang2026hrlad,
  title   = {Data-Driven Homotopic Reinforcement Learning for Time Series
             Anomaly Detection in Consumer Electronics},
  author  = {Jiang, Wen-Dong and Chang, Chih-Yung and Huang, Tzu-Chia
             and Gao, Shu-Jian and Wang, Chong and Roy, Diptendu Sinha},
  journal = {IEEE Transactions on Consumer Electronics},
  year    = {2026},
  note    = {Manuscript ID: TCE-2026-04-1642}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the
[LICENSE](LICENSE) file for details.

---

<div align="center">

<p><strong>Repository:</strong> <a href="https://github.com/7WD1/Papercode">https://github.com/7WD1/Papercode</a></p>

<p><em>For questions, please contact: wendongjiang@ieee.org</em></p>

</div>
