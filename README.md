# Industrial Machine Sound Anomaly Detection Dashboard

> **Acoustic Condition-Monitoring System for Industrial Predictive Maintenance**  
> Unsupervised acoustic anomaly detection using Section-Specific MFCC Convolutional Autoencoders on the MIMII DUE fan dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Dataset-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## 1. Overview

This project provides an industry-grade monitoring dashboard for non-contact acoustic condition monitoring of industrial ventilation and cooling fans. The system analyzes raw sound recordings, extracts 40-channel Mel-Frequency Cepstral Coefficients (MFCCs), and passes them through section-specific Convolutional Autoencoders to calculate reconstruction errors. When the reconstruction error exceeds the locked 95th-percentile validation threshold calibrated on normal operational sounds, an anomaly alert is triggered.

---

## 2. Key Features

- **Industrial UI/UX**: Premium dark SCADA-inspired interface with responsive telemetry cards, real-time alert banners, and Altair score-versus-threshold comparison meters.
- **Real-Time PyTorch Inference**: Direct integration with frozen section-specific CNN Autoencoder checkpoints with zero mock data.
- **Audio Diagnostics Studio**: Upload custom `.wav` files or select pre-verified benchmark samples covering normal and anomalous operational states across all sections.
- **Dual Signal Inspector**: Real-time rendering of 16 kHz time-domain waveforms and 40-coefficient MFCC spectral heatmaps.
- **Session Telemetry & Audit Log**: Live tracking of analyzed recordings, normal/anomalous proportions, and full CSV export capabilities.
- **Scientific Rigor & Transparency**: Full reporting of empirical benchmark results, bootstrap confidence intervals, and honest scientific limitations.

---

## 3. Quickstart & Launch Instructions

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Streamlit 1.30+
- Librosa & SoundFile

### Running the Dashboard
Activate the existing virtual environment and launch Streamlit:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
streamlit run app/dashboard.py
```

Alternatively, run directly with Python:
```bash
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

The application will launch at:
`http://localhost:8501`

---

## 4. Project Structure

```
industrial-machine-failure-detection/
├── app/
│   ├── inference.py            # Backend PyTorch inference engine
│   ├── dashboard.py            # Primary Streamlit dashboard entry point
│   ├── main.py                 # Convenience application launcher
│   ├── styles.py               # Dark industrial SCADA styling & CSS
│   ├── components/
│   │   ├── header.py           # Industrial branding & telemetry header
│   │   ├── cards.py            # KPI metrics cards & alert banners
│   │   └── visualizations.py   # Waveform, MFCC spectrogram & gauge charts
│   └── views/
│       ├── overview.py         # Live session telemetry & audit log
│       ├── audio_analysis.py   # Audio upload, playback & diagnostic studio
│       ├── model_info.py       # Architecture & methodology documentation
│       ├── research_performance.py # Empirical benchmarks & CI analysis
│       └── about.py            # MIMII DUE attribution & dataset hierarchy
├── data/
│   ├── raw/fan/                # MIMII DUE audio splits (train, source_test, target_test)
│   └── processed/              # Preprocessed spectrogram / MFCC feature tensors
├── models/
│   ├── mfcc_section_models/    # Section-specific MFCC CNN Autoencoder checkpoints
│   ├── isolation_forest/       # Baseline Isolation Forest models & scalers
│   └── section_models/         # Baseline Mel spectrogram checkpoints
├── reports/                    # Benchmark logs, evaluation CSVs, and figures
└── src/                        # Training, preprocessing & evaluation scripts
```

---

## 5. Machine Sections & Validation Thresholds

Thresholds are locked at the **95th percentile** of reconstruction errors computed strictly on the normal validation split (101 recordings per section) to enforce a nominal ~5% false alarm rate ($p = 0.95$):

| Machine Section | 95th-Percentile Threshold | Normal Val Count | Validation FPR | Target Test Recall | Target Test ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| `section_00` | `0.00347179` | 101 | 4.95% (5/101) | 8.00% | 0.7126 |
| `section_01` | `0.00448493` | 101 | 4.95% (5/101) | 3.00% | 0.5323 |
| `section_02` | `0.00319858` | 101 | 4.95% (5/101) | 6.00% | 0.5824 |
| **Target Macro Mean** | — | **303** | **4.95%** | **5.67%** | **0.6091** |

---

## 6. Research Performance Summary

As documented in `reports/FINAL_EXPERIMENT_ANALYSIS.md` and `reports/CONFIDENCE_INTERVAL_ANALYSIS.md`:

- **Target Macro ROC-AUC**: `0.6091` (95% Bootstrap CI: `0.5651` – `0.6543`)
- **Target Macro Recall (at P95 Threshold)**: `5.67%` (95% CI: `3.33%` – `8.33%`)
- **Target Macro Precision**: `49.81%` (95% CI: `32.26%` – `67.25%`)
- **Target Macro F1-Score**: `10.16%` (95% CI: `5.83%` – `14.48%`)
- **Target Macro Accuracy**: `50.17%` (95% CI: `48.33%` – `51.83%`)
- **Paired Bootstrap Difference vs. Isolation Forest**: `-0.0014` (95% CI: `[-0.0365, +0.0338]`, overlapping zero)

### Scientific Notice & Operational Scope
This system is an **experimental acoustic condition-monitoring prototype**, not a field-validated machine failure predictor. It has limited sensitivity (5.67% recall) due to mean-pooling dilution across 10-second recordings and acoustic domain shifts between factory environments. It should be used as part of a multi-modal condition monitoring strategy alongside vibration accelerometers and temperature sensors.

---

## 7. Dataset Attribution

Audio recordings are from the fan subset of **MIMII DUE** (Malfunctioning Industrial Machine Investigation and Examination under Domain Shifted Conditions):
- **Zenodo DOI**: [10.5281/zenodo.4740355](https://doi.org/10.5281/zenodo.4740355)
- **License**: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

