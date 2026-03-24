# MITM Research Implementation Platform

This repository contains the implementation pipeline for our research on Quantum-Classical Deep Learning for MITM Detection. It currently implements the Hybrid LSTM-CNN architecture for Man-in-the-Middle attack detection in IoT environments, replicating **Sinha et al., 2025** — *"A High Performance Hybrid LSTM-CNN Secure Architecture for IoT Environments Using Deep Learning"*.

## Quick Start (Google Colab)

```python
# 1. Clone/mount the repo, then:
from src.experiments.runner import run_all

# Run all experiments (LSTM-CNN + 5 baselines)
results = run_all()

# Or run a single model for quick testing
from src.experiments.runner import run_single
metrics = run_single("lstm_cnn", sample_frac=0.1)  # 10% data for speed
```

## Project Structure

```
Implementation/
├── config.yaml              # All hyperparameters (from Paper Tables 4 & 5)
├── requirements.txt         # Python dependencies
├── src/
│   ├── config.py            # Config loader
│   ├── data_processing/
│   │   ├── loader.py        # Kaggle download + CSV loading + splitting
│   │   └── preprocessor.py  # Clean, encode, scale, SMOTE, PCA
│   ├── models/
│   │   ├── base.py          # Abstract base class
│   │   ├── lstm_cnn.py      # Hybrid LSTM-CNN (main model)
│   │   └── baselines.py     # CNN, RNN, LSTM, BiLSTM, GRU
│   └── experiments/
│       ├── metrics.py       # Accuracy, Precision, Recall, F1, FPR, AUC-ROC
│       ├── visualization.py # Plots: confusion matrix, ROC, SHAP, etc.
│       └── runner.py        # Experiment orchestrator
├── data/raw/                # BoT-IoT CSV files (auto-downloaded via Kaggle API)
└── results/                 # Generated outputs (figures, tables, models)
```

## Models

| Model | Architecture | Paper |
|---|---|---|
| **LSTM-CNN (Hybrid)** | 3 LSTM(256) -> 3 Conv1D(64,128,256) -> Dense -> Sigmoid | Sinha et al., 2025 (Main) |
| CNN | 2 Conv1D layers | Baseline |
| RNN | 2 SimpleRNN layers | Baseline |
| LSTM | 2 LSTM layers | Baseline |
| BiLSTM | 2 Bidirectional LSTM layers | Baseline |
| GRU | 2 GRU layers | Baseline |

## Dataset

**BoT-IoT** (5% sample) — Cyber Range Lab, UNSW Canberra. Downloaded automatically via Kaggle API.

## Future Work

- Quantum ML integration (VQC, QAE via PennyLane)
- Multi-dataset evaluation (CIC-IDS2017, UNSW-NB15)
- Novel contribution for research originality
