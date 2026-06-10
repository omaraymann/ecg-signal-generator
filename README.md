# ECG Signal Generator

A GAN-based system for generating synthetic ECG signals that preserve real heartbeat morphology while containing no patient-identifiable information.

## Overview

Built as part of **CE903 Group Project** at the University of Essex (Team 8), supervised by **Dr Delaram Jarchi**.

The system trains a Generative Adversarial Network on 8,244 ECG recordings from the [PhysioNet/CinC Challenge 2017 dataset](https://physionet.org/content/challenge-2017/) across three rhythm classes: **Normal**, **Atrial Fibrillation**, and **Other**.

## Architecture

- **Generator**: Bidirectional LSTM (BiLSTM) with 256 hidden units, Xavier/orthogonal weight initialisation, dropout 0.5, sigmoid output — produces synthetic ECG sequences of shape `(batch, 3000, 1)` from Gaussian noise `(batch, 3000, 5)`
- **Discriminator**: CNN with Conv1d layers (32→64→128→256 filters), LeakyReLU, global average pooling, BCEWithLogitsLoss — classifies real vs. generated ECG signals

## Preprocessing Pipeline

1. Load `.mat` files from PhysioNet 2017
2. Bandpass filter (4th-order Butterworth, 0.5–40 Hz) to remove baseline wander and noise
3. Per-recording min-max normalisation to `[0, 1]`
4. Segment into non-overlapping 3,000-sample windows (10 seconds at 300 Hz)
5. Recording-level 80/20 train-test split to prevent data leakage

**Output:** 21,021 training segments (`ecg_train.npy`) and 5,298 test segments (`ecg_test.npy`)

## Dataset

This project uses the **PhysioNet/CinC Challenge 2017** dataset:

> Clifford, G., Liu, C., Moody, B., Silva, I., Li, Q., Johnson, A., & Mark, R. (2017).
> AF Classification from a Short Single Lead ECG Recording.
> *Computing in Cardiology Conference (CinC)*, 44.
> https://physionet.org/content/challenge-2017/

**Download:** https://physionet.org/content/challenge-2017/1.0.0/

After downloading, place the `training2017/` folder in the project root and run:

```bash
python preprocess.py
```

This will generate `processed_data/ecg_train.npy` and `processed_data/ecg_test.npy`.

## Results

All three quantitative metrics passed acceptance thresholds on the held-out test set:

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| RMSE | 0.298 | < 0.30 | ✅ Pass |
| PRD | 63.89% | < 70% | ✅ Pass |
| Fréchet Distance | 0.358 | < 0.85 | ✅ Pass |

## Files

| File | Description |
|------|-------------|
| `preprocess.py` | Full preprocessing pipeline — filter, normalise, segment, train/test split |
| `final_code.ipynb` | Main GAN training notebook (256 hidden units, 50 epochs, best visual results) |
| `models/generator_epoch10.pth` | Generator checkpoint at epoch 10 |
| `models/generator_final.pth` | Final generator weights |
| `models/discriminator_epoch10.pth` | Discriminator checkpoint at epoch 10 |
| `models/discriminator_final.pth` | Final discriminator weights |

## Usage

```python
# Preprocess raw data and get a DataLoader
from preprocess import get_dataloader
dataloader = get_dataloader(data_dir='./training2017')
for batch in dataloader:
    real_data = batch[0]  # shape: (100, 3000, 1)
```

```python
# Load trained generator and generate synthetic ECG
import torch
# (define Generator class first — see final_code.ipynb)
generator = Generator(hidden_dim=256)
generator.load_state_dict(torch.load('models/generator_final.pth'))
generator.eval()

noise = torch.randn(1, 3000, 5)
with torch.no_grad():
    synthetic_ecg = generator(noise)  # shape: (1, 3000, 1)
```

## Dependencies

```
torch
numpy
scipy
matplotlib
scikit-learn
pandas
```

## Team

University of Essex — School of Computer Science and Electronic Engineering  
CE903 Group Project, Team 8 — March 2026

## Reference

Zhu et al., "Electrocardiogram generation with a bidirectional lstm-cnn generative adversarial network," *Scientific Reports*, vol. 9, 2019.
