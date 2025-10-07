# LSTM Forecasting for Financial Time Series

**Citadel Quant Challenge 2025 — Research Track**

---

## Overview

This project explores an **autoregressive decoder-only LSTM** for multivariate financial time series forecasting. The goal is to predict missing financial features (`Y1`, `Y2`) from known features (`A–N`) using temporal dependencies.

The model was developed for the *Citadel Quant Challenge 2025 (Research Track)* to test whether sequential deep learning can improve short-term financial prediction.

---

## Methodology

### Model

A **decoder-only LSTM** predicts the missing part of a feature vector at each time step:

* **Input:** [known features at time t, predicted (or true) missing features at t−1]
* **Output:** missing features at time t
* **Training:** teacher forcing (uses ground truth from previous step)
* **Inference:** autoregressive (feeds its own predictions)

### Data

* Train/test data in `./data/`, each with 14 known + 2 missing columns
* The `time` column is dropped
* Sequences split into windows of length **1000**, step size **100**
* Chronological split: 90% train, 10% validation

### Training

| Setting    | Value          |
| ---------- | -------------- |
| Hidden Dim | 32             |
| Loss       | MSE            |
| Optimizer  | Adam (lr=1e-3) |
| Epochs     | 20             |
| Batch Size | 32             |

The model minimizes MSE between predicted and true missing parts (`xb_missing[:,1:,:]`).

---

## Results

* **Training loss:** ↓ steadily from ~0.7 → 0.18
* **Validation loss:** stabilizes around **0.40**
* Model generalizes well with mild overfitting

Visualizations:

* Plotly loss curves (train vs. val)
* Time series plots for predicted vs. ground truth `Y1`, `Y2`

---

## Inference

After training, the model predicts `Y1`, `Y2` for the test data autoregressively and saves results as:

```
final_inference_submission.csv
```



