🧠 LSTM Forecasting for Financial Time Series

Citadel Quant Challenge 2025 — Research Track

📄 Overview

This project investigates the use of an autoregressive LSTM model for multivariate financial time series forecasting.
It was developed as part of the Research Track of the Citadel Quant Challenge 2025, with the objective of assessing the performance of sequential models on market data.

The core idea is to design a decoder-only LSTM capable of predicting the missing parameters of a financial feature vector over a future horizon, given the known parameters and historical information.

⚙️ Model Architecture

The model implements an autoregressive decoder-only LSTM:

At each timestep, the input consists of the known part of the vector and the previously predicted missing part.

The output is the missing part of the current vector.

During training, teacher forcing is used to stabilize learning.

During inference, the model predicts autoregressively, using its own past outputs.

🔧 Key Features

Sliding window mechanism (horizon, step) for time series segmentation

Chronological train/validation split (no temporal leakage)

Interactive training and validation loss curves using Plotly

Evaluation with R² metric on the validation set
