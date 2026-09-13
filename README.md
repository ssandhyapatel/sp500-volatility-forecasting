# S&P 500 Volatility Forecasting

> **Multi-model ensemble for predicting S&P 500 realized volatility 3 months ahead using Robert Shiller's historical data.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---


Most "stock prediction" projects try to forecast **price direction** — which is mostly noise. This project targets **realized volatility**, which is genuinely forecastable and directly relevant to:
- Portfolio risk management
- Options pricing
- VaR (Value at Risk) calculations
- Regime detection

Built with **real historical data** (1950–2026), not synthetic or toy datasets.

---

## Architecture

```
sp500-volatility-forecasting/
├── data/                    # Data directory (empty, fetched at runtime)
├── notebooks/               # Jupyter notebooks for exploration
├── results/                 # Output plots and metrics
├── src/                     # Source code modules
│   ├── data_loader.py       # Fetch & preprocess Shiller S&P 500 data
│   ├── features.py          # Feature engineering (17 features)
│   ├── models.py            # 5 model implementations
│   ├── ensemble.py          # Weighted & equal ensemble methods
│   ├── evaluation.py        # Metrics: RMSE, MAE, R², Directional Accuracy
│   └── visualization.py     # 4-panel result plots
├── app.py                   # FastAPI backend
├── streamlit_app.py         # Streamlit dashboard
├── Dockerfile               # Container for deployment
├── docker-compose.yml       # Multi-service orchestration
├── config.py                # Hyperparameters & constants
├── main.py                  # End-to-end training pipeline
└── requirements.txt         # Dependencies
```

---

## Models

| Model | Family | Role |
|---|---|---|
| **Naive** | Baseline | Last observed volatility |
| **Ridge** | Linear | Regularized linear regression |
| **Random Forest** | Tree Ensemble | Non-linear pattern capture |
| **XGBoost** | Gradient Boosting | Sequential error correction |
| **LSTM** | Deep Learning | Sequence modeling (6-month lookback) |
| **Weighted Ensemble** | Meta-Learner | Inverse-error-weighted combination |

---

## Features (17 total)

| Category | Features |
|---|---|
| **Lagged Volatility** | vol_lag_1m, vol_lag_2m, vol_lag_3m, vol_lag_6m, vol_lag_12m |
| **Lagged Returns** | return_lag_1m, return_lag_2m, return_lag_3m |
| **Rolling Statistics** | vol_mean_3m, vol_mean_6m, vol_max_3m |
| **Valuation** | pe10 (CAPE ratio), interest_rate, dividend_yield |
| **Calendar** | month_sin, month_cos |

---

## Results (Out-of-Sample: 2015–2026)

| Model | RMSE | MAE | R² | Directional Accuracy |
|---|---|---|---|---|
| Naive | 0.0836 | 0.0545 | -0.49 | 49.6% |
| Ridge | 0.0658 | 0.0451 | 0.07 | 50.4% |
| **Random Forest** | **0.0592** | **0.0378** | **0.25** | **62.8%** |
| XGBoost | 0.0646 | 0.0447 | 0.11 | 61.2% |
| LSTM | 0.0747 | 0.0520 | -0.19 | 37.2% |
| Ensemble (Equal) | 0.0655 | 0.0448 | 0.08 | 57.3% |
| **Ensemble (Weighted)** | **0.0628** | **0.0428** | **0.16** | **59.7%** |

**Key insight**: Random Forest achieves a **29% RMSE reduction** over the naive baseline and correctly predicts volatility direction **62.8%** of the time — including the COVID-19 crash spike in March 2020.

---

## Quick Start

### 1. Training Pipeline

```bash
# Clone repo
git clone https://github.com/yourusername/sp500-volatility-forecasting.git
cd sp500-volatility-forecasting

# Install dependencies
pip install -r requirements.txt

# Run full training pipeline
python main.py
```

### 2. Deploy API + Dashboard

```bash
# Start FastAPI backend
uvicorn app:app --reload

# In another terminal, start Streamlit dashboard
streamlit run streamlit_app.py
```

### 3. Docker Deployment

```bash
# Build and run both services
docker-compose up --build

# API available at: http://localhost:8000
# Dashboard at: http://localhost:8501
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Get volatility prediction |
| `/latest` | GET | Latest data point |
| `/history` | GET | Historical predictions |

### Example API Call

```python
import requests

payload = {
    "vol_lag_1m": 0.15,
    "vol_lag_2m": 0.14,
    "vol_lag_3m": 0.13,
    "vol_lag_6m": 0.12,
    "vol_lag_12m": 0.11,
    "return_lag_1m": 0.01,
    "return_lag_2m": 0.005,
    "return_lag_3m": -0.002,
    "vol_mean_3m": 0.14,
    "vol_mean_6m": 0.13,
    "vol_max_3m": 0.18,
    "pe10": 30.0,
    "interest_rate": 4.5,
    "dividend_yield": 0.015,
    "month_sin": 0.5,
    "month_cos": 0.866
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
# {"model": "ensemble", "predicted_volatility": 0.1423, "confidence": "Medium", "timestamp": "2026-08-20T..."}
```

---

## Dashboard Features

- **Live Prediction**: Input features and get real-time volatility forecast with confidence level
- **Historical Data**: Interactive plot of historical volatility with summary statistics
- **Model Info**: Architecture details, feature list, and performance metrics

---

## Key Design Decisions

### Temporal Split (No Data Leakage)
- **Train**: 1950–2014 (64 years)
- **Test**: 2015–2026 (11 years, fully out-of-sample)
- No random shuffling — time series order is preserved

### Realized Volatility Calculation
- Uses **6-month rolling standard deviation** of log returns
- Annualized by multiplying by √12
- Target: **3-month-ahead** realized volatility

### Ensemble Weighting
- Weights learned via **inverse validation MSE**
- XGBoost receives highest weight (~66%) due to lowest validation error
- Ridge receives lowest weight (~7%)

---

## Data Source

- **Robert Shiller S&P 500 Dataset**: [datasets/s-and-p-500](https://github.com/datasets/s-and-p-500)
- Monthly data from 1871–present including price, earnings, dividends, and interest rates

---



Copyright (c) 2026 Sandhya Patel. All Rights Reserved.

