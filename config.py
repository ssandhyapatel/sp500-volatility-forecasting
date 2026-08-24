"""Configuration constants for S&P 500 Volatility Forecasting."""

# Data
DATA_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
START_DATE = "1950-01-01"

# Feature Engineering
VOL_WINDOW_SHORT = 6      # months for realized vol calculation
VOL_WINDOW_LONG = 12
TARGET_HORIZON = 3        # months ahead to forecast
VOL_LAGS = [1, 2, 3, 6, 12]
RETURN_LAGS = [1, 2, 3]

# Train/Test Split
TRAIN_CUTOFF = "2015-01-01"

# Models
RANDOM_STATE = 42
N_ESTIMATORS_RF = 200
MAX_DEPTH_RF = 8
MIN_SAMPLES_LEAF_RF = 5

N_ESTIMATORS_XGB = 200
MAX_DEPTH_XGB = 4
LEARNING_RATE_XGB = 0.05
SUBSAMPLE_XGB = 0.8
COLSAMPLE_XGB = 0.8

# LSTM
LSTM_SEQ_LEN = 6
LSTM_UNITS_1 = 32
LSTM_UNITS_2 = 16
DENSE_UNITS = 8
DROPOUT_RATE = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 16

# Ensemble
ENSEMBLE_VAL_RATIO = 0.2
