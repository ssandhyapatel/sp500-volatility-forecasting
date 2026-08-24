"""Model implementations for volatility forecasting."""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from config import (
    RANDOM_STATE, N_ESTIMATORS_RF, MAX_DEPTH_RF, MIN_SAMPLES_LEAF_RF,
    N_ESTIMATORS_XGB, MAX_DEPTH_XGB, LEARNING_RATE_XGB, SUBSAMPLE_XGB, COLSAMPLE_XGB,
    LSTM_SEQ_LEN, LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    LSTM_EPOCHS, LSTM_BATCH_SIZE
)


class NaiveModel:
    """Baseline: predict last observed volatility."""

    def predict(self, X, features_df):
        """Predict using vol_lag_1m as naive forecast."""
        vol_lag_idx = list(features_df.columns).index('vol_lag_1m')
        return X[:, vol_lag_idx]


class RidgeModel:
    """Ridge regression with regularization."""

    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha, random_state=RANDOM_STATE)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class RandomForestModel:
    """Random Forest regressor."""

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS_RF,
            max_depth=MAX_DEPTH_RF,
            min_samples_leaf=MIN_SAMPLES_LEAF_RF,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_


class XGBoostModel:
    """XGBoost regressor."""

    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS_XGB,
            max_depth=MAX_DEPTH_XGB,
            learning_rate=LEARNING_RATE_XGB,
            subsample=SUBSAMPLE_XGB,
            colsample_bytree=COLSAMPLE_XGB,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def feature_importances(self):
        return self.model.feature_importances_


class LSTMModel:
    """LSTM neural network for time series forecasting."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.seq_len = LSTM_SEQ_LEN

    def _create_sequences(self, X, y, seq_len):
        """Create sequences for LSTM input."""
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i-seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit(self, X, y, validation_split=0.1):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y, self.seq_len)

        # Build model
        n_features = X.shape[1]
        self.model = Sequential([
            LSTM(LSTM_UNITS_1, return_sequences=True, 
                 input_shape=(self.seq_len, n_features)),
            Dropout(DROPOUT_RATE),
            LSTM(LSTM_UNITS_2),
            Dropout(DROPOUT_RATE),
            Dense(DENSE_UNITS, activation='relu'),
            Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse')

        # Train
        self.model.fit(
            X_seq, y_seq,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=validation_split,
            verbose=0
        )
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)), self.seq_len)
        return self.model.predict(X_seq, verbose=0).flatten()
