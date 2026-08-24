"""Feature engineering for volatility forecasting."""

import pandas as pd
import numpy as np
from config import VOL_LAGS, RETURN_LAGS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create full feature set from raw data.

    Features include:
    - Lagged volatility (1,2,3,6,12 months)
    - Lagged returns (1,2,3 months)
    - Rolling statistics (mean, max over 3/6 months)
    - Market valuation (CAPE, interest rate, dividend yield)
    - Cyclical calendar encodings (month sin/cos)

    Args:
        df: DataFrame with Date, realized_vol_6m, log_return, PE10, Long Interest Rate, Dividend, SP500

    Returns:
        DataFrame with all engineered features + target
    """
    features = pd.DataFrame(index=df.index)
    features['Date'] = df['Date']

    # Lagged volatility features
    for lag in VOL_LAGS:
        features[f'vol_lag_{lag}m'] = df['realized_vol_6m'].shift(lag)

    # Lagged return features
    for lag in RETURN_LAGS:
        features[f'return_lag_{lag}m'] = df['log_return'].shift(lag)

    # Rolling statistics on volatility
    features['vol_mean_3m'] = df['realized_vol_6m'].shift(1).rolling(3).mean()
    features['vol_mean_6m'] = df['realized_vol_6m'].shift(1).rolling(6).mean()
    features['vol_max_3m'] = df['realized_vol_6m'].shift(1).rolling(3).max()

    # Market valuation features
    features['pe10'] = df['PE10']
    features['interest_rate'] = df['Long Interest Rate']
    features['dividend_yield'] = df['Dividend'] / df['SP500']

    # Cyclical calendar features
    features['month'] = df['Date'].dt.month
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

    # Target
    features['target'] = df['target_vol_3m']

    # Drop rows with any NaN
    features_clean = features.dropna().copy()

    return features_clean


def get_feature_columns(features_df: pd.DataFrame) -> list:
    """Get list of feature column names (excluding Date and target).

    Args:
        features_df: DataFrame with engineered features

    Returns:
        List of feature column names
    """
    return [c for c in features_df.columns if c not in ['Date', 'target']]
