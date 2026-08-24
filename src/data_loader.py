"""Data loading and preprocessing for S&P 500 volatility forecasting."""

import pandas as pd
import numpy as np
from config import DATA_URL, START_DATE


def load_shiller_data() -> pd.DataFrame:
    """Load Robert Shiller S&P 500 dataset from public URL.

    Returns:
        DataFrame with columns: Date, SP500, PE10, Long Interest Rate, Dividend
    """
    df = pd.read_csv(DATA_URL)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df[df['Date'] >= START_DATE].copy()
    df = df[df['SP500'] > 0].copy()
    return df


def calculate_returns_and_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate log returns and realized volatility.

    Args:
        df: DataFrame with 'SP500' column

    Returns:
        DataFrame with added columns: log_return, realized_vol_6m, realized_vol_12m, target_vol_3m
    """
    df = df.copy()

    # Log returns
    df['log_return'] = np.log(df['SP500'] / df['SP500'].shift(1))

    # Realized volatility (annualized std dev of monthly returns)
    df['realized_vol_6m'] = df['log_return'].rolling(window=6).std() * np.sqrt(12)
    df['realized_vol_12m'] = df['log_return'].rolling(window=12).std() * np.sqrt(12)

    # Target: next 3-month realized volatility
    df['target_vol_3m'] = df['realized_vol_6m'].shift(-3)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in fundamental data.

    Args:
        df: DataFrame with PE10, Long Interest Rate columns

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    for col in ['PE10', 'Long Interest Rate']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    return df


def temporal_split(df: pd.DataFrame, cutoff_date: str):
    """Split data temporally to prevent lookahead bias.

    Args:
        df: DataFrame with 'Date' column
        cutoff_date: Date string for train/test split

    Returns:
        train_df, test_df
    """
    train_df = df[df['Date'] < cutoff_date].copy()
    test_df = df[df['Date'] >= cutoff_date].copy()
    return train_df, test_df
