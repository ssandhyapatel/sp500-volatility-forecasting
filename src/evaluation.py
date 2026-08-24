"""Evaluation metrics for volatility forecasting."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate(y_true, y_pred, model_name):
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model

    Returns:
        Dictionary of metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy: did we predict up/down correctly?
    if len(y_true) > 1:
        dir_acc = np.mean(
            np.sign(y_pred[1:] - y_pred[:-1]) == 
            np.sign(y_true[1:] - y_true[:-1])
        )
    else:
        dir_acc = np.nan

    return {
        'Model': model_name,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'Dir_Acc': round(dir_acc, 4)
    }


def print_results(results_list):
    """Pretty print results table."""
    df = pd.DataFrame(results_list)
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    # Highlight best model
    best_idx = df['RMSE'].idxmin()
    best_model = df.loc[best_idx, 'Model']
    best_rmse = df.loc[best_idx, 'RMSE']
    print(f"\nBest Model: {best_model} (RMSE: {best_rmse})")
