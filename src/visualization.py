"""Visualization utilities for volatility forecasting results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(dates, y_true, predictions_dict, save_path='results/forecast_comparison.png'):
    """Create comprehensive 4-panel visualization.

    Args:
        dates: Array of dates
        y_true: Actual volatility values
        predictions_dict: Dict of {model_name: predictions}
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Time series comparison
    ax1 = axes[0, 0]
    ax1.plot(dates, y_true, 'k-', linewidth=2, label='Actual Volatility', alpha=0.8)
    colors = {'Random Forest': '#e74c3c', 'XGBoost': '#3498db', 
              'Weighted Ensemble': '#2ecc71', 'Naive': '#95a5a6'}
    for name, preds in predictions_dict.items():
        if name in colors:
            ax1.plot(dates, preds, color=colors[name], linewidth=1.5, 
                    label=name, alpha=0.8)
    ax1.axvline(pd.Timestamp('2020-03-01'), color='gray', linestyle=':', 
                alpha=0.7, label='COVID Crash')
    ax1.set_title('S&P 500 Realized Volatility Forecasting (3-Month Ahead)', 
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Annualized Volatility')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Feature Importance
    ax2 = axes[0, 1]
    # Placeholder - filled by main.py with actual importances
    ax2.set_title('Feature Importance (XGBoost)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Importance')

    # Plot 3: Predicted vs Actual
    ax3 = axes[1, 0]
    if 'Random Forest' in predictions_dict:
        ax3.scatter(y_true, predictions_dict['Random Forest'], alpha=0.6, 
                   color='#e74c3c', s=50, label='Random Forest')
    if 'Weighted Ensemble' in predictions_dict:
        ax3.scatter(y_true, predictions_dict['Weighted Ensemble'], alpha=0.6, 
                   color='#2ecc71', s=50, label='Weighted Ensemble')
    min_val = min(y_true.min(), min(p.min() for p in predictions_dict.values()))
    max_val = max(y_true.max(), max(p.max() for p in predictions_dict.values()))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
            label='Perfect Prediction')
    ax3.set_xlabel('Actual Volatility')
    ax3.set_ylabel('Predicted Volatility')
    ax3.set_title('Predicted vs Actual Volatility', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Model comparison bar chart
    ax4 = axes[1, 1]
    ax4.set_title('Model Comparison: RMSE', fontsize=13, fontweight='bold')
    ax4.set_ylabel('RMSE')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def plot_feature_importance(feature_names, importances, save_path='results/feature_importance.png'):
    """Plot horizontal bar chart of feature importances.

    Args:
        feature_names: List of feature names
        importances: Array of importance values
        save_path: Path to save figure
    """
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp_df = imp_df.sort_values('Importance', ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp_df['Feature'], imp_df['Importance'], color='#3498db')
    ax.set_title('XGBoost Feature Importance (Top 10)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Feature importance plot saved to {save_path}")
    plt.show()
