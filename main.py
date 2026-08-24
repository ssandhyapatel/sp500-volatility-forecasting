"""End-to-end pipeline for S&P 500 Volatility Forecasting.

Usage:
    python main.py

This script:
1. Loads Robert Shiller S&P 500 data
2. Engineers features
3. Trains multiple models (Ridge, RF, XGBoost, LSTM)
4. Creates weighted ensemble
5. Evaluates all models
6. Generates visualizations
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import TRAIN_CUTOFF
from src.data_loader import load_shiller_data, calculate_returns_and_volatility, clean_data, temporal_split
from src.features import engineer_features, get_feature_columns
from src.models import NaiveModel, RidgeModel, RandomForestModel, XGBoostModel, LSTMModel
from src.ensemble import WeightedEnsemble, EqualEnsemble
from src.evaluation import evaluate, print_results
from src.visualization import plot_results, plot_feature_importance


def main():
    print("="*70)
    print("S&P 500 VOLATILITY FORECASTING PIPELINE")
    print("="*70)

    # 1. Load data
    print("\n[1/6] Loading data...")
    raw_df = load_shiller_data()
    print(f"      Loaded {len(raw_df)} months of data from {raw_df['Date'].min().date()} to {raw_df['Date'].max().date()}")

    # 2. Calculate returns and volatility
    print("\n[2/6] Calculating returns and realized volatility...")
    df = calculate_returns_and_volatility(raw_df)
    df = clean_data(df)

    # 3. Engineer features
    print("\n[3/6] Engineering features...")
    features_df = engineer_features(df)
    feature_cols = get_feature_columns(features_df)
    print(f"      Created {len(feature_cols)} features")
    print(f"      Clean dataset: {len(features_df)} rows")

    # 4. Train/test split
    print("\n[4/6] Splitting data temporally...")
    train_df, test_df = temporal_split(features_df, TRAIN_CUTOFF)
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    print(f"      Train: {len(X_train)} samples ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"      Test:  {len(X_test)} samples ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    # 5. Train models
    print("\n[5/6] Training models...")

    # Naive baseline
    naive = NaiveModel()
    y_naive = naive.predict(X_test, features_df)

    # Ridge
    ridge = RidgeModel(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_ridge = ridge.predict(X_test)
    print("      Ridge: trained")

    # Random Forest
    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    print("      Random Forest: trained")

    # XGBoost
    xgb = XGBoostModel()
    xgb.fit(X_train, y_train)
    y_xgb = xgb.predict(X_test)
    print("      XGBoost: trained")

    # LSTM (requires sequence creation, so we train on full train set)
    lstm = LSTMModel()
    lstm.fit(X_train, y_train)
    y_lstm = lstm.predict(X_test)
    print("      LSTM: trained")

    # 6. Ensemble
    print("\n[6/6] Creating ensemble...")

    # Validation split for ensemble weighting
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    # Fit models on validation for weight calculation
    ridge_val = RidgeModel(alpha=1.0).fit(X_train[:-val_size], y_train[:-val_size])
    rf_val = RandomForestModel().fit(X_train[:-val_size], y_train[:-val_size])
    xgb_val = XGBoostModel().fit(X_train[:-val_size], y_train[:-val_size])
    lstm_val = LSTMModel().fit(X_train[:-val_size], y_train[:-val_size])

    models_for_ensemble = {
        'Ridge': ridge_val,
        'Random Forest': rf_val,
        'XGBoost': xgb_val,
        'LSTM': lstm_val
    }

    weighted_ens = WeightedEnsemble(models_for_ensemble)
    weighted_ens.fit_weights(X_val, y_val)
    y_ensemble = weighted_ens.predict(X_test)

    equal_ens = EqualEnsemble({'Ridge': ridge, 'RF': rf, 'XGB': xgb, 'LSTM': lstm})
    y_equal = equal_ens.predict(X_test)

    # 7. Evaluate
    print("\n" + "="*70)
    results = []
    results.append(evaluate(y_test, y_naive, 'Naive'))
    results.append(evaluate(y_test, y_ridge, 'Ridge'))
    results.append(evaluate(y_test, y_rf, 'Random Forest'))
    results.append(evaluate(y_test, y_xgb, 'XGBoost'))
    results.append(evaluate(y_test, y_lstm, 'LSTM'))
    results.append(evaluate(y_test, y_equal, 'Ensemble (Equal)'))
    results.append(evaluate(y_test, y_ensemble, 'Ensemble (Weighted)'))
    print_results(results)

    # 8. Visualizations
    print("\nGenerating visualizations...")
    test_dates = test_df['Date'].values

    predictions_dict = {
        'Random Forest': y_rf,
        'XGBoost': y_xgb,
        'Weighted Ensemble': y_ensemble,
        'Naive': y_naive
    }

    plot_results(test_dates, y_test, predictions_dict, 
                save_path='results/forecast_comparison.png')

    plot_feature_importance(feature_cols, xgb.feature_importances(),
                           save_path='results/feature_importance.png')

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("Results saved to: results/")
    print("  - forecast_comparison.png")
    print("  - feature_importance.png")


if __name__ == '__main__':
    main()
