"""FastAPI backend for S&P 500 Volatility Forecasting.

Run: uvicorn app:app --reload
"""

import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    VOL_LAGS, RETURN_LAGS, LSTM_SEQ_LEN
)
from src.data_loader import load_shiller_data, calculate_returns_and_volatility, clean_data
from src.features import engineer_features, get_feature_columns
from src.models import RidgeModel, RandomForestModel, XGBoostModel

app = FastAPI(
    title="S&P 500 Volatility Forecasting API",
    description="Predict 3-month-ahead S&P 500 realized volatility using ensemble ML models",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODELS = {}
FEATURE_COLS = None
FEATURES_DF = None


class VolatilityInput(BaseModel):
    """Input for single prediction."""
    vol_lag_1m: float
    vol_lag_2m: float
    vol_lag_3m: float
    vol_lag_6m: float
    vol_lag_12m: float
    return_lag_1m: float
    return_lag_2m: float
    return_lag_3m: float
    vol_mean_3m: float
    vol_mean_6m: float
    vol_max_3m: float
    pe10: float
    interest_rate: float
    dividend_yield: float
    month_sin: float
    month_cos: float


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    model: str
    predicted_volatility: float
    confidence: str
    timestamp: str


@app.on_event("startup")
def load_models():
    """Train models on full dataset at startup."""
    global MODELS, FEATURE_COLS, FEATURES_DF

    print("Loading data and training models...")

    # Load and preprocess data
    raw_df = load_shiller_data()
    df = calculate_returns_and_volatility(raw_df)
    df = clean_data(df)

    # Engineer features
    features_df = engineer_features(df)
    FEATURE_COLS = get_feature_columns(features_df)
    FEATURES_DF = features_df

    # Train on full dataset
    X = features_df[FEATURE_COLS].values
    y = features_df['target'].values

    MODELS['ridge'] = RidgeModel(alpha=1.0).fit(X, y)
    MODELS['random_forest'] = RandomForestModel().fit(X, y)
    MODELS['xgboost'] = XGBoostModel().fit(X, y)

    print(f"Models trained on {len(X)} samples")


@app.get("/")
def root():
    return {
        "message": "S&P 500 Volatility Forecasting API",
        "version": "1.0.0",
        "endpoints": [
            "/predict - POST: Get volatility prediction",
            "/health - GET: Health check",
            "/latest - GET: Latest data point",
            "/history - GET: Historical predictions"
        ]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": list(MODELS.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: VolatilityInput):
    """Predict volatility using ensemble of trained models."""
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Convert input to array
    features = np.array([[
        input_data.vol_lag_1m, input_data.vol_lag_2m, input_data.vol_lag_3m,
        input_data.vol_lag_6m, input_data.vol_lag_12m,
        input_data.return_lag_1m, input_data.return_lag_2m, input_data.return_lag_3m,
        input_data.vol_mean_3m, input_data.vol_mean_6m, input_data.vol_max_3m,
        input_data.pe10, input_data.interest_rate, input_data.dividend_yield,
        input_data.month_sin, input_data.month_cos
    ]])

    # Get predictions from all models
    predictions = {}
    for name, model in MODELS.items():
        predictions[name] = float(model.predict(features)[0])

    # Ensemble prediction (equal weights for API simplicity)
    ensemble_pred = np.mean(list(predictions.values()))

    # Confidence based on model agreement
    std_dev = np.std(list(predictions.values()))
    if std_dev < 0.01:
        confidence = "High"
    elif std_dev < 0.03:
        confidence = "Medium"
    else:
        confidence = "Low"

    return PredictionResponse(
        model="ensemble",
        predicted_volatility=round(ensemble_pred, 4),
        confidence=confidence,
        timestamp=pd.Timestamp.now().isoformat()
    )


@app.get("/latest")
def get_latest():
    """Get the latest data point with features."""
    if FEATURES_DF is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    latest = FEATURES_DF.iloc[-1]
    return {
        "date": str(latest['Date']),
        "features": {col: round(latest[col], 4) for col in FEATURE_COLS},
        "actual_volatility": round(latest['target'], 4) if not pd.isna(latest['target']) else None
    }


@app.get("/history")
def get_history(limit: int = 50):
    """Get recent historical data points."""
    if FEATURES_DF is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    recent = FEATURES_DF.tail(limit)
    return {
        "count": len(recent),
        "data": [
            {
                "date": str(row['Date']),
                "volatility": round(row['target'], 4) if not pd.isna(row['target']) else None
            }
            for _, row in recent.iterrows()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
