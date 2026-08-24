"""Streamlit dashboard for S&P 500 Volatility Forecasting.

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# API endpoint
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="S&P 500 Volatility Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 S&P 500 Volatility Forecasting Dashboard")
st.markdown("""
**Multi-model ensemble for predicting S&P 500 realized volatility 3 months ahead.**
Built with Random Forest, XGBoost, and Ridge Regression on Robert Shiller's historical data (1950–2026).
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Live Prediction", "Historical Data", "Model Info"])

if page == "Live Prediction":
    st.header("🔮 Make a Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Volatility Features")
        vol_lag_1m = st.number_input("Volatility Lag 1M", value=0.15, min_value=0.0, max_value=1.0, step=0.01)
        vol_lag_2m = st.number_input("Volatility Lag 2M", value=0.14, min_value=0.0, max_value=1.0, step=0.01)
        vol_lag_3m = st.number_input("Volatility Lag 3M", value=0.13, min_value=0.0, max_value=1.0, step=0.01)
        vol_lag_6m = st.number_input("Volatility Lag 6M", value=0.12, min_value=0.0, max_value=1.0, step=0.01)
        vol_lag_12m = st.number_input("Volatility Lag 12M", value=0.11, min_value=0.0, max_value=1.0, step=0.01)

        vol_mean_3m = st.number_input("Vol Mean 3M", value=0.14, min_value=0.0, max_value=1.0, step=0.01)
        vol_mean_6m = st.number_input("Vol Mean 6M", value=0.13, min_value=0.0, max_value=1.0, step=0.01)
        vol_max_3m = st.number_input("Vol Max 3M", value=0.18, min_value=0.0, max_value=1.0, step=0.01)

    with col2:
        st.subheader("Market Features")
        return_lag_1m = st.number_input("Return Lag 1M", value=0.01, min_value=-0.5, max_value=0.5, step=0.001)
        return_lag_2m = st.number_input("Return Lag 2M", value=0.005, min_value=-0.5, max_value=0.5, step=0.001)
        return_lag_3m = st.number_input("Return Lag 3M", value=-0.002, min_value=-0.5, max_value=0.5, step=0.001)

        pe10 = st.number_input("CAPE Ratio (PE10)", value=30.0, min_value=5.0, max_value=50.0, step=0.1)
        interest_rate = st.number_input("Interest Rate (%)", value=4.5, min_value=0.0, max_value=20.0, step=0.1)
        dividend_yield = st.number_input("Dividend Yield", value=0.015, min_value=0.0, max_value=0.1, step=0.001)

        month = st.selectbox("Month", range(1, 13), index=datetime.now().month - 1)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

    if st.button("🚀 Predict Volatility", type="primary"):
        payload = {
            "vol_lag_1m": vol_lag_1m,
            "vol_lag_2m": vol_lag_2m,
            "vol_lag_3m": vol_lag_3m,
            "vol_lag_6m": vol_lag_6m,
            "vol_lag_12m": vol_lag_12m,
            "return_lag_1m": return_lag_1m,
            "return_lag_2m": return_lag_2m,
            "return_lag_3m": return_lag_3m,
            "vol_mean_3m": vol_mean_3m,
            "vol_mean_6m": vol_mean_6m,
            "vol_max_3m": vol_max_3m,
            "pe10": pe10,
            "interest_rate": interest_rate,
            "dividend_yield": dividend_yield,
            "month_sin": month_sin,
            "month_cos": month_cos
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()

                col_res1, col_res2, col_res3 = st.columns(3)
                with col_res1:
                    st.metric("Predicted Volatility", f"{result['predicted_volatility']:.4f}")
                with col_res2:
                    st.metric("Confidence", result['confidence'])
                with col_res3:
                    st.metric("Model", result['model'].upper())

                # Interpretation
                vol = result['predicted_volatility']
                if vol < 0.1:
                    st.success("📉 Low volatility regime expected. Risk-off environment.")
                elif vol < 0.2:
                    st.info("📊 Moderate volatility. Normal market conditions.")
                elif vol < 0.3:
                    st.warning("📈 Elevated volatility. Increased hedging recommended.")
                else:
                    st.error("🚨 High volatility regime! Significant market stress expected.")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")
            st.info("Make sure the FastAPI backend is running on port 8000")

elif page == "Historical Data":
    st.header("📊 Historical Volatility Data")

    try:
        response = requests.get(f"{API_URL}/history?limit=100")
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'])

            # Plot
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(df['date'], df['volatility'], color='#e74c3c', linewidth=1.5)
            ax.set_title('Historical S&P 500 Realized Volatility', fontsize=14, fontweight='bold')
            ax.set_ylabel('Annualized Volatility')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Volatility", f"{df['volatility'].mean():.4f}")
            with col2:
                st.metric("Max Volatility", f"{df['volatility'].max():.4f}")
            with col3:
                st.metric("Current", f"{df['volatility'].iloc[-1]:.4f}")
        else:
            st.error("Failed to fetch historical data")
    except Exception as e:
        st.error(f"Connection error: {e}")

elif page == "Model Info":
    st.header("🤖 Model Information")

    st.subheader("Architecture")
    st.markdown("""
    - **Random Forest**: 200 trees, max depth 8 — captures non-linear regime switches
    - **XGBoost**: 200 estimators, learning rate 0.05 — sequential error correction
    - **Ridge Regression**: L2 regularized linear model — baseline interpretability
    - **Ensemble**: Equal-weighted combination of all three models
    """)

    st.subheader("Features")
    st.markdown("""
    | Category | Features |
    |---|---|
    | Lagged Volatility | 1M, 2M, 3M, 6M, 12M |
    | Lagged Returns | 1M, 2M, 3M |
    | Rolling Stats | Mean 3M/6M, Max 3M |
    | Valuation | CAPE, Interest Rate, Dividend Yield |
    | Calendar | Month sin/cos |
    """)

    st.subheader("Performance")
    st.markdown("""
    | Model | RMSE | Directional Accuracy |
    |---|---|---|
    | Random Forest | 0.0592 | 62.8% |
    | XGBoost | 0.0646 | 61.2% |
    | Ridge | 0.0658 | 50.4% |
    | Ensemble | 0.0628 | 59.7% |
    """)
