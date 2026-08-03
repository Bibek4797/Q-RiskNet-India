from datetime import datetime, timedelta
import streamlit as st
from src.config.settings import TICKER_MAP, MODEL_CFG, GIRF_CFG

def render_sidebar():
    """
    Renders sidebar controls and returns configuration dictionary.
    """
    st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=60)
    st.sidebar.title("📌 Q-RiskNet Controls")

    st.sidebar.subheader("📅 Data Configuration")
    selected_sectors = st.sidebar.multiselect(
        "Select Sectoral Indices",
        options=list(TICKER_MAP.keys()),
        default=list(TICKER_MAP.keys())[:7]
    )

    today = datetime.today()
    five_years_ago = today - timedelta(days=5 * 365)
    start_date = st.sidebar.date_input("Start Date", value=five_years_ago)
    end_date = st.sidebar.date_input("End Date", value=today)

    st.sidebar.subheader("⚙️ Model Settings")
    model_choice = st.sidebar.radio("Model Type", ["Quantile VAR (QVAR)", "Quantile LSTM"])
    quantile = st.sidebar.slider("Quantile Value (τ)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

    optim_mode = st.sidebar.radio("Hyperparameter Engine", ["⚡ Auto-Optimized (Recommended)", "⚙️ Custom Parameters"])

    lags = MODEL_CFG.get("qvar", {}).get("default_lags", 2)
    seq_len = MODEL_CFG.get("quantile_lstm", {}).get("default_seq_len", 5)
    epochs = MODEL_CFG.get("quantile_lstm", {}).get("default_epochs", 50)
    hidden_dim = "auto"

    if optim_mode == "⚙️ Custom Parameters":
        if model_choice == "Quantile VAR (QVAR)":
            lags = st.sidebar.slider("Autoregressive Lags (p)", min_value=1, max_value=5, value=lags)
        else:
            seq_len = st.sidebar.slider("Sequence Length (Lags)", min_value=3, max_value=15, value=seq_len)
            epochs = st.sidebar.slider("LSTM Epochs", min_value=10, max_value=100, value=30, step=5)
            hidden_dim = st.sidebar.slider("Hidden Layer Nodes", min_value=8, max_value=64, value=16, step=8)
    else:
        st.sidebar.caption("🤖 Auto-tunes hidden layers, early-stopping convergence, and network topology automatically.")

    volatility_proxy = st.sidebar.selectbox("Risk / Volatility Metric", ["Log Returns", "GARCH(1,1) Volatility"])
    forecast_horizon = st.sidebar.slider("Forecast Horizon (H)", min_value=5, max_value=30, value=GIRF_CFG.get("default_horizon", 10))

    return {
        "selected_sectors": selected_sectors,
        "start_date": start_date,
        "end_date": end_date,
        "model_choice": model_choice,
        "quantile": quantile,
        "optim_mode": optim_mode,
        "lags": lags,
        "seq_len": seq_len,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "volatility_proxy": volatility_proxy,
        "forecast_horizon": forecast_horizon
    }
