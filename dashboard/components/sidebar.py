"""
Q-RiskNet India — Clean Executive Sidebar Controls
Copyright (c) 2026 Bibek Rout
"""
from datetime import datetime, timedelta
import streamlit as st
from src.config.settings import TICKER_MAP, MODEL_CFG, GIRF_CFG


def render_sidebar():
    """
    Renders clean, global sidebar controls and returns configuration dictionary.
    """
    st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=60)
    st.sidebar.title("📌 Q-RiskNet Platform")

    st.sidebar.subheader("🗺️ Research Navigation")
    page_choice = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Home",
            "📊 Data Center",
            "🔬 Econometric Diagnostics",
            "📈 Volatility Modelling",
            "📊 QVAR Analysis",
            "🌊 Connectedness & Spillover",
            "🕸️ Network Topology",
            "🔮 Forecasting Benchmark",
            "🔬 Research Validation",
            "📋 Reports Center",
            "ℹ️ About"
        ],
        index=0
    )

    st.sidebar.markdown("---")
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

    if start_date >= end_date:
        st.sidebar.error("⚠️ Start Date must be earlier than End Date.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Global Model Parameters")
    lags = st.sidebar.slider(
        "Autoregressive Lags (p)",
        min_value=1, max_value=5,
        value=MODEL_CFG.get("qvar", {}).get("default_lags", 2)
    )
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (H)",
        min_value=5, max_value=30,
        value=GIRF_CFG.get("default_horizon", 10)
    )

    return {
        "page_choice": page_choice,
        "selected_sectors": selected_sectors,
        "start_date": start_date,
        "end_date": end_date,
        "lags": lags,
        "forecast_horizon": forecast_horizon,
        # Default model settings
        "seq_len": MODEL_CFG.get("quantile_lstm", {}).get("default_seq_len", 5),
        "epochs": MODEL_CFG.get("quantile_lstm", {}).get("default_epochs", 30),
        "hidden_dim": "auto"
    }
