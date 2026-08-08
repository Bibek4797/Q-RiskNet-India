"""
Q-RiskNet India — Premium Sidebar Navigation
Copyright (c) 2026 Bibek Rout
"""
from datetime import datetime, timedelta
import streamlit as st
from src.config.settings import TICKER_MAP, MODEL_CFG, GIRF_CFG


def render_sidebar():
    """
    Renders the premium sidebar with navigation and configuration controls.
    Returns the configuration dictionary.
    """
    # ── Brand Header ───────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style='text-align:center; padding: 18px 0 12px;'>
        <div style='font-size:2.4rem; margin-bottom:6px;'>📡</div>
        <div style='font-family:"Space Grotesk",sans-serif; font-size:1.15rem; font-weight:700;
                    background:linear-gradient(135deg,#818cf8,#a78bfa); -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:-0.01em;'>
            Q-RiskNet India
        </div>
        <div style='font-size:0.7rem; color:#475569; text-transform:uppercase;
                    letter-spacing:0.1em; margin-top:2px;'>Risk Analytics Platform</div>
    </div>
    <div style='height:1px; background:linear-gradient(90deg,transparent,rgba(99,102,241,0.5),transparent);
                margin-bottom:16px;'></div>
    """, unsafe_allow_html=True)

    # ── Navigation ─────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='font-size:0.7rem; font-weight:700; color:#475569; text-transform:uppercase;"
        "letter-spacing:0.1em; margin-bottom:8px; padding-left:2px;'>Navigation</div>",
        unsafe_allow_html=True
    )

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
        index=0,
        label_visibility="collapsed"
    )

    # ── Divider ────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='height:1px; background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3),transparent);"
        "margin:14px 0;'></div>",
        unsafe_allow_html=True
    )

    # ── Data Configuration ─────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='font-size:0.7rem; font-weight:700; color:#475569; text-transform:uppercase;"
        "letter-spacing:0.1em; margin-bottom:8px; padding-left:2px;'>Data Configuration</div>",
        unsafe_allow_html=True
    )

    selected_sectors = st.sidebar.multiselect(
        "Sectoral Indices",
        options=list(TICKER_MAP.keys()),
        default=list(TICKER_MAP.keys())[:7],
        help="Select NSE sectoral indices to include in analysis"
    )

    today = datetime.today()
    five_years_ago = today - timedelta(days=5 * 365)
    start_date = st.sidebar.date_input("Start Date", value=five_years_ago)
    end_date = st.sidebar.date_input("End Date", value=today)

    if start_date >= end_date:
        st.sidebar.error("⚠️ Start Date must be earlier than End Date.")

    # ── Model Parameters ───────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='height:1px; background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3),transparent);"
        "margin:14px 0;'></div>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<div style='font-size:0.7rem; font-weight:700; color:#475569; text-transform:uppercase;"
        "letter-spacing:0.1em; margin-bottom:8px; padding-left:2px;'>Model Parameters</div>",
        unsafe_allow_html=True
    )

    lags = st.sidebar.slider(
        "Autoregressive Lags (p)",
        min_value=1, max_value=5,
        value=MODEL_CFG.get("qvar", {}).get("default_lags", 2),
        help="Number of lags for the QVAR model"
    )
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (H)",
        min_value=5, max_value=30,
        value=GIRF_CFG.get("default_horizon", 10),
        help="GIRF / spillover decomposition horizon"
    )

    # ── Footer ─────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='height:1px; background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3),transparent);"
        "margin:14px 0;'></div>"
        "<div style='font-size:0.68rem; color:#334155; text-align:center; padding:4px 0 8px;'>"
        "© 2026 Bibek Rout &nbsp;·&nbsp; MIT License"
        "</div>",
        unsafe_allow_html=True
    )

    return {
        "page_choice": page_choice,
        "selected_sectors": selected_sectors,
        "start_date": start_date,
        "end_date": end_date,
        "lags": lags,
        "forecast_horizon": forecast_horizon,
        "seq_len": MODEL_CFG.get("quantile_lstm", {}).get("default_seq_len", 5),
        "epochs": MODEL_CFG.get("quantile_lstm", {}).get("default_epochs", 30),
        "hidden_dim": "auto"
    }
