"""
Q-RiskNet India — Sidebar Navigation & Controls
Copyright (c) 2026 Bibek Rout
"""
from datetime import datetime, timedelta
import streamlit as st
from src.config.settings import TICKER_MAP, MODEL_CFG, GIRF_CFG


def render_sidebar():
    """
    Renders clean sidebar navigation and controls.
    Returns configuration dictionary.
    """
    # ── Brand ──────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style='padding: 10px 4px 8px;'>
        <div style='font-family:"Space Grotesk",sans-serif; font-size:1.05rem; font-weight:700;
                    color:#c7d2fe;'>Q-RiskNet India</div>
        <div style='font-size:0.68rem; color:#475569; text-transform:uppercase;
                    letter-spacing:0.08em; margin-top:2px;'>Systemic Risk Analytics</div>
    </div>
    <div style='height:1px; background:rgba(99,102,241,0.18); margin-bottom:14px;'></div>
    """, unsafe_allow_html=True)

    # ── Navigation ─────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='font-size:0.67rem; font-weight:700; color:#334155; text-transform:uppercase;"
        "letter-spacing:0.09em; margin-bottom:6px;'>Navigation</div>",
        unsafe_allow_html=True
    )

    page_choice = st.sidebar.radio(
        "Section",
        [
            "🏠 Overview",
            "📈 Market & Risk",
            "🌊 Connectedness",
            "🕸️ Network",
            "💼 Portfolio & Validation"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown(
        "<div style='height:1px; background:rgba(99,102,241,0.15); margin:12px 0;'></div>",
        unsafe_allow_html=True
    )

    # ── Data ───────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='font-size:0.67rem; font-weight:700; color:#334155; text-transform:uppercase;"
        "letter-spacing:0.09em; margin-bottom:6px;'>Data</div>",
        unsafe_allow_html=True
    )

    selected_sectors = st.sidebar.multiselect(
        "Sectors",
        options=list(TICKER_MAP.keys()),
        default=list(TICKER_MAP.keys())[:7],
        help="NSE sectoral indices to include in the analysis"
    )

    today = datetime.today()
    five_years_ago = today - timedelta(days=5 * 365)
    start_date = st.sidebar.date_input("Start date", value=five_years_ago)
    end_date = st.sidebar.date_input("End date", value=today)

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date.")

    st.sidebar.markdown(
        "<div style='height:1px; background:rgba(99,102,241,0.15); margin:12px 0;'></div>",
        unsafe_allow_html=True
    )

    # ── Model Parameters ───────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='font-size:0.67rem; font-weight:700; color:#334155; text-transform:uppercase;"
        "letter-spacing:0.09em; margin-bottom:6px;'>Model Parameters</div>",
        unsafe_allow_html=True
    )

    lags = st.sidebar.slider(
        "Lag order (p)",
        min_value=1, max_value=5,
        value=MODEL_CFG.get("qvar", {}).get("default_lags", 2),
        help="Number of lagged periods used in the quantile VAR model. Higher values capture longer-range dependencies."
    )
    forecast_horizon = st.sidebar.slider(
        "Spillover horizon (days)",
        min_value=5, max_value=30,
        value=GIRF_CFG.get("default_horizon", 10),
        help="Number of days ahead over which impulse responses and spillover percentages are accumulated."
    )

    # ── Footer ─────────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='height:1px; background:rgba(99,102,241,0.15); margin:12px 0;'></div>"
        "<div style='font-size:0.67rem; color:#334155; text-align:center; padding:4px 0;'>"
        "© 2026 Bibek Rout · MIT License"
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
