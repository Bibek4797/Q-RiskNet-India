"""
Q-RiskNet India — Executive Quantitative Finance Platform Entry Point
Copyright (c) 2026 Bibek Rout
"""
import os
import sys
import pandas as pd
import streamlit as st

# Ensure root directory is on python path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

import src.data.pipeline as pipe
import src.econometrics.garch as garch
import src.econometrics.diagnostics_runner as diag_runner
import src.econometrics.volatility_runner as vol_runner
import src.models.qvar_runner as qvar_runner
import src.diagnostics.logger as diag

from dashboard.utils.theme import setup_page_config, inject_custom_css, render_header
from dashboard.components.sidebar import render_sidebar

import dashboard.pages.home as home_page
import dashboard.pages.data_center as data_center_page
import dashboard.pages.diagnostics as diagnostics_page
import dashboard.pages.volatility as volatility_page
import dashboard.pages.qvar_analysis as qvar_page
import dashboard.pages.connectedness as connectedness_page
import dashboard.pages.network as network_page
import dashboard.pages.forecasting as forecasting_page
import dashboard.pages.validation as validation_page
import dashboard.pages.reports as reports_page
import dashboard.pages.about as about_page


# Initialize page configuration & CSS theme
setup_page_config()
inject_custom_css()

# Initialize session state for persistent outputs
for key in ['pipeline_output', 'diag_output', 'vol_output', 'qvar_output', 'spillover_df', 'metrics']:
    if key not in st.session_state:
        st.session_state[key] = None

# Render Sidebar Navigation & Configuration
cfg = render_sidebar()

# Render Header Banner
render_header()


# ── Cached Data Pipeline & Runner Wrappers ──────────────────────────────────
@st.cache_data(show_spinner="Executing Financial Data Engineering Pipeline...")
def execute_pipeline(sectors, start, end):
    return pipe.run_data_pipeline(sectors, start, end, save_artifacts=True)


@st.cache_data(show_spinner="Executing Econometric Diagnostics Suite...")
def execute_diagnostics(returns_df):
    return diag_runner.run_all_econometric_diagnostics(returns_df, save_reports=True)


@st.cache_data(show_spinner="Fitting GARCH Family Volatility Models...")
def execute_volatility_models(returns_df):
    return vol_runner.run_all_volatility_models(returns_df, save_reports=True)


@st.cache_data(show_spinner="Fitting Multi-Quantile QVAR Models...")
def execute_qvar_suite(returns_df, p_lag):
    return qvar_runner.run_all_qvar_diagnostics(
        returns_df, p=p_lag,
        quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        save_reports=True
    )


@st.cache_data(show_spinner="Estimating GARCH Volatility Proxy...")
def execute_garch_proxy(returns_df):
    garch_cols = {}
    for col in returns_df.columns:
        garch_cols[col] = garch.estimate_garch_volatility(returns_df[col])
    return pd.DataFrame(garch_cols, index=returns_df.index).dropna()


# ── Data Loading Logic ──────────────────────────────────────────────────────
page = cfg["page_choice"]

# Pages that require processed data
data_required_pages = [
    "📊 Data Center",
    "🔬 Econometric Diagnostics",
    "📈 Volatility Modelling",
    "📊 QVAR Analysis",
    "🌊 Connectedness & Spillover",
    "🕸️ Network Topology",
    "🔮 Forecasting Benchmark",
    "🔬 Research Validation"
]

prices_df, returns_df, features_dict, val_report = None, None, None, None
diag_res, vol_res, qvar_res, model_input = None, None, None, None

if page in data_required_pages:
    selected_sectors = cfg["selected_sectors"]
    if len(selected_sectors) < 2:
        st.error("⚠️ Please select at least two sectoral indices in the sidebar to run quantitative analysis.")
        st.stop()

    try:
        pipeline_res = execute_pipeline(tuple(selected_sectors), cfg["start_date"], cfg["end_date"])
        st.session_state['pipeline_output'] = pipeline_res
        prices_df = pipeline_res["prices"]
        returns_df = pipeline_res["returns"]
        features_dict = pipeline_res["features"]
        val_report = pipeline_res["validation"]

        if page in ["🔬 Econometric Diagnostics"]:
            diag_res = execute_diagnostics(returns_df)
            st.session_state['diag_output'] = diag_res

        if page in ["📈 Volatility Modelling"]:
            vol_res = execute_volatility_models(returns_df)
            st.session_state['vol_output'] = vol_res

        if page in ["📊 QVAR Analysis"]:
            qvar_res = execute_qvar_suite(returns_df, cfg["lags"])
            st.session_state['qvar_output'] = qvar_res

        if page in ["🌊 Connectedness & Spillover"]:
            if cfg["volatility_proxy"] == "GARCH(1,1) Volatility":
                model_input = execute_garch_proxy(returns_df)
            else:
                model_input = returns_df.copy()

    except Exception as e:
        diag.log_error("Data pipeline or estimation failure", e)
        st.error(f"❌ Error during data ingestion or model estimation: {str(e)}")
        st.info("💡 Try selecting different sectors or date ranges in the sidebar.")
        st.stop()

# ── Router / Page Rendering ──────────────────────────────────────────────────
if page == "🏠 Home":
    home_page.render_page()
elif page == "📊 Data Center":
    data_center_page.render_page(prices_df, returns_df, features_dict, val_report, cfg)
elif page == "🔬 Econometric Diagnostics":
    diagnostics_page.render_page(returns_df, diag_res)
elif page == "📈 Volatility Modelling":
    volatility_page.render_page(returns_df, vol_res)
elif page == "📊 QVAR Analysis":
    qvar_page.render_page(returns_df, qvar_res)
elif page == "🌊 Connectedness & Spillover":
    connectedness_page.render_page(model_input, returns_df, cfg)
elif page == "🕸️ Network Topology":
    network_page.render_page(returns_df)
elif page == "🔮 Forecasting Benchmark":
    forecasting_page.render_page(returns_df)
elif page == "🔬 Research Validation":
    validation_page.render_page(returns_df)
elif page == "📋 Reports Center":
    reports_page.render_page()
elif page == "ℹ️ About":
    about_page.render_page()
