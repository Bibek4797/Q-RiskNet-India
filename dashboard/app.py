"""
Q-RiskNet India — Master Dashboard Entry Point
Copyright (c) 2026 Bibek Rout
"""
import os
import sys
import streamlit as st

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

import dashboard.pages.overview as overview_page
import dashboard.pages.market_risk as market_risk_page
import dashboard.pages.connectedness as connectedness_page
import dashboard.pages.network as network_page
import dashboard.pages.portfolio_validation as portfolio_validation_page


# ── Cached data-loading functions ────────────────────────────────────────────

@st.cache_data(show_spinner="Loading market data…")
def execute_pipeline(sectors, start, end):
    return pipe.run_data_pipeline(sectors, start, end, save_artifacts=True)


@st.cache_data(show_spinner="Running statistical diagnostics…")
def execute_diagnostics(returns_df):
    return diag_runner.run_all_econometric_diagnostics(returns_df, save_reports=True)


@st.cache_data(show_spinner="Fitting volatility models…")
def execute_volatility_models(returns_df):
    return vol_runner.run_all_volatility_models(returns_df, save_reports=True)


def main():
    """Main dashboard application runner."""
    setup_page_config()
    inject_custom_css()

    # Initialize session state keys
    for key in ['pipeline_output', 'diag_output', 'vol_output', 'spillover_df', 'metrics',
                'active_model_label', 'portfolio_backtest_res', 'pv_fc_results', 'pv_fc_sec',
                'var_backtest_res', 'sens_results']:
        if key not in st.session_state:
            st.session_state[key] = None

    cfg = render_sidebar()
    render_header()

    page = cfg["page_choice"]
    prices_df = returns_df = features_dict = val_report = None
    diag_res = vol_res = None

    # ── Data Requirements ─────────────────────────────────────────────
    selected_sectors = cfg["selected_sectors"]
    if len(selected_sectors) < 2:
        st.warning("Select at least 2 sectors in the sidebar to begin analysis.")
        st.stop()

    try:
        pipeline_res = execute_pipeline(
            tuple(selected_sectors),
            str(cfg["start_date"]),
            str(cfg["end_date"])
        )
        prices_df = pipeline_res["prices"]
        returns_df = pipeline_res["returns"]
        features_dict = pipeline_res["features"]
        val_report = pipeline_res["validation"]

        # Load diagnostics and volatility only for Market & Risk to avoid unnecessary compute
        if page == "📈 Market & Risk":
            diag_res = execute_diagnostics(returns_df)
            vol_res = execute_volatility_models(returns_df)

    except Exception as e:
        diag.log_error("Data pipeline failure", e)
        st.error(f"Data loading error: {str(e)}")
        st.info("Try different sectors or a wider date range in the sidebar.")
        st.stop()

    # ── Page Routing ──────────────────────────────────────────────────
    if page == "🏠 Overview":
        overview_page.render_page(returns_df, cfg)

    elif page == "📈 Market & Risk":
        market_risk_page.render_page(prices_df, returns_df, features_dict, val_report, diag_res, vol_res, cfg)

    elif page == "🌊 Connectedness":
        connectedness_page.render_page(returns_df, returns_df, cfg)

    elif page == "🕸️ Network":
        network_page.render_page(returns_df)

    elif page == "💼 Portfolio & Validation":
        portfolio_validation_page.render_page(returns_df, cfg)


if __name__ == "__main__":
    main()
