"""
Q-RiskNet India — Forecasting Benchmark Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd

import src.forecasting.evaluator as evaluator
from dashboard.components.charts import (
    render_forecast_benchmark_chart,
    render_feature_importance_chart
)
from dashboard.components.exports import download_csv
from dashboard.components.status import render_empty_state


def render_page(returns_df):
    """Renders the Forecasting Benchmark page."""

    st.header("🔮 Forecasting Benchmark & Predictive Modelling")
    st.caption("Head-to-head comparison of Random Walk, ARIMA, Random Forest, Gradient Boosting, "
               "SVR, and PyTorch Quantile LSTM via out-of-sample RMSE, MAE, Directional Accuracy, "
               "and the Diebold-Mariano test.")

    sector = st.selectbox("Target Sector to Forecast", list(returns_df.columns), key="fc_sec")
    run_fc = st.button("🚀 Run Full Forecast Benchmark", type="primary", key="run_fc")

    if run_fc:
        with st.spinner(f"Running all 7 benchmark models for **{sector}** …"):
            try:
                fc_res = evaluator.run_all_forecast_benchmarks(
                    returns_df, target_sector=sector, train_ratio=0.80, save_reports=True)
                st.session_state["fc_results"] = fc_res
                st.session_state["fc_sector"] = sector
            except Exception as e:
                st.error(f"❌ Forecast benchmark error: {str(e)}")
                return

    if st.session_state.get("fc_results") is None:
        render_empty_state("Select a target sector and click **Run Full Forecast Benchmark** "
                           "to compare all models.")
        return

    fc_res = st.session_state["fc_results"]
    fc_sector = st.session_state.get("fc_sector", sector)

    # ── Performance Rankings ──────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"📊 Model Performance for {fc_sector}")
    comp = fc_res["summary_df"]
    st.dataframe(comp.style.highlight_min(subset=["RMSE", "MAE"], color="#264653")
                 .highlight_max(subset=["Directional_Accuracy_Pct"], color="#2a9d8f"),
                 use_container_width=True)
    download_csv(comp, "forecast_benchmark_summary.csv", key="dl_fc_comp")

    # ── Forecast Overlay ──────────────────────────────────────────────
    st.subheader("📈 Actual vs Predicted Overlay")
    render_forecast_benchmark_chart(fc_res["predictions_df"], fc_sector)

    # ── Diebold-Mariano ───────────────────────────────────────────────
    st.subheader("📐 Diebold-Mariano Test (vs Random Walk)")
    dm = fc_res["dm_df"]
    if dm is not None and not dm.empty:
        st.dataframe(dm.style.apply(
            lambda row: ["background-color: #264653" if row["Significantly_Superior"]
                         else "" for _ in row], axis=1), use_container_width=True)
        download_csv(dm, "diebold_mariano_tests.csv", key="dl_dm")
    else:
        st.info("ℹ️ No Diebold-Mariano comparison available.")
