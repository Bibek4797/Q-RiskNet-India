"""
Q-RiskNet India — Volatility Modelling Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd

import src.econometrics.volatility as vol_mod
import src.econometrics.volatility_runner as vol_runner
from dashboard.components.charts import render_conditional_volatility_chart
from dashboard.components.exports import download_csv


def render_page(returns_df, vol_res):
    """Renders the Volatility Modelling page."""

    st.header("📈 Volatility Modelling & Asymmetry Analysis")
    st.caption("Comparative estimation of ARCH(1), GARCH(1,1), EGARCH(1,1,1), and GJR-GARCH(1,1,1). "
               "Evaluates conditional volatility, persistence, shock half-life, and asymmetric leverage.")

    sector = st.selectbox("Select Sector for Volatility Modelling",
                          list(returns_df.columns), key="vol_sec")

    if not sector:
        return

    st.markdown(f"### Model Comparison for **{sector}**")
    sector_df = vol_runner.compare_volatility_models_for_sector(returns_df[sector])
    display_df = sector_df.drop(columns=["fit_result"], errors="ignore")
    st.dataframe(display_df, use_container_width=True)
    st.caption("Sorted by AIC. Lower AIC / BIC → superior parsimonious fit.")
    download_csv(display_df, f"volatility_comparison_{sector}.csv", key="dl_vol_comp")

    model_name = st.radio("Select Model to Inspect",
                          list(sector_df["Model"].values), horizontal=True)

    row = sector_df[sector_df["Model"] == model_name].iloc[0]
    res_obj = row["fit_result"]

    cond_vol = res_obj.conditional_volatility / (res_obj.scale if res_obj.scale else 1.0)
    render_conditional_volatility_chart(returns_df[sector], cond_vol, model_name)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Multi-Step Volatility Forecasts")
        fc = vol_mod.generate_multi_step_volatility_forecast(res_obj, horizons=[1, 5, 20])
        st.table(pd.DataFrame([
            {"Horizon": "1-Day (t+1)", "Annualised Vol": f"{fc['Forecast_1d_Vol_Pct']:.2f}%"},
            {"Horizon": "5-Day (t+5)", "Annualised Vol": f"{fc['Forecast_5d_Vol_Pct']:.2f}%"},
            {"Horizon": "20-Day (t+20)", "Annualised Vol": f"{fc['Forecast_20d_Vol_Pct']:.2f}%"},
        ]))
    with c2:
        st.markdown("#### Parameter & Persistence Metrics")
        st.json({
            "Persistence (P)": f"{row['Persistence']:.4f}",
            "Half-Life (days)": f"{row['Half_Life_Days']:.1f}",
            "Long-Run Vol": f"{row['Long_Run_Vol_Pct']:.2f}%",
            "Asymmetry γ": f"{row['Gamma_Asymmetry']}"
        })
