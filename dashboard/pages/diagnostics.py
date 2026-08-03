"""
Q-RiskNet India — Econometric Diagnostics Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st

import src.econometrics.autocorr as autocorr
import src.econometrics.hetero as hetero
import src.econometrics.distribution as dist_mod
from dashboard.components.charts import (
    render_acf_pacf_chart, render_rolling_variance_chart,
    render_kde_comparison_chart
)
from dashboard.components.exports import download_csv


def render_page(returns_df, diag_res):
    """Renders the Econometric Diagnostics page."""

    st.header("🔬 Econometric Diagnostics & Statistical Assumption Validation")
    st.caption("Rigorous verification of stationarity, autocorrelation, heteroskedasticity, "
               "fat tails, non-linearity, and structural breaks.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Stationarity (ADF/KPSS/ZA)",
        "2. Autocorrelation (ACF/LB)",
        "3. Volatility Clustering (ARCH-LM)",
        "4. Distribution & Tails (JB/KDE)",
        "5. Non-Linearity & Breaks"
    ])

    # ── Tab 1: Stationarity ───────────────────────────────────────────
    with tab1:
        st.subheader("📌 Unit Root & Stationarity Tests")
        st.dataframe(diag_res["stationarity"], use_container_width=True)
        st.caption("ADF: Null = Unit Root · KPSS: Null = Trend Stationary · "
                   "Zivot-Andrews: Null = Unit Root w/ Structural Break")
        download_csv(diag_res["stationarity"], "stationarity_tests.csv", key="dl_stat")

    # ── Tab 2: Autocorrelation ────────────────────────────────────────
    with tab2:
        st.subheader("🔄 Autocorrelation & Serial Correlation")
        st.dataframe(diag_res["autocorrelation"], use_container_width=True)
        sector = st.selectbox("Select Sector for ACF/PACF", list(returns_df.columns),
                              key="diag_acf_sec")
        if sector:
            data = autocorr.compute_acf_pacf(returns_df[sector], nlags=20)
            render_acf_pacf_chart(data["lags"], data["acf"], data["pacf"], sector)
        download_csv(diag_res["autocorrelation"], "autocorrelation_tests.csv", key="dl_acf")

    # ── Tab 3: ARCH-LM ───────────────────────────────────────────────
    with tab3:
        st.subheader("⚡ Heteroskedasticity & Volatility Clustering")
        st.dataframe(diag_res["heteroskedasticity"], use_container_width=True)
        sector_h = st.selectbox("Select Sector for Rolling Variance", list(returns_df.columns),
                                key="diag_arch_sec")
        if sector_h:
            rv = hetero.compute_rolling_variance(returns_df[sector_h])
            render_rolling_variance_chart(rv, sector_h)
        download_csv(diag_res["heteroskedasticity"], "heteroskedasticity_tests.csv", key="dl_het")

    # ── Tab 4: Distribution ───────────────────────────────────────────
    with tab4:
        st.subheader("📊 Distribution Analysis & Fat Tail Behaviour")
        st.dataframe(diag_res["distribution"], use_container_width=True)
        sector_d = st.selectbox("Select Sector for KDE vs Normal Overlay",
                                list(returns_df.columns), key="diag_dist_sec")
        if sector_d:
            kde = dist_mod.get_kde_comparison(returns_df[sector_d])
            render_kde_comparison_chart(returns_df[sector_d], kde["x"], kde["gaussian_pdf"])
        download_csv(diag_res["distribution"], "distribution_tests.csv", key="dl_dist")

    # ── Tab 5: Non-Linearity & Structural Breaks ─────────────────────
    with tab5:
        st.subheader("🌀 Non-Linearity & Structural Break Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### BDS Test for Non-Linear Dependence")
            st.dataframe(diag_res["nonlinearity"], use_container_width=True)
        with c2:
            st.markdown("#### OLS CUSUM Parameter Stability")
            st.dataframe(diag_res["structural_breaks"], use_container_width=True)
