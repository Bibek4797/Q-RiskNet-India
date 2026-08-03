"""
Q-RiskNet India — QVAR Analysis Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import plotly.express as px

import src.models.qvar as qvar
from dashboard.components.charts import render_qvar_heatmap, render_qvar_girf_chart
from dashboard.components.exports import download_csv


def render_page(returns_df, qvar_res):
    """Renders the Quantile VAR Analysis page."""

    st.header("📊 Quantile Vector Autoregression (QVAR)")
    st.caption("Cross-sector dependence structures across extreme bearish (τ=0.05), "
               "normal (τ=0.50), and bullish (τ=0.95) market regimes.")

    models_dict = qvar_res["models_dict"]
    summary_df = qvar_res["summary_df"]

    # ── Quantile Selector ─────────────────────────────────────────────
    c_ctrl, c_heat = st.columns([1, 2])
    with c_ctrl:
        st.markdown("#### QVAR Controls")
        selected_q = st.select_slider(
            "Select Quantile (τ)",
            options=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            value=0.05
        )
        regime = ("Extreme Bearish" if selected_q <= 0.10
                  else "Median (Normal)" if selected_q == 0.50
                  else "Bullish (Rally)")
        st.info(f"Active Regime: **{regime}**")

    with c_heat:
        q_model = models_dict.get(selected_q)
        if q_model:
            coeff = q_model.get_coefficient_matrix(lag=1)
            render_qvar_heatmap(coeff, selected_q)

    st.markdown("---")

    # ── Coefficient Stability Across Quantiles ────────────────────────
    st.subheader("🌀 Coefficient Stability Across Quantiles")
    cp1, cp2 = st.columns(2)
    with cp1:
        target = st.selectbox("Target (Response)", list(returns_df.columns), key="qv_tgt")
    with cp2:
        source = st.selectbox("Source (Impulse)", list(returns_df.columns), key="qv_src")

    if target and source:
        pair = summary_df[(summary_df["Target_Sector"] == target) &
                          (summary_df["Source_Sector"] == source)]
        fig = px.line(pair, x="Quantile", y="Coefficient", markers=True,
                      title=f"Φ₁({target} ← {source}) Across Quantiles τ")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── GIRF ──────────────────────────────────────────────────────────
    st.subheader("⚡ Generalised Impulse Response Functions (GIRF)")
    shock_sec = st.selectbox("Shock Origin Sector", list(returns_df.columns), key="girf_sec")
    if shock_sec and q_model:
        girf_sim = qvar.compute_qvar_girf(q_model, returns_df,
                                           shocked_sector=shock_sec,
                                           shock_size_std=2.0, horizon=10)
        render_qvar_girf_chart(girf_sim, shock_sec, selected_q)

    # ── Export ─────────────────────────────────────────────────────────
    with st.expander("📥 Export QVAR Coefficients"):
        download_csv(summary_df, "qvar_coefficient_summary.csv", key="dl_qvar")
