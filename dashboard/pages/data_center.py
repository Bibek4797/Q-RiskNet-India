"""
Q-RiskNet India — Data Center & Pipeline Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd
import plotly.express as px

import src.econometrics.stats as stats
from dashboard.components.charts import (
    render_prices_chart, render_drawdowns_chart,
    render_rolling_volatility_chart, render_correlation_chart
)
from dashboard.components.tables import render_descriptive_table
from dashboard.components.exports import download_csv


def render_page(prices_df, returns_df, features_dict, val_report, cfg):
    """Renders the Data Center & Pipeline page."""

    st.header("📊 Data Center & Financial Engineering Pipeline")
    st.caption("Enterprise-grade data ingestion, quality validation, and feature engineering for NSE sectoral indices.")

    # ── KPI Row ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", val_report["total_rows"])
    c2.metric("Selected Sectors", len(cfg["selected_sectors"]))
    c3.metric("Date Coverage",
              f"{prices_df.index[0].strftime('%Y-%m-%d')} → {prices_df.index[-1].strftime('%Y-%m-%d')}")
    quality_status = "✅ CLEAN" if val_report["is_valid"] else "⚠️ WARNINGS"
    c4.metric("Data Quality", quality_status)

    if val_report["warnings"]:
        with st.expander("🔍 Inspect Data Validation Warnings"):
            for w in val_report["warnings"]:
                st.warning(w)

    st.markdown("---")

    # ── View Selector ─────────────────────────────────────────────────
    view = st.radio(
        "Select Feature View",
        ["Base-100 Prices", "Log Returns", "Historical Drawdowns",
         "20d Rolling Volatility", "Pearson Correlation", "Descriptive Statistics"],
        horizontal=True
    )

    if view == "Base-100 Prices":
        render_prices_chart(prices_df)
    elif view == "Log Returns":
        fig = px.line(returns_df, x=returns_df.index, y=returns_df.columns,
                      title="Sectoral Percentage Log Returns (%)")
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
    elif view == "Historical Drawdowns":
        render_drawdowns_chart(features_dict["drawdowns"])
    elif view == "20d Rolling Volatility":
        render_rolling_volatility_chart(features_dict["volatility_20d"], window_label="20-Day")
    elif view == "Pearson Correlation":
        render_correlation_chart(returns_df.corr())
    elif view == "Descriptive Statistics":
        desc = stats.get_descriptive_stats(returns_df)
        render_descriptive_table(desc)
        download_csv(desc, "descriptive_statistics.csv", key="dl_desc_stats")

    # ── Export ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📥 Export Raw Data"):
        ec1, ec2 = st.columns(2)
        with ec1:
            download_csv(prices_df, "prices_data.csv", "📥 Download Prices CSV", key="dl_prices")
        with ec2:
            download_csv(returns_df, "log_returns_data.csv", "📥 Download Returns CSV", key="dl_returns")
