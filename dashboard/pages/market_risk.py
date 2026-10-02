"""
Q-RiskNet India — Market & Risk Section
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd
import plotly.express as px

import src.econometrics.stats as stats
import src.econometrics.volatility as vol_mod
import src.econometrics.volatility_runner as vol_runner
from dashboard.components.charts import (
    render_prices_chart, render_drawdowns_chart,
    render_rolling_volatility_chart, render_correlation_chart,
    render_conditional_volatility_chart, _render_plotly
)
from dashboard.components.tables import render_descriptive_table
from dashboard.components.exports import download_csv


# ── Compact diagnostic summary helper ─────────────────────────────────────────
def _render_diagnostic_summary(diag_res):
    """Renders a compact, readable diagnostic summary table."""
    rows = []

    # Stationarity
    if "stationarity" in diag_res and diag_res["stationarity"] is not None:
        stat_df = diag_res["stationarity"]
        for _, row in stat_df.iterrows():
            decision = row.get("Decision", "")
            interpretation = "Stationary ✓" if "stationary" in str(decision).lower() else "Non-stationary (unit root)"
            rows.append({
                "Diagnostic": f"Stationarity — {row.get('Sector', '')}",
                "Result": row.get("Decision", ""),
                "Interpretation": interpretation
            })

    # ARCH effects
    if "heteroskedasticity" in diag_res and diag_res["heteroskedasticity"] is not None:
        arch_df = diag_res["heteroskedasticity"]
        for _, row in arch_df.iterrows():
            present = row.get("ARCH_Effects_Present", False)
            interpretation = "Volatility clustering present ✓" if present else "No ARCH effects detected"
            rows.append({
                "Diagnostic": f"ARCH Effects — {row.get('Sector', '')}",
                "Result": "Detected" if present else "Not detected",
                "Interpretation": interpretation
            })

    # Distribution
    if "distribution" in diag_res and diag_res["distribution"] is not None:
        dist_df = diag_res["distribution"]
        for _, row in dist_df.iterrows():
            p = row.get("JB_p_value", 1.0)
            interpretation = "Non-normal tails detected ✓" if float(p) < 0.05 else "No strong evidence of non-normality"
            rows.append({
                "Diagnostic": f"Normality (JB) — {row.get('Sector', '')}",
                "Result": f"p = {float(p):.4f}",
                "Interpretation": interpretation
            })

    if rows:
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("Diagnostic results not available.")


def render_page(prices_df, returns_df, features_dict, val_report, diag_res, vol_res, cfg):
    """Renders the streamlined Market & Risk section."""

    st.markdown("### Market & Risk Analysis")
    st.caption("Sector price series, returns, econometric diagnostic tests, and asymmetric GARCH volatility.")

    tab_data, tab_vol = st.tabs(["📊 Market Data & Tests", "⚡ Asymmetric Volatility (GARCH)"])

    # ── Tab 1: Market Data & Econometric Tests ─────────────────────────
    with tab_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", val_report.get("total_rows", "—"))
        c2.metric("Sectors", len(cfg.get("selected_sectors", [])))
        start_str = prices_df.index[0].strftime('%b %d, %Y') if prices_df is not None else "—"
        end_str = prices_df.index[-1].strftime('%b %d, %Y') if prices_df is not None else "—"
        c3.metric("From", start_str)
        c4.metric("To", end_str)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        view = st.radio(
            "View",
            ["Prices (Base 100)", "Log Returns (%)", "Daily Simple Returns (%)", "Correlation Matrix", "Descriptive Stats"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if view == "Prices (Base 100)":
            render_prices_chart(prices_df)
        elif view == "Log Returns (%)":
            fig = px.line(
                returns_df, x=returns_df.index, y=returns_df.columns,
                title="Daily Percentage Log Returns: r_t = ln(P_t / P_{t-1}) × 100",
                labels={"value": "Log Return (%)", "variable": "Sector"}
            )
            _render_plotly(fig, height=440)
        elif view == "Daily Simple Returns (%)":
            simple_ret = features_dict.get("daily_returns", returns_df)
            fig = px.line(
                simple_ret, x=simple_ret.index, y=simple_ret.columns,
                title="Daily Simple Returns: R_t = (P_t - P_{t-1}) / P_{t-1} × 100",
                labels={"value": "Simple Return (%)", "variable": "Sector"}
            )
            _render_plotly(fig, height=440)
        elif view == "Correlation Matrix":
            render_correlation_chart(returns_df.corr())
        elif view == "Descriptive Stats":
            desc = stats.get_descriptive_stats(returns_df)
            render_descriptive_table(desc)
            with st.expander("Download data"):
                download_csv(desc, "descriptive_statistics.csv", key="dl_desc_mr")

        # ── Pillar A: The 3 Core Econometric Tests & Inferences ───────
        st.markdown("---")
        st.markdown("#### 🔬 Econometric Pre-Testing (Foundational Sanity Checks)")
        st.caption("Empirical justification for non-linear tail-risk and volatility modeling.")

        if diag_res is not None:
            _render_diagnostic_summary(diag_res)

            with st.expander("Detailed Test Results & Empirical Inferences", expanded=False):
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("**1. Stationarity (ADF Test):** Confirms return series are $I(0)$ stationary. Prevents spurious regressions in time-series models.")
                    if "stationarity" in diag_res and diag_res["stationarity"] is not None:
                        stat_df = diag_res["stationarity"]
                        cols_stat = [c for c in ["Sector", "Test", "Statistic", "p_value", "Decision"] if c in stat_df.columns]
                        st.dataframe(stat_df[cols_stat], use_container_width=True, hide_index=True)

                    st.markdown("**2. ARCH Effects (Engle's LM Test):** Rejection of constant variance proves volatility is heteroskedastic and clustered.")
                    if "heteroskedasticity" in diag_res and diag_res["heteroskedasticity"] is not None:
                        arch_df = diag_res["heteroskedasticity"]
                        cols_arch = [c for c in ["Sector", "Test", "LM_Statistic", "p_value", "ARCH_Effects_Present"] if c in arch_df.columns]
                        st.dataframe(arch_df[cols_arch], use_container_width=True, hide_index=True)

                with col_t2:
                    st.markdown("**3. Distributional Normality (Jarque-Bera Test):** Rejection of Gaussianity ($p < 0.001$) proves presence of fat tails, justifying Quantile Modeling & CVaR.")
                    if "distribution" in diag_res and diag_res["distribution"] is not None:
                        dist_df = diag_res["distribution"]
                        cols_dist = [c for c in ["Sector", "Mean (%)", "Std_Dev (%)", "Skewness", "Kurtosis", "JB_p_value", "Is_Normal", "Tail_Behavior"] if c in dist_df.columns]
                        st.dataframe(dist_df[cols_dist], use_container_width=True, hide_index=True)
        else:
            st.info("Computing econometric diagnostics…")

    # ── Tab 2: Asymmetric Volatility (GJR-GARCH & GARCH) ───────────────
    with tab_vol:
        st.caption("GARCH models estimate time-varying conditional volatility σ_t. GJR-GARCH captures asymmetric leverage effects where negative shocks amplify volatility more than positive ones.")

        sector = st.selectbox("Select Sector for Volatility Analysis", list(returns_df.columns), key="mr_vol_sec")

        if sector:
            with st.spinner(f"Fitting volatility models for {sector}…"):
                all_models_df = vol_runner.compare_volatility_models_for_sector(returns_df[sector])
                # Filter to only GJR-GARCH and standard GARCH
                sector_df = all_models_df[all_models_df["Model"].isin(["GJR-GARCH(1,1,1)", "GARCH(1,1)"])].copy()

            display_df = sector_df.drop(columns=["fit_result"], errors="ignore")

            st.markdown(f"**Model Comparison: GJR-GARCH(1,1,1) vs GARCH(1,1) — {sector}**")
            st.caption("Lower AIC/BIC = better fit. Negative Shock Sensitivity (γ) > 0 confirms asymmetric leverage effect.")

            friendly_display = display_df.rename(columns={
                "Half_Life_Days": "Volatility Half-Life (days)",
                "Long_Run_Vol_Pct": "Long-Run Volatility (%)",
                "Gamma_Asymmetry": "Negative Shock Sensitivity (γ)"
            })
            show_cols = [c for c in ["Model", "AIC", "BIC", "Persistence", "Volatility Half-Life (days)",
                                      "Long-Run Volatility (%)", "Negative Shock Sensitivity (γ)"]
                         if c in friendly_display.columns]
            st.dataframe(friendly_display[show_cols], use_container_width=True, hide_index=True)

            model_name = st.radio(
                "Active Volatility Model",
                list(sector_df["Model"].values),
                horizontal=True,
                key="mr_model_choice"
            )
            row = sector_df[sector_df["Model"] == model_name].iloc[0]
            res_obj = row["fit_result"]

            # Dynamic conditional volatility chart (±2σ bands)
            cond_vol = res_obj.conditional_volatility / (res_obj.scale if res_obj.scale else 1.0)
            render_conditional_volatility_chart(returns_df[sector], cond_vol, model_name)

            # Volatility forecast & persistence metrics
            cv1, cv2, cv3 = st.columns(3)
            with cv1:
                fc = vol_mod.generate_multi_step_volatility_forecast(res_obj, horizons=[1, 5, 20])
                cv1.metric("Tomorrow's Volatility", f"{fc['Forecast_1d_Vol_Pct']:.1f}%",
                           help="Annualized 1-day ahead conditional volatility forecast")
                cv1.metric("5-Day Volatility", f"{fc['Forecast_5d_Vol_Pct']:.1f}%")
            with cv2:
                cv2.metric("20-Day Volatility", f"{fc['Forecast_20d_Vol_Pct']:.1f}%")
                cv2.metric("Long-Run Volatility", f"{row['Long_Run_Vol_Pct']:.1f}%",
                           help="Unconditional long-run volatility implied by model parameters")
            with cv3:
                cv3.metric("Volatility Persistence", f"{row['Persistence']:.3f}",
                           help="Close to 1 means shocks decay slowly (long memory)")
                half_life = row.get("Half_Life_Days", "—")
                cv3.metric("Shock Half-Life", f"{half_life:.0f}d" if isinstance(half_life, (int, float)) and not pd.isna(half_life) else "—",
                           help="Days until a volatility shock decays by 50%")
