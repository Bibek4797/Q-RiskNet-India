"""
Q-RiskNet India — Connectedness & Spillover Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd

import src.models.qvar as qvar
import src.models.quantile_lstm as qlstm
import src.forecasting.girf as girf
import src.diagnostics.logger as diag
from dashboard.components.kpi_cards import render_kpi_cards
from dashboard.components.charts import render_spillover_charts, render_rolling_tci_chart
from dashboard.components.tables import render_spillover_matrix_table
from dashboard.components.exports import download_csv


def render_page(model_input, returns_df, cfg):
    """Renders the Connectedness & Spillover page."""

    st.header("🌊 Dynamic Connectedness & Systemic Risk Transmission")
    st.caption("Diebold-Yilmaz GFEVD-based directional spillovers, Total Connectedness Index (TCI), "
               "and dynamic rolling-window TCI analysis.")

    # Use central sidebar configuration directly to prevent duplicate widgets
    mcfg = cfg

    conn_tab, tci_tab = st.tabs(["📊 Static Connectedness", "🕒 Dynamic Rolling TCI"])

    # ── Static Connectedness ──────────────────────────────────────────
    with conn_tab:
        st.subheader(f"Spillover Analysis ({mcfg['model_choice']} at τ={mcfg['quantile']:.2f})")

        run_btn = st.button("🚀 Calculate Spillovers", type="primary", key="run_spill")
        if run_btn or st.session_state.get("spillover_df") is not None:
            if run_btn:
                try:
                    progress = st.progress(0, text="Fitting model…")

                    def _progress(c, t):
                        progress.progress(int(c / t * 100), f"Epoch {c}/{t}…")

                    if mcfg["model_choice"] == "Quantile VAR (QVAR)":
                        m = qvar.QVARModel(p=mcfg["lags"], quantile=mcfg["quantile"])
                        m.fit(model_input)
                    else:
                        m = qlstm.LSTMQuantileModel(
                            seq_len=mcfg["seq_len"], hidden_dim=mcfg["hidden_dim"],
                            quantile=mcfg["quantile"], epochs=mcfg["epochs"],
                            early_stopping=True, patience=5)
                        m.fit(model_input, progress_callback=_progress)

                    progress.progress(100, "Computing GIRF spillovers…")
                    spill = girf.compute_spillover_matrix(m, model_input, horizon=mcfg["forecast_horizon"])
                    met = girf.calculate_connectedness_metrics(spill)
                    st.session_state["spillover_df"] = spill
                    st.session_state["metrics"] = met
                    progress.empty()
                except Exception as e:
                    diag.log_error("Model fitting failure", e)
                    st.error(f"❌ Error: {str(e)}")
                    return

            spill_df = st.session_state.get("spillover_df")
            metrics = st.session_state.get("metrics")

            if spill_df is not None and metrics is not None:
                render_kpi_cards(metrics)
                st.markdown("---")
                render_spillover_charts(metrics)
                st.subheader("Diebold-Yilmaz Spillover Matrix (%)")
                render_spillover_matrix_table(spill_df, metrics)

                with st.expander("📥 Export Spillover Matrix"):
                    download_csv(spill_df, "spillover_matrix.csv", key="dl_spill")

    # ── Dynamic Rolling TCI ───────────────────────────────────────────
    with tci_tab:
        st.subheader("🕒 Dynamic Time-Varying TCI")
        cr1, cr2 = st.columns(2)
        with cr1:
            win = st.slider("Window Size (days)", 60, 500, 200, 20)
        with cr2:
            step = st.slider("Step Size (days)", 5, 60, 20, 5)

        run_roll = st.button("🔄 Compute Rolling TCI", key="run_roll")
        if run_roll:
            if len(model_input) < win:
                st.error(f"Data length ({len(model_input)}) < window ({win}). Reduce window.")
            else:
                dates, tci_vals = [], []
                total = (len(model_input) - win) // step + 1
                bar = st.progress(0, "Rolling TCI…")
                for idx, i in enumerate(range(0, len(model_input) - win + 1, step)):
                    sub = model_input.iloc[i:i + win]
                    try:
                        rm = qvar.QVARModel(p=mcfg["lags"], quantile=mcfg["quantile"])
                        rm.fit(sub)
                        rs = girf.compute_spillover_matrix(rm, sub, horizon=mcfg["forecast_horizon"])
                        rm_met = girf.calculate_connectedness_metrics(rs)
                        dates.append(sub.index[-1])
                        tci_vals.append(rm_met["TCI"])
                    except Exception:
                        pass
                    bar.progress(int((idx + 1) / total * 100), f"Window {idx + 1}/{total}")
                bar.empty()
                if dates:
                    roll_df = pd.DataFrame({"Date": dates, "Rolling TCI (%)": tci_vals}).set_index("Date")
                    render_rolling_tci_chart(roll_df, win, step)
                    download_csv(roll_df, "rolling_tci.csv", key="dl_rtci")
