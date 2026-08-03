"""
Q-RiskNet India — Connectedness & Spillover Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd

import src.models.qvar as qvar
import src.models.quantile_lstm as qlstm
import src.forecasting.girf as girf
import src.econometrics.garch as garch
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

    conn_tab, tci_tab = st.tabs(["📊 Static Connectedness", "🕒 Dynamic Rolling TCI"])

    # ── Static Connectedness ──────────────────────────────────────────
    with conn_tab:
        st.subheader("⚙️ Spillover Model Controls")

        c1, c2, c3 = st.columns(3)
        with c1:
            model_choice = st.radio("Model Architecture", ["Quantile VAR (QVAR)", "Quantile LSTM"],
                                    horizontal=True, key="conn_model_choice")
        with c2:
            vol_proxy = st.selectbox("Data Series / Risk Proxy", ["Log Returns", "GARCH(1,1) Volatility"],
                                     key="conn_vol_proxy")
        with c3:
            quantile = st.slider("Quantile Value (τ)", 0.01, 0.99, 0.50, 0.01, key="conn_tau")

        # Compute volatility proxy data input if requested
        if vol_proxy == "GARCH(1,1) Volatility":
            garch_cols = {}
            for col in returns_df.columns:
                garch_cols[col] = garch.estimate_garch_volatility(returns_df[col])
            active_input = pd.DataFrame(garch_cols, index=returns_df.index).dropna()
        else:
            active_input = returns_df.copy()

        st.markdown(f"**Selected Model**: `{model_choice}` at $\\tau={quantile:.2f}$ using `{vol_proxy}`")

        run_btn = st.button("🚀 Calculate Spillovers", type="primary", key="run_spill")

        if run_btn or st.session_state.get("spillover_df") is not None:
            if run_btn:
                try:
                    progress = st.progress(0, text="Fitting model…")

                    def _progress(c, t):
                        progress.progress(int(c / t * 100), f"Epoch {c}/{t}…")

                    if model_choice == "Quantile VAR (QVAR)":
                        m = qvar.QVARModel(p=cfg["lags"], quantile=quantile)
                        m.fit(active_input)
                    else:
                        m = qlstm.LSTMQuantileModel(
                            seq_len=cfg.get("seq_len", 5),
                            hidden_dim=cfg.get("hidden_dim", "auto"),
                            quantile=quantile,
                            epochs=cfg.get("epochs", 30),
                            early_stopping=True, patience=5
                        )
                        m.fit(active_input, progress_callback=_progress)

                    progress.progress(100, "Computing GIRF spillovers…")
                    spill = girf.compute_spillover_matrix(m, active_input, horizon=cfg["forecast_horizon"])
                    met = girf.calculate_connectedness_metrics(spill)
                    st.session_state["spillover_df"] = spill
                    st.session_state["metrics"] = met
                    st.session_state["active_model_label"] = f"{model_choice} (τ={quantile:.2f}, {vol_proxy})"
                    progress.empty()
                except Exception as e:
                    diag.log_error("Model fitting failure", e)
                    st.error(f"❌ Error during model fitting: {str(e)}")
                    return

            spill_df = st.session_state.get("spillover_df")
            metrics = st.session_state.get("metrics")
            model_label = st.session_state.get("active_model_label", f"{model_choice} (τ={quantile:.2f})")

            if spill_df is not None and metrics is not None:
                st.markdown("---")
                st.info(f"Showing results for: **{model_label}**")
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
            if len(returns_df) < win:
                st.error(f"Data length ({len(returns_df)}) < window ({win}). Reduce window.")
            else:
                dates, tci_vals = [], []
                total = (len(returns_df) - win) // step + 1
                bar = st.progress(0, "Rolling TCI…")
                for idx, i in enumerate(range(0, len(returns_df) - win + 1, step)):
                    sub = returns_df.iloc[i:i + win]
                    try:
                        rm = qvar.QVARModel(p=cfg["lags"], quantile=0.50)
                        rm.fit(sub)
                        rs = girf.compute_spillover_matrix(rm, sub, horizon=cfg["forecast_horizon"])
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
