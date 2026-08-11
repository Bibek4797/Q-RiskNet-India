"""
Q-RiskNet India — Connectedness & Systemic Risk Flow Section
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd

import src.models.qvar as qvar
import src.forecasting.girf as girf
import src.econometrics.garch as garch
import src.diagnostics.logger as diag
from dashboard.components.kpi_cards import render_kpi_cards
from dashboard.components.charts import render_spillover_charts, render_rolling_tci_chart
from dashboard.components.tables import render_spillover_matrix_table
from dashboard.components.exports import download_csv


# ── Regime label helper ───────────────────────────────────────────────────────
def _regime_label(q: float) -> str:
    if q <= 0.10:
        return "Downside / tail-risk conditions"
    elif q <= 0.25:
        return "Below-median market conditions"
    elif q <= 0.40:
        return "Mild-downside market conditions"
    elif q <= 0.60:
        return "Normal / median market conditions"
    elif q <= 0.75:
        return "Mild-upside market conditions"
    elif q <= 0.90:
        return "Above-median market conditions"
    else:
        return "Bullish / upper-tail market conditions"


def render_page(model_input, returns_df, cfg):
    """Renders the Connectedness section with user-friendly risk flow labels."""

    st.markdown("### Systemic Risk Flow & Connectedness")
    st.caption("Measure how much risk each sector transmits to or receives from others. High connectedness signals that shocks will spread rapidly across the market.")

    # ── Show cached results if available (no forced rerun) ─────────────
    metrics = st.session_state.get("metrics")
    spill_df = st.session_state.get("spillover_df")

    # Display active model badge if exists
    active_label = st.session_state.get("active_model_label")
    if active_label:
        st.caption(f"Showing: {active_label}")

    # KPI summary (from cached results or after computation)
    if metrics is not None:
        render_kpi_cards(metrics)
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    conn_tab, tci_tab = st.tabs(["Spillover Map", "Connectedness Over Time"])

    # ── Tab 1: Spillover Map ──────────────────────────────────────────
    with conn_tab:
        # Controls — clean single row
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            vol_proxy = st.selectbox(
                "Risk input",
                ["Log Returns", "Conditional Volatility (GJR-GARCH)"],
                key="conn_vol_proxy",
                help="Choose whether to model spillovers in raw log-return space or in estimated conditional volatility space."
            )
        with c2:
            quantile = st.select_slider(
                "Market regime (τ)",
                options=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
                value=0.50,
                key="conn_tau"
            )
        with c3:
            run_btn = st.button("Run Analysis", type="primary", key="run_spill")

        st.caption(f"τ = {quantile:.2f} — {_regime_label(quantile)}. Lower quantiles capture tail/downside risk spillovers.")

        # Prepare input data
        if vol_proxy == "Conditional Volatility (GJR-GARCH)":
            garch_cols = {}
            for col in returns_df.columns:
                try:
                    garch_cols[col] = garch.estimate_garch_volatility(returns_df[col])
                except Exception:
                    garch_cols[col] = returns_df[col]
            active_input = pd.DataFrame(garch_cols, index=returns_df.index).dropna()
        else:
            active_input = returns_df.copy()

        # Run analysis
        if run_btn:
            try:
                with st.spinner("Fitting quantile model and computing spillovers…"):
                    m = qvar.QVARModel(p=cfg["lags"], quantile=quantile)
                    m.fit(active_input)
                    spill_df = girf.compute_spillover_matrix(m, active_input, horizon=cfg["forecast_horizon"])
                    metrics = girf.calculate_connectedness_metrics(spill_df)
                    st.session_state["spillover_df"] = spill_df
                    st.session_state["metrics"] = metrics
                    st.session_state["active_model_label"] = (
                        f"QVAR (τ={quantile:.2f}, {_regime_label(quantile)}, {vol_proxy})"
                    )
                st.rerun()
            except Exception as e:
                diag.log_error("Model fitting failure", e)
                st.error(f"Error during model fitting: {str(e)}")
                return

        # Display results
        if spill_df is not None and metrics is not None:
            st.markdown("---")

            # Risk flow charts
            render_spillover_charts(metrics)

            # Spillover matrix
            st.markdown("**Directional Risk Flow Matrix (%)**")
            st.caption("How much of each sector's forecast variance is explained by shocks from other sectors. Rows = receiving sector. Columns = transmitting sector.")
            render_spillover_matrix_table(spill_df, metrics)

            with st.expander("Download results"):
                download_csv(spill_df, "risk_spillover_matrix.csv", key="dl_spill")

            with st.expander("Advanced QVAR details"):
                st.markdown("""
**Method:** Equation-by-equation multi-quantile VAR framework. Each sector's returns are
regressed on lagged returns of all sectors at quantile τ using quantile regression.

**Spillover computation:** GIRF (Generalized Impulse Response Function) simulations apply a
+2σ shock to each transmitting sector and measure forecast error variance absorbed by each
receiving sector over the specified horizon H.

**Metric interpretation:**
- **TCI (Systemic Connectedness):** Fraction of the total forecast variance explained by
  cross-sector spillovers (higher = more interconnected).
- **Risk Transmitted:** Sum of all risk exported to other sectors.
- **Risk Received:** Sum of all risk imported from other sectors.
- **Net Risk Flow:** Transmitted minus Received. Positive = net transmitter.
                """)
        elif not run_btn:
            st.info("Click **Run Analysis** to compute directional risk spillovers.")

    # ── Tab 2: Connectedness Over Time ────────────────────────────────
    with tci_tab:
        st.caption("Shows how systemic connectedness changes over time. High periods indicate market stress when shocks spread across all sectors simultaneously.")

        cr1, cr2 = st.columns(2)
        with cr1:
            win = st.slider("Rolling window (trading days)", 60, 500, 200, 20,
                            help="Length of each estimation window")
        with cr2:
            step = st.slider("Step size (days)", 5, 60, 20, 5,
                             help="How many days the window advances at each step")

        run_roll = st.button("Compute Rolling Connectedness", key="run_roll")

        if run_roll:
            if returns_df is None or len(returns_df) < win:
                st.error(f"Not enough data ({len(returns_df) if returns_df is not None else 0} obs) for window size {win}. Reduce window size or increase date range.")
            else:
                dates, tci_vals = [], []
                total = max(1, (len(returns_df) - win) // step + 1)
                bar = st.progress(0, text="Computing rolling risk flow… (this may take a few minutes)")
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
                    bar.progress(int((idx + 1) / total * 100), f"Window {idx + 1} of {total}")
                bar.empty()
                if dates:
                    roll_df = pd.DataFrame({"Date": dates, "Rolling TCI (%)": tci_vals}).set_index("Date")
                    st.session_state["rolling_tci_df"] = roll_df
                    st.session_state["rolling_tci_win"] = win
                    st.session_state["rolling_tci_step"] = step
                else:
                    st.warning("No windows completed successfully. Try different parameters.")

        roll_df = st.session_state.get("rolling_tci_df")
        if roll_df is not None:
            c_rt1, c_rt2 = st.columns([4, 1])
            with c_rt1:
                st.markdown("**Rolling Systemic Connectedness (TCI)**")
            with c_rt2:
                if st.button("↺ Reset view", key="btn_reset_rolling_tci"):
                    st.session_state["key_rolling_tci"] = st.session_state.get("key_rolling_tci", 0) + 1

            win_val = st.session_state.get("rolling_tci_win", win)
            step_val = st.session_state.get("rolling_tci_step", step)
            chart_key = f"rolling_tci_{st.session_state.get('key_rolling_tci', 0)}"
            render_rolling_tci_chart(roll_df, win_val, step_val, key=chart_key)
            with st.expander("Download rolling TCI"):
                download_csv(roll_df, "rolling_systemic_connectedness.csv", key="dl_rtci")
