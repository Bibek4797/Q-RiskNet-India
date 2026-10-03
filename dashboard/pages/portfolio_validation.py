"""
Q-RiskNet India — Model Benchmark & Validation Section
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd
import plotly.express as px

import src.portfolio.backtest as p_bt
import src.portfolio.stress_test as p_st
import src.econometrics.tail_risk as tail_risk
import src.forecasting.evaluator as evaluator
import src.diagnostics.validation_runner as val_runner
from dashboard.components.charts import render_forecast_benchmark_chart, _render_plotly
from dashboard.components.exports import download_csv

# ==============================================================================
# FEATURE FLAGS (Controlled for strict CV alignment)
# (Underlying algorithms in src/ remain 100% intact. Set to True to restore in UI)
# ==============================================================================
SHOW_PORTFOLIO_OPTIMIZATION = False
SHOW_VAR_BACKTESTING = False
SHOW_ROBUSTNESS_ANALYSIS = False

# ── Stress scenario labels ────────────────────────────────────────────────────
_STRESS_LABELS = {
    "Broad_Market_Crash": "Broad Market Crash\n(-15% equity shock)",
    "Volatility_Spike": "High-Volatility Spike\n(2.5σ return shock)",
    "Banking_Crisis": "Systemic Banking Crisis\n(-25% bank sector shock)",
}


def _render_portfolio_section(returns_df):
    """Renders the Portfolio Risk & Optimization sub-tab."""
    with st.expander("Portfolio assumptions", expanded=False):
        st.markdown("""
- **Equal Weight** — Baseline: equal allocation across all selected sectors (1/K).
- **Minimum Variance (MPT)** — Markowitz mean-variance optimization. Minimizes portfolio return variance subject to long-only constraints.
- **Risk Parity (ERC)** — Equal Risk Contribution. Each sector contributes an identical fraction of total portfolio volatility.
- **CVaR Optimization** — Minimizes the average expected loss beyond the 95th percentile (Expected Shortfall).

All strategies are subject to: each weight >= 0, sum of weights = 1, max single position <= w_max.
        """)

    max_w = st.slider(
        "Max sector position (w_max)",
        0.20, 1.00, 0.40, 0.05,
        help="Maximum weight allowed for any single sector"
    )
    run_p_bt = st.button("Run Portfolio Optimization & Backtest", type="primary", key="run_p_bt")

    if run_p_bt:
        with st.spinner("Optimizing portfolios and running backtest…"):
            try:
                bt_res = p_bt.run_portfolio_backtest(returns_df, max_weight=max_w)
                st.session_state["portfolio_backtest_res"] = bt_res
            except Exception as e:
                st.error(f"Portfolio optimization error: {str(e)}")

    bt_res = st.session_state.get("portfolio_backtest_res")

    if bt_res is not None:
        st.markdown("---")
        st.markdown("**Strategy Comparison**")
        summary = bt_res["summary_df"]

        col_rename = {
            "Annualized_Return_Pct": "Return (% p.a.)",
            "Annualized_Vol_Pct": "Volatility (% p.a.)",
            "Sharpe_Ratio": "Sharpe Ratio",
            "Max_Drawdown_Pct": "Max Drawdown (%)",
            "VaR_95_Pct": "VaR 95% (%)",
            "CVaR_95_Pct": "Expected Shortfall 95% (%)"
        }
        friendly_summary = summary.rename(columns=col_rename)

        try:
            styled = friendly_summary.style \
                .highlight_max(subset=["Return (% p.a.)", "Sharpe Ratio"], color="#10b981") \
                .highlight_min(subset=["Volatility (% p.a.)", "Max Drawdown (%)", "VaR 95% (%)", "Expected Shortfall 95% (%)"], color="#6366f1")
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(friendly_summary, use_container_width=True)

        c_eq1, c_eq2 = st.columns([4, 1])
        with c_eq1:
            st.markdown("**Cumulative Growth (Base = 100)**")
            st.caption("Out-of-sample backtest with monthly rebalancing.")
        with c_eq2:
            if st.button("↺ Reset view", key="btn_reset_equity_curves"):
                st.session_state["key_equity_curves"] = st.session_state.get("key_equity_curves", 0) + 1

        fig_eq = px.line(
            bt_res["equity_df"],
            x=bt_res["equity_df"].index,
            y=bt_res["equity_df"].columns,
            labels={"value": "Portfolio Value (Base 100)", "variable": "Strategy"},
            title="Cumulative Growth (Base = 100)"
        )
        _render_plotly(fig_eq, height=400, key=f"eq_curves_{st.session_state.get('key_equity_curves', 0)}")

        st.markdown("**Sector Weights by Strategy (%)**")
        weights_df = pd.DataFrame(bt_res["weights_dict"]) * 100.0
        fig_w = px.bar(
            weights_df,
            barmode="group",
            title="Sector Weights by Strategy (%)",
            labels={"value": "Weight (%)", "variable": "Strategy"}
        )
        _render_plotly(fig_w, height=360)

        st.markdown("---")
        st.markdown("**Stress Test Scenarios**")
        st.caption(
            "Estimated portfolio loss under three hypothetical market shocks. "
            "Results are scenario-based estimates, not forecasts."
        )

        with st.spinner("Running stress scenarios…"):
            try:
                st_df = p_st.run_portfolio_stress_test(bt_res["weights_dict"], returns_df)
                st.dataframe(st_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Stress test error: {str(e)}")

        with st.expander("Download results"):
            download_csv(bt_res["summary_df"], "portfolio_performance_comparison.csv", key="dl_p_comp")

    elif not run_p_bt:
        st.info("Click **Run Portfolio Optimization & Backtest** to compare allocation strategies.")


def _render_forecast_benchmark_section(returns_df):
    """Renders the PyTorch Quantile LSTM vs ARIMA & SVR Walk-Forward Benchmark."""
    st.markdown("**Walk-Forward Model Benchmark (PyTorch Quantile LSTM vs ARIMA & SVR)**")
    st.caption("Chronological expanding-window walk-forward evaluation (no look-ahead bias). Benchmarks deep-learning tail-risk modeling against classical econometric and ML baselines.")

    sec = st.selectbox(
        "Target sector",
        list(returns_df.columns),
        key="pv_fc_sec",
        label_visibility="visible"
    )
    run_fc_btn = st.button("Run Walk-Forward Benchmark", type="primary", key="run_pv_fc")

    if run_fc_btn:
        with st.spinner(f"Evaluating models on expanding windows for {sec}…"):
            try:
                fc_res = evaluator.run_all_forecast_benchmarks(
                    returns_df, target_sector=sec, train_ratio=0.80, save_reports=True
                )
                st.session_state["pv_fc_results"] = fc_res
                st.session_state["pv_fc_sec"] = sec
            except Exception as e:
                st.error(f"Forecast evaluation error: {str(e)}")

    fc_res = st.session_state.get("pv_fc_results")
    target_sec = st.session_state.get("pv_fc_sec", sec)

    if fc_res is not None:
        st.markdown(f"*Out-of-sample benchmark results for {target_sec}*")

        summary_display = fc_res["summary_df"].copy()
        if "Directional_Accuracy_Pct" in summary_display.columns:
            summary_display = summary_display.drop(columns=["Directional_Accuracy_Pct"])

        col_rename = {
            "RMSE": "RMSE",
            "MAE": "MAE",
            "Pinball_Loss": "Pinball Loss (quantile)"
        }
        friendly_fc = summary_display.rename(columns=col_rename)

        try:
            styled_fc = friendly_fc.style \
                .highlight_min(subset=["RMSE", "MAE", "Pinball Loss (quantile)"], color="#10b981")
            st.dataframe(styled_fc, use_container_width=True)
        except Exception:
            st.dataframe(friendly_fc, use_container_width=True)

        render_forecast_benchmark_chart(fc_res["predictions_df"], target_sec)

        with st.expander("Diebold-Mariano test (vs naive random walk)"):
            st.caption("Tests whether each model's forecast errors are statistically different from a naive random walk. Negative DM statistic = model outperforms random walk.")
            st.dataframe(fc_res["dm_df"], use_container_width=True)

    elif not run_fc_btn:
        st.info("Select a sector and click **Run Walk-Forward Benchmark** to evaluate PyTorch Quantile LSTM against ARIMA and SVR.")


def _render_var_backtest_section(returns_df):
    """Renders the Kupiec and Christoffersen VaR Backtesting suite."""
    st.markdown("**VaR Backtesting**")
    st.caption(
        "Verifies whether the number of VaR exceptions matches theoretical frequency. "
        "Kupiec tests exception counts; Christoffersen tests whether exceptions cluster in time."
    )

    run_var_btn = st.button("Run VaR Backtests", key="run_var_bt")
    if run_var_btn:
        with st.spinner("Running VaR backtests…"):
            try:
                tr_df = tail_risk.run_full_tail_risk_suite(returns_df, confidence_levels=[0.95, 0.99])
                st.session_state["var_backtest_res"] = tr_df
            except Exception as e:
                st.error(f"VaR backtest error: {str(e)}")

    tr_df = st.session_state.get("var_backtest_res")
    if tr_df is not None:
        st.dataframe(tr_df, use_container_width=True)
        with st.expander("Download VaR results"):
            download_csv(tr_df, "var_backtesting_results.csv", key="dl_tr_res")
    elif not run_var_btn:
        st.info("Click **Run VaR Backtests** to validate historical Value-at-Risk exceptions.")


def _render_robustness_section(returns_df):
    """Renders parameter robustness sensitivity tables."""
    st.markdown("**Robustness Analysis**")
    st.caption("Evaluates stability when rolling window length (W) and forecast horizon (H) are varied.")

    run_sens_btn = st.button("Run Robustness Analysis", key="run_sens")
    if run_sens_btn:
        with st.spinner("Evaluating parameter sensitivity…"):
            try:
                sens_res = val_runner.run_master_validation_suite(returns_df, save_reports=True)
                st.session_state["sens_results"] = sens_res
            except Exception as e:
                st.error(f"Robustness analysis error: {str(e)}")

    sens_res = st.session_state.get("sens_results")
    if sens_res is not None:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("*Window sensitivity (W)*")
            st.dataframe(sens_res.get("window_df"), use_container_width=True)
        with r2:
            st.markdown("*Horizon sensitivity (H)*")
            st.dataframe(sens_res.get("horizon_df"), use_container_width=True)
    elif not run_sens_btn:
        st.info("Click **Run Robustness Analysis** to test parameter stability.")


def render_page(returns_df, cfg):
    """Renders the Model Benchmark & Validation section."""
    if SHOW_PORTFOLIO_OPTIMIZATION:
        st.markdown("### Portfolio Risk & Validation")
        st.caption("Compare allocation strategies, evaluate model forecast accuracy, and stress-test portfolios under adverse scenarios.")
        tab_port, tab_val = st.tabs(["Portfolio Risk", "Model Validation"])
        with tab_port:
            _render_portfolio_section(returns_df)
        with tab_val:
            _render_forecast_benchmark_section(returns_df)
            if SHOW_VAR_BACKTESTING:
                st.markdown("---")
                _render_var_backtest_section(returns_df)
            if SHOW_ROBUSTNESS_ANALYSIS:
                st.markdown("---")
                _render_robustness_section(returns_df)
    else:
        st.markdown("### 🤖 Model Benchmark")
        st.caption("Developed a PyTorch Quantile LSTM under Pinball Loss and benchmarked it against ARIMA(1,0,1) and Support Vector Regression (SVR) across chronological walk-forward windows.")
        _render_forecast_benchmark_section(returns_df)

        if SHOW_VAR_BACKTESTING:
            st.markdown("---")
            _render_var_backtest_section(returns_df)

        if SHOW_ROBUSTNESS_ANALYSIS:
            st.markdown("---")
            _render_robustness_section(returns_df)
