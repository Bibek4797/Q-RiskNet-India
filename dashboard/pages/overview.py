"""
Q-RiskNet India — Overview Section
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
import pandas as pd

import src.models.qvar as qvar
import src.forecasting.girf as girf
from dashboard.components.kpi_cards import render_kpi_cards
from dashboard.components.charts import render_spillover_charts, render_rolling_tci_chart


@st.cache_data(show_spinner=False)
def _compute_baseline_spillovers(returns_tuple, lags, horizon):
    """Cache baseline spillover computation to avoid re-running on every render."""
    import pandas as pd
    returns_df = pd.DataFrame(list(returns_tuple[1]), index=returns_tuple[0], columns=returns_tuple[2])
    m = qvar.QVARModel(p=lags, quantile=0.50)
    m.fit(returns_df)
    spill = girf.compute_spillover_matrix(m, returns_df, horizon=horizon)
    met = girf.calculate_connectedness_metrics(spill)
    return spill, met


def render_page(returns_df, cfg):
    """Renders the executive Overview page."""

    # ── Page Title ────────────────────────────────────────────────────
    st.markdown("""
    <div style='margin-bottom: 4px;'>
        <span style='font-family:"Space Grotesk",sans-serif; font-size:2rem; font-weight:700;
                     color:#e2e8f0;'>Q-RiskNet India</span>
    </div>
    <div style='font-size:0.9rem; color:#64748b; margin-bottom:20px;'>
        Systemic Risk Analytics for Indian Equity Sectors
    </div>
    """, unsafe_allow_html=True)

    if returns_df is None or len(returns_df.columns) < 2:
        st.info("Select at least 2 sectors in the sidebar to view systemic risk metrics.")
        return

    # ── Auto-compute baseline spillovers (cached) ─────────────────────
    metrics = st.session_state.get("metrics")
    spill_df = st.session_state.get("spillover_df")

    if metrics is None or spill_df is None:
        with st.spinner("Computing systemic risk metrics…"):
            try:
                # Use hashable args for caching
                idx = tuple(returns_df.index)
                vals = [tuple(row) for row in returns_df.values]
                cols = tuple(returns_df.columns)
                spill_df, metrics = _compute_baseline_spillovers(
                    (idx, vals, cols),
                    cfg.get("lags", 2),
                    cfg.get("forecast_horizon", 10)
                )
                st.session_state["spillover_df"] = spill_df
                st.session_state["metrics"] = metrics
            except Exception as e:
                st.warning(f"Could not compute baseline spillovers: {str(e)}")

    # ── KPI Cards ─────────────────────────────────────────────────────
    if metrics is not None:
        render_kpi_cards(metrics)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("### Where is risk flowing?")
        st.caption("Net risk flow across sectors — positive means transmitting more risk than receiving.")
        render_spillover_charts(metrics)

    # ── How to Read This ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### How to read this")

    items = [
        ("Higher TCI → more interconnected markets",
         "When the Total Connectedness Index is high, a shock in one sector spreads more easily to others. Above 65% signals elevated systemic risk."),
        ("Positive Net Risk Flow → sector transmits risk",
         "Sectors with a positive net flow export more volatility than they absorb. They act as sources of systemic stress."),
        ("Negative Net Risk Flow → sector absorbs risk",
         "Sectors with a negative net flow receive more volatility than they transmit. They act as buffers or shock absorbers."),
        ("Higher volatility → larger expected price swings",
         "Annualized volatility measures the magnitude of daily return fluctuations. Higher values indicate greater uncertainty."),
    ]
    for title, body in items:
        with st.expander(title):
            st.caption(body)

    # ── Methodology (fully collapsed) ─────────────────────────────────
    with st.expander("View methodology"):
        st.markdown("""
**Multi-Quantile VAR Framework** — Estimates equation-by-equation quantile regressions at
τ ∈ {0.05, 0.50, 0.95} to capture risk transmission under tail (crisis), median (normal),
and upper-tail (rally) market conditions.

**Simulation-Based Connectedness** — Uses Generalized Impulse Response Function (GIRF)
simulations to compute non-linear spillover percentages inspired by the Diebold-Yılmaz
forecast error variance decomposition approach.

**Asymmetric Volatility (GJR-GARCH)** — Captures the leverage effect where negative
return shocks generate disproportionately larger volatility increases than positive shocks.

**Financial Network Science** — Constructs directed risk graphs, Minimum Spanning Trees
(MST), and spectral community clusters to identify systemic hubs and risk backbone.

**Quantile LSTM** — A PyTorch sequence model trained under Pinball Loss with
chronological train/validation split and early stopping on out-of-sample pinball loss.
        """)
