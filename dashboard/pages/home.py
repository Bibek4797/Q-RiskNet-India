"""
Q-RiskNet India — Home Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st


def render_page():
    """Renders the landing / overview page."""

    st.markdown("<h1 style='text-align:center;'>🇮🇳 Q-RiskNet India</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#94a3b8;font-size:1.15rem;'>"
                "Enterprise Quantitative Finance &amp; Systemic Risk Research Platform</p>",
                unsafe_allow_html=True)
    st.caption("Analysing dynamic risk spillovers, connectedness, and network topology "
               "in the Indian equity market using Quantile VAR, GARCH, and deep learning.")

    st.markdown("---")

    # ── Key Capabilities ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏗️ Research Phases", "10")
    c2.metric("📊 NSE Sectors", "10")
    c3.metric("🧪 Statistical Tests", "12+")
    c4.metric("🤖 Forecast Models", "7")

    st.markdown("---")

    # ── Pipeline Status ───────────────────────────────────────────────
    st.subheader("📋 Research Pipeline Completion Status")
    phases = [
        ("1", "Repository Architecture"),
        ("2", "Financial Data Engineering"),
        ("3", "Econometric Diagnostics"),
        ("4", "Volatility Modelling"),
        ("5", "Quantile VAR (QVAR)"),
        ("6", "Dynamic Connectedness"),
        ("7", "Financial Network Science"),
        ("8", "Forecasting Benchmark"),
        ("9", "Research Validation"),
        ("10", "Enterprise Analytics Platform"),
    ]
    cols = st.columns(5)
    for idx, (num, name) in enumerate(phases):
        with cols[idx % 5]:
            st.success(f"**Phase {num}**")
            st.caption(name)

    st.markdown("---")

    # ── Core Research Questions ────────────────────────────────────────
    st.subheader("🔬 Core Research Questions")
    st.markdown("""
| # | Hypothesis | Methodology |
|:-:|:---|:---|
| $H_1$ | Tail risk connectedness ($\\tau=0.05$) significantly exceeds median ($\\tau=0.50$) connectedness. | QVAR + Diebold-Yilmaz GFEVD |
| $H_2$ | Negative equity shocks trigger disproportionately higher volatility (leverage effect). | GJR-GARCH / EGARCH asymmetric $\\gamma$ |
| $H_3$ | Nifty Bank / Financial Services is the persistent net systemic risk transmitter. | Network PageRank + Out-Degree Centrality |
    """)

    st.markdown("---")

    # ── Methodology Overview ──────────────────────────────────────────
    m1, m2 = st.columns(2)
    with m1:
        st.subheader("📐 Econometric Framework")
        st.markdown("""
- **Stationarity**: ADF, KPSS, Zivot-Andrews
- **Volatility**: ARCH, GARCH, EGARCH, GJR-GARCH
- **Dependence**: Quantile VAR (7 quantiles)
- **Connectedness**: Diebold-Yilmaz GFEVD / TCI
- **Network**: Centrality, spectral clustering, MST
        """)
    with m2:
        st.subheader("🤖 Predictive Framework")
        st.markdown("""
- **Baselines**: Random Walk, Historical Mean
- **Classical**: ARIMA(1,0,1)
- **Machine Learning**: Random Forest, Gradient Boosting, SVR
- **Deep Learning**: PyTorch Quantile LSTM
- **Evaluation**: RMSE, MAE, DA%, Diebold-Mariano test
        """)

    st.markdown("---")
    st.caption("© 2026 Bibek Rout  ·  MIT License  ·  Q-RiskNet India v1.0.0")
