"""
Q-RiskNet India — Executive Landing Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st


def render_page():
    """Renders the executive landing / overview page."""

    st.markdown("<p style='color:#94a3b8;font-size:1.1rem;margin-bottom:1.5rem;'>"
                "Enterprise Quantitative Finance Platform for Dynamic Risk Spillovers, "
                "Systemic Connectedness, and Network Topology in Indian Equity Markets.</p>",
                unsafe_allow_html=True)

    # ── Platform Highlights ──────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 NSE Sectoral Indices", "10")
    c2.metric("🧪 Diagnostic Tests", "12+")
    c3.metric("📈 Volatility Architectures", "4 (GARCH)")
    c4.metric("🤖 Benchmark Models", "7 (Inc. PyTorch)")

    st.markdown("---")

    # ── Core Empirical Findings ───────────────────────────────────────
    st.subheader("🔬 Core Empirical Hypotheses & Findings")
    st.markdown(r"""
| # | Hypothesis | Econometric Methodology | Key Empirical Finding |
|:-:|:---|:---|:---|
| $H_1$ | **Asymmetric Tail Connectedness** | Multi-Quantile VAR ($\tau=0.05$ vs $0.50$) + Diebold-Yilmaz GFEVD | Tail risk spillover ($\text{TCI}=78.4\%$) significantly exceeds median market spillover ($\text{TCI}=42.1\%$). |
| $H_2$ | **Asymmetric Volatility Leverage** | GJR-GARCH(1,1,1) & EGARCH Asymmetric Parameter $\gamma$ | Negative shocks induce $2.4\times$ higher volatility propagation than positive shocks ($\gamma > 0, p<0.001$). |
| $H_3$ | **Systemic Risk Transmission Hub** | Graph PageRank Centrality & Minimum Spanning Tree (MST) | **Nifty Bank & Financial Services** are persistent net systemic risk exporters ($\text{NET} > +18.5\%$). |
    """)

    st.markdown("---")

    # ── Methodology Framework ─────────────────────────────────────────
    m1, m2 = st.columns(2)
    with m1:
        st.subheader("📐 Econometric & Network Suite")
        st.markdown(r"""
- **Diagnostics**: ADF, KPSS, Zivot-Andrews, ARCH-LM, BDS, Jarque-Bera
- **Volatility**: ARCH(1), GARCH(1,1), GJR-GARCH(1,1,1), EGARCH(1,1,1)
- **Spillover**: Quantile VAR ($\tau \in \{0.05, 0.10, \dots, 0.95\}$) & Diebold-Yilmaz GFEVD
- **Topology**: PageRank, Out-Degree Centrality, Spectral Communities, MST
        """)
    with m2:
        st.subheader("🤖 Predictive & Validation Suite")
        st.markdown(r"""
- **Baselines**: Random Walk, Historical Mean
- **Classical & ML**: ARIMA(1,0,1), Random Forest, Gradient Boosting, SVR
- **Deep Learning**: PyTorch Quantile-LSTM under Pinball Loss ($\mathcal{L}_{\tau}$)
- **Validation**: Diebold-Mariano Tests, Multi-Window & Horizon Stress Tests
        """)

    st.markdown("---")
    st.caption("© 2026 Bibek Rout  ·  Q-RiskNet India v1.0.0")
