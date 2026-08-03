"""
Q-RiskNet India — About Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st


def render_page():
    """Renders the About / Documentation page."""

    st.header("ℹ️ About Q-RiskNet India")

    st.markdown("""
**Q-RiskNet India** is an enterprise quantitative finance research platform for
analysing dynamic risk spillovers, systemic risk transmission, financial
connectedness, and network topology in the Indian equity market.

The platform integrates classical econometrics, modern machine learning, and
financial network science into a unified, reproducible research environment.
    """)

    st.markdown("---")

    st.subheader("👤 Author")
    st.markdown("""
- **Name**: Bibek Rout
- **License**: MIT License
- **Repository**: [github.com/Bibek4797/Q-RiskNet-India](https://github.com/Bibek4797/Q-RiskNet-India)
    """)

    st.markdown("---")

    st.subheader("📚 Methodology References")
    st.markdown("""
| Reference | Domain |
|:---|:---|
| Diebold & Yilmaz (2012, 2014) | Generalised Forecast Error Variance Decomposition (GFEVD) spillover index |
| Glosten, Jagannathan & Runkle (1993) | GJR-GARCH asymmetric leverage effect |
| Nelson (1991) | Exponential GARCH (EGARCH) |
| Bouri et al. (2021) | Quantile connectedness in financial markets |
| Koenker & Bassett (1978) | Quantile regression foundations |
| Bollerslev (1986) | Generalised Autoregressive Conditional Heteroskedasticity |
    """)

    st.markdown("---")

    st.subheader("🛠️ Technology Stack")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**Core**
- Python 3.10+
- Streamlit
- Pandas / NumPy / SciPy
        """)
    with c2:
        st.markdown("""
**Econometrics**
- statsmodels
- arch (GARCH)
- scikit-learn
        """)
    with c3:
        st.markdown("""
**Deep Learning & Viz**
- PyTorch
- Plotly
- NetworkX
        """)

    st.markdown("---")
    st.subheader("📄 Documentation Index")
    docs = [
        "docs/Econometric_Methodology.md",
        "docs/Volatility_Modelling_Methodology.md",
        "docs/QVAR_Methodology.md",
        "docs/Connectedness_Methodology.md",
        "docs/Network_Science_Methodology.md",
        "docs/Forecasting_Methodology.md",
        "docs/Research_Validation_Framework.md",
        "docs/Dashboard_Architecture.md",
    ]
    for d in docs:
        st.markdown(f"- `{d}`")

    st.markdown("---")
    st.caption("© 2026 Bibek Rout  ·  MIT License  ·  Q-RiskNet India v1.0.0")
