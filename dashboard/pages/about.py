"""
Q-RiskNet India — Premium About Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st


def render_page():
    """Renders the premium About / Documentation page."""

    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(99,102,241,0.1),rgba(167,139,250,0.05));
                border:1px solid rgba(99,102,241,0.25);border-radius:14px;padding:20px 24px;
                margin-bottom:24px;'>
        <div style='font-family:"Space Grotesk",sans-serif;font-size:1.35rem;font-weight:700;
                    color:#c7d2fe;margin-bottom:8px;'>About Q-RiskNet India</div>
        <div style='color:#94a3b8;font-size:0.93rem;line-height:1.7;'>
            An enterprise quantitative finance platform for analysing <strong style='color:#c7d2fe;'>
            dynamic risk spillovers</strong>, <strong style='color:#c7d2fe;'>systemic connectedness</strong>,
            and <strong style='color:#c7d2fe;'>financial network topology</strong> in Indian equity markets.
            Integrating classical econometrics, deep learning, and network science into a unified,
            reproducible research environment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Author ────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(99,102,241,0.2);
                border-radius:12px;padding:18px 22px;margin-bottom:16px;'>
        <div style='display:flex;align-items:center;gap:16px;'>
            <div style='font-size:3rem;'>👤</div>
            <div>
                <div style='font-family:"Space Grotesk",sans-serif;font-size:1.1rem;font-weight:700;
                            color:#e2e8f0;'>Bibek Rout</div>
                <div style='color:#64748b;font-size:0.83rem;margin-top:2px;'>Author &amp; Maintainer</div>
                <div style='margin-top:10px;'>
                    <a href='https://github.com/Bibek4797/Q-RiskNet-India' target='_blank'
                       style='background:rgba(99,102,241,0.18);border:1px solid rgba(99,102,241,0.35);
                              border-radius:6px;padding:4px 12px;color:#a5b4fc;font-size:0.8rem;
                              font-weight:600;text-decoration:none;margin-right:8px;'>
                        🔗 GitHub Repo
                    </a>
                    <span style='font-size:0.78rem;color:#475569;'>MIT License &nbsp;·&nbsp; v1.0.0</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:0.1em;margin:20px 0 12px;'>Technology Stack</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    tech_groups = [
        ("🐍 Core", [("Python 3.10+", "Runtime"), ("Streamlit", "Dashboard"), ("Pandas / NumPy / SciPy", "Data")]),
        ("📐 Econometrics", [("statsmodels", "Time Series"), ("arch", "GARCH Family"), ("scikit-learn", "ML / SVR")]),
        ("🤖 DL & Viz", [("PyTorch", "Quantile LSTM"), ("Plotly", "Charting"), ("NetworkX", "Graph Analysis")]),
    ]
    for col, (title, items) in zip([c1, c2, c3], tech_groups):
        with col:
            rows = "".join(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:7px 0;border-bottom:1px solid rgba(99,102,241,0.1);'>"
                f"<span style='color:#c7d2fe;font-weight:500;font-size:0.85rem;'>{name}</span>"
                f"<span style='color:#475569;font-size:0.75rem;'>{role}</span></div>"
                for name, role in items
            )
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(99,102,241,0.18);
                        border-radius:12px;padding:16px 18px;'>
                <div style='font-size:0.75rem;font-weight:700;color:#6366f1;text-transform:uppercase;
                            letter-spacing:0.09em;margin-bottom:10px;'>{title}</div>
                {rows}
            </div>
            """, unsafe_allow_html=True)

    # ── References ────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:0.1em;margin:24px 0 12px;'>Methodology References</div>
    """, unsafe_allow_html=True)

    refs = [
        ("Diebold & Yilmaz (2012, 2014)", "Generalised FEVD spillover index (TCI)"),
        ("Glosten, Jagannathan & Runkle (1993)", "GJR-GARCH asymmetric leverage effect"),
        ("Nelson (1991)", "Exponential GARCH (EGARCH)"),
        ("Bouri et al. (2021)", "Quantile connectedness in financial markets"),
        ("Koenker & Bassett (1978)", "Quantile regression foundations"),
        ("Bollerslev (1986)", "Generalised ARCH (GARCH)"),
    ]
    for author, desc in refs:
        st.markdown(f"""
        <div style='display:flex;align-items:flex-start;gap:12px;padding:9px 0;
                    border-bottom:1px solid rgba(99,102,241,0.08);'>
            <div style='min-width:6px;height:6px;background:#6366f1;border-radius:50%;
                        margin-top:6px;flex-shrink:0;'></div>
            <div>
                <span style='color:#c7d2fe;font-weight:600;font-size:0.85rem;'>{author}</span>
                <span style='color:#64748b;font-size:0.82rem;'> — {desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Docs ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style='font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;
                letter-spacing:0.1em;margin:24px 0 12px;'>Documentation Index</div>
    """, unsafe_allow_html=True)

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
    cols = st.columns(2)
    for i, d in enumerate(docs):
        with cols[i % 2]:
            st.markdown(
                f"<div style='font-size:0.82rem;color:#64748b;padding:4px 0;'>"
                f"<span style='color:#4f46e5;margin-right:6px;'>📄</span>{d}</div>",
                unsafe_allow_html=True
            )

    st.markdown("<div class='qrn-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;font-size:0.76rem;color:#334155;padding:4px 0 12px;'>"
        "© 2026 <strong style='color:#475569;'>Bibek Rout</strong> &nbsp;·&nbsp; "
        "Q-RiskNet India v1.0.0 &nbsp;·&nbsp; MIT License"
        "</div>",
        unsafe_allow_html=True
    )
