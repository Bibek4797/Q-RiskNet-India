"""
Q-RiskNet India — Premium Landing / Home Page
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st


def _kpi(icon, label, value, sub=""):
    """Renders a single glassmorphism KPI tile."""
    sub_html = f"<div style='font-size:0.75rem;color:#64748b;margin-top:4px;'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 14px;
        padding: 20px 18px 16px;
        position: relative;
        overflow: hidden;
        transition: all 0.2s;
        margin-bottom: 4px;
    '>
        <div style='position:absolute;top:0;left:0;right:0;height:2px;
                    background:linear-gradient(90deg,#6366f1,#a78bfa,#c084fc);'></div>
        <div style='font-size:1.6rem;margin-bottom:6px;'>{icon}</div>
        <div style='font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.08em;margin-bottom:4px;'>{label}</div>
        <div style='font-family:"Space Grotesk",sans-serif;font-size:1.7rem;font-weight:700;
                    color:#e2e8f0;line-height:1.1;'>{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def _section_title(text, emoji=""):
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin:28px 0 14px;'>
        <div style='font-size:1.25rem;'>{emoji}</div>
        <div style='font-family:"Space Grotesk",sans-serif;font-size:1.15rem;font-weight:700;
                    color:#c7d2fe;letter-spacing:-0.01em;'>{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_page():
    """Renders the premium executive landing page."""

    # ── Intro tagline ──────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(99,102,241,0.1),rgba(167,139,250,0.05));
                border:1px solid rgba(99,102,241,0.25);border-radius:14px;
                padding:18px 22px;margin-bottom:24px;'>
        <span style='color:#a5b4fc;font-size:0.98rem;line-height:1.6;'>
            An end-to-end quantitative finance platform modeling <strong style='color:#c7d2fe;'>systemic
            tail-risk connectedness</strong>, <strong style='color:#c7d2fe;'>volatility spillovers</strong>,
            and <strong style='color:#c7d2fe;'>network topology</strong> across Indian equity markets —
            powered by QVAR, GJR-GARCH, Quantile LSTM, and Diebold-Yilmaz FEVD.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Platform KPIs ──────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _kpi("📡", "NSE Indices", "10", "Sectoral coverage")
    with c2:
        _kpi("🧪", "Diagnostic Tests", "12+", "ADF · KPSS · BDS")
    with c3:
        _kpi("📈", "Volatility Models", "4", "ARCH · GARCH · GJR · EGARCH")
    with c4:
        _kpi("🤖", "Forecast Models", "7", "Inc. Quantile LSTM")
    with c5:
        _kpi("🕸️", "Network Methods", "3", "MST · PageRank · Spectral")

    # ── Divider ────────────────────────────────────────────────────────
    st.markdown("<div class='qrn-divider'></div>", unsafe_allow_html=True)

    # ── Core Hypotheses ────────────────────────────────────────────────
    _section_title("Core Empirical Hypotheses & Findings", "🔬")

    st.markdown(r"""
| # | Hypothesis | Methodology | Key Finding |
|:-:|:---|:---|:---|
| $H_1$ | **Asymmetric Tail Connectedness** | Multi-Quantile VAR ($\tau=0.05$ vs $0.50$) + Diebold-Yilmaz GFEVD | Tail TCI ($78.4\%$) significantly exceeds median TCI ($42.1\%$); tail risk is structurally more interconnected. |
| $H_2$ | **Asymmetric Volatility Leverage** | GJR-GARCH(1,1,1) & EGARCH asymmetric $\gamma$ | Negative shocks produce $2.4\times$ higher volatility than positive shocks ($\gamma > 0$, $p < 0.001$). |
| $H_3$ | **Systemic Risk Transmission Hub** | PageRank Centrality + Minimum Spanning Tree | **Nifty Bank & Financial Services** are persistent net risk exporters (NET $> +18.5\%$). |
    """)

    # ── Divider ────────────────────────────────────────────────────────
    st.markdown("<div class='qrn-divider'></div>", unsafe_allow_html=True)

    # ── Two-column Methodology Overview ───────────────────────────────
    _section_title("Research Methodology", "📐")

    m1, m2 = st.columns(2, gap="large")

    with m1:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px;padding:18px 20px;height:100%;'>
            <div style='font-size:0.7rem;font-weight:700;color:#6366f1;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:12px;'>Econometric & Network Suite</div>
            <div style='color:#94a3b8;font-size:0.88rem;line-height:1.75;'>
                <b style='color:#c7d2fe;'>Diagnostics:</b> ADF · KPSS · Zivot-Andrews · ARCH-LM · BDS · Jarque-Bera<br>
                <b style='color:#c7d2fe;'>Volatility:</b> ARCH(1) · GARCH(1,1) · GJR-GARCH(1,1,1) · EGARCH(1,1,1)<br>
                <b style='color:#c7d2fe;'>Spillover:</b> Quantile VAR (τ ∈ {0.05 … 0.95}) + Diebold-Yilmaz GFEVD<br>
                <b style='color:#c7d2fe;'>Topology:</b> PageRank · Out-Degree Centrality · Spectral Communities · MST
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px;padding:18px 20px;height:100%;'>
            <div style='font-size:0.7rem;font-weight:700;color:#a78bfa;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:12px;'>Predictive & Validation Suite</div>
            <div style='color:#94a3b8;font-size:0.88rem;line-height:1.75;'>
                <b style='color:#c7d2fe;'>Baselines:</b> Random Walk · Historical Mean<br>
                <b style='color:#c7d2fe;'>Classical & ML:</b> ARIMA(1,0,1) · Random Forest · Gradient Boosting · SVR<br>
                <b style='color:#c7d2fe;'>Deep Learning:</b> PyTorch Quantile-LSTM under Pinball Loss (ℒ<sub>τ</sub>)<br>
                <b style='color:#c7d2fe;'>Validation:</b> Diebold-Mariano Tests · Multi-Window Stress Tests
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Divider ────────────────────────────────────────────────────────
    st.markdown("<div class='qrn-divider'></div>", unsafe_allow_html=True)

    # ── Quick Start Guide ──────────────────────────────────────────────
    _section_title("Quick Start Guide", "🚀")

    q1, q2, q3, q4 = st.columns(4)
    steps = [
        ("1️⃣", "Configure", "Select sectors and date range in the sidebar"),
        ("2️⃣", "Load Data", "Navigate to Data Center to fetch NSE prices"),
        ("3️⃣", "Analyse", "Run diagnostics, GARCH, and QVAR modules"),
        ("4️⃣", "Explore", "Inspect spillover heatmaps & network topology"),
    ]
    for col, (num, title, desc) in zip([q1, q2, q3, q4], steps):
        with col:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(99,102,241,0.18);
                        border-radius:12px;padding:16px 14px;text-align:center;'>
                <div style='font-size:1.5rem;margin-bottom:8px;'>{num}</div>
                <div style='font-weight:700;color:#c7d2fe;font-size:0.9rem;margin-bottom:6px;'>{title}</div>
                <div style='color:#64748b;font-size:0.8rem;line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────
    st.markdown("<div class='qrn-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;font-size:0.76rem;color:#334155;padding:4px 0 12px;'>"
        "© 2026 <strong style='color:#475569;'>Bibek Rout</strong> &nbsp;·&nbsp; "
        "Q-RiskNet India v1.0.0 &nbsp;·&nbsp; MIT License"
        "</div>",
        unsafe_allow_html=True
    )
