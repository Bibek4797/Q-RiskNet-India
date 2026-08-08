"""
Q-RiskNet India — Premium Executive Theme & Custom CSS
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
from src.config.settings import DASHBOARD_CFG


def setup_page_config():
    """Initializes Streamlit page configuration."""
    st.set_page_config(
        page_title="Q-RiskNet India | Systemic Risk Analytics",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def inject_custom_css():
    """Injects premium glassmorphism dark UI with animations."""
    st.markdown("""
    <style>
    /* ── Google Fonts ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Root & Global ────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: radial-gradient(ellipse at 10% 0%, #1a0a3e 0%, #050914 40%, #0a0f1e 100%) !important;
    }

    /* ── Hide default Streamlit nav ────────────────────── */
    [data-testid="stSidebarNav"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Main Header ───────────────────────────────────── */
    .qrn-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 30%, #a78bfa 60%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin-bottom: 0.15rem;
    }

    .qrn-subheader {
        font-size: 0.95rem;
        color: #64748b;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    .qrn-badge {
        display: inline-block;
        background: linear-gradient(90deg, rgba(99,102,241,0.2), rgba(167,139,250,0.2));
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #a78bfa;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    .qrn-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.5), rgba(167,139,250,0.3), transparent);
        border: none;
        margin: 1.5rem 0;
    }

    /* ── Glassmorphism Metric Cards ────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(99,102,241,0.25) !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
        position: relative;
        overflow: hidden;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(99,102,241,0.25) !important;
        border-color: rgba(99,102,241,0.5) !important;
    }

    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #a78bfa, #c084fc);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        line-height: 1.2 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    /* ── Tab Styling ────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(99,102,241,0.15);
    }

    .stTabs [data-baseweb="tab"] {
        height: auto !important;
        min-height: 40px;
        white-space: normal !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #64748b !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #a78bfa !important;
        background: rgba(99,102,241,0.1) !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
    }

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 10px 24px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(79,70,229,0.35) !important;
        letter-spacing: 0.02em;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(79,70,229,0.55) !important;
        filter: brightness(1.08) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080d1c 0%, #0d1526 100%) !important;
        border-right: 1px solid rgba(99,102,241,0.2) !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }

    /* Sidebar nav radio: hide circle dots, custom button style */
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 4px;
        display: flex;
        flex-direction: column;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(99,102,241,0.18) !important;
        border-radius: 9px !important;
        padding: 10px 14px !important;
        margin-bottom: 2px !important;
        cursor: pointer !important;
        transition: all 0.18s ease !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(99,102,241,0.12) !important;
        border-color: rgba(99,102,241,0.4) !important;
        color: #c7d2fe !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(135deg, rgba(79,70,229,0.35), rgba(124,58,237,0.25)) !important;
        border-color: #6366f1 !important;
        color: #c7d2fe !important;
        font-weight: 600 !important;
        box-shadow: 0 0 0 1px rgba(99,102,241,0.3), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    }

    /* Sidebar sliders */
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: #6366f1 !important;
    }

    /* ── Expander ─────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    .streamlit-expanderContent {
        border: 1px solid rgba(99,102,241,0.15) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* ── Info / Warning / Error Banners ──────────────────── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border: none !important;
        font-size: 0.88rem !important;
    }

    /* ── DataFrame / Table ────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* ── Progress Bar ─────────────────────────────────────── */
    [data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
        border-radius: 4px !important;
    }

    /* ── Selectbox / Slider / Input ───────────────────────── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(99,102,241,0.25) !important;
        border-radius: 9px !important;
    }

    /* ── Section Headers (st.header / st.subheader) ────────── */
    h1 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; color: #e2e8f0 !important; }
    h2 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; color: #c7d2fe !important; }
    h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; color: #a5b4fc !important; }

    /* ── Caption / Small Text ─────────────────────────────── */
    .stCaption { color: #475569 !important; font-size: 0.82rem !important; }

    /* ── Custom Info Card ─────────────────────────────────── */
    .info-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(167,139,250,0.06));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 18px 22px;
        margin: 12px 0;
    }

    .stat-chip {
        display: inline-block;
        background: rgba(99,102,241,0.18);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #a5b4fc;
        margin: 2px 4px 2px 0;
    }

    /* ── Hypothesis Table Styling ─────────────────────────── */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 0.88rem !important;
    }
    th {
        background: rgba(99,102,241,0.2) !important;
        color: #c7d2fe !important;
        font-weight: 600 !important;
        padding: 10px 14px !important;
        text-align: left !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
    }
    td {
        padding: 9px 14px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        color: #cbd5e1 !important;
        vertical-align: top !important;
    }
    tr:hover td {
        background: rgba(99,102,241,0.07) !important;
    }
    tr:nth-child(even) td {
        background: rgba(255,255,255,0.02) !important;
    }

    /* ── Scrollbar ─────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #050914; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6366f1; }

    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Renders the premium animated dashboard header."""
    st.markdown("""
    <div style='margin-bottom: 0.5rem;'>
        <span class='qrn-badge'>Live Data</span>
        <span class='qrn-badge'>NSE India</span>
        <span class='qrn-badge'>v1.0.0</span>
    </div>
    <div class='qrn-header'>Q-RiskNet India</div>
    <div class='qrn-subheader'>Systemic Risk &amp; Volatility Spillover Intelligence Platform</div>
    <div class='qrn-divider'></div>
    """, unsafe_allow_html=True)
