"""
Q-RiskNet India — Clean Professional Theme
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
    """Injects clean professional dark theme CSS."""
    st.markdown("""
    <style>
    /* ── Google Fonts ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Root & Global ────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: #0b1120 !important;
    }

    /* ── Hide default Streamlit chrome ─────────────────── */
    [data-testid="stSidebarNav"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Header ─────────────────────────────────────────── */
    .qrn-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.01em;
        margin-bottom: 2px;
    }

    .qrn-subheader {
        font-size: 0.82rem;
        color: #475569;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    .qrn-divider {
        height: 1px;
        background: rgba(99, 102, 241, 0.2);
        border: none;
        margin: 0.75rem 0 1.25rem;
    }

    /* ── Metric Cards ─────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        line-height: 1.2 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    /* ── Tabs ────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        padding: 3px;
        border: 1px solid rgba(99, 102, 241, 0.12);
    }

    .stTabs [data-baseweb="tab"] {
        height: auto !important;
        min-height: 36px;
        border-radius: 7px !important;
        padding: 7px 18px !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        color: #64748b !important;
        background: transparent !important;
        border: none !important;
        transition: color 0.15s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #a78bfa !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.18) !important;
        color: #c7d2fe !important;
        font-weight: 700 !important;
    }

    /* ── Buttons ──────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: #4f46e5 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 9px 22px !important;
        transition: background 0.2s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #4338ca !important;
    }

    .stButton > button:not([kind="primary"]) {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 8px 18px !important;
    }

    .stButton > button:not([kind="primary"]):hover {
        background: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #c7d2fe !important;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #080d1c !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    }

    /* Sidebar nav radio styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 3px;
        display: flex;
        flex-direction: column;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(99, 102, 241, 0.08) !important;
        border-color: rgba(99, 102, 241, 0.25) !important;
        color: #c7d2fe !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #c7d2fe !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: #6366f1 !important;
    }

    /* ── Expander ──────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
    }

    .streamlit-expanderContent {
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* ── Alerts ───────────────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        font-size: 0.875rem !important;
    }

    /* ── DataFrames ──────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* ── Progress Bar ──────────────────────────────────── */
    [data-testid="stProgress"] > div > div > div {
        background: #6366f1 !important;
        border-radius: 4px !important;
    }

    /* ── Inputs ─────────────────────────────────────────── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 8px !important;
    }

    /* ── Typography ──────────────────────────────────────── */
    h1 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; color: #e2e8f0 !important; }
    h2 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; color: #c7d2fe !important; }
    h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; color: #e2e8f0 !important; }

    p, li { color: #94a3b8 !important; line-height: 1.65 !important; }
    strong { color: #c7d2fe !important; }

    .stCaption { color: #475569 !important; font-size: 0.8rem !important; }

    /* ── Tables ──────────────────────────────────────────── */
    table { border-collapse: collapse !important; width: 100% !important; font-size: 0.87rem !important; }
    th {
        background: rgba(99, 102, 241, 0.12) !important;
        color: #a5b4fc !important;
        font-weight: 600 !important;
        padding: 9px 12px !important;
        text-align: left !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
    }
    td {
        padding: 8px 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: #cbd5e1 !important;
        vertical-align: top !important;
    }
    tr:hover td { background: rgba(99, 102, 241, 0.06) !important; }
    tr:nth-child(even) td { background: rgba(255, 255, 255, 0.015) !important; }

    /* ── Scrollbar ──────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0b1120; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #6366f1; }

    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Renders a minimal, professional dashboard header."""
    st.markdown("""
    <div class='qrn-header'>Q-RiskNet India</div>
    <div class='qrn-subheader'>Systemic Risk Analytics · NSE India</div>
    <div class='qrn-divider'></div>
    """, unsafe_allow_html=True)
