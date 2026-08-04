"""
Q-RiskNet India — Executive Theme & Custom CSS Styling
Copyright (c) 2026 Bibek Rout
"""
import streamlit as st
from src.config.settings import DASHBOARD_CFG


def setup_page_config():
    """Initializes Streamlit page configuration."""
    st.set_page_config(
        page_title=DASHBOARD_CFG.get("page_title", "Q-RiskNet India | Volatility Spillover Dashboard"),
        page_icon=DASHBOARD_CFG.get("page_icon", "📈"),
        layout=DASHBOARD_CFG.get("layout", "wide"),
        initial_sidebar_state=DASHBOARD_CFG.get("initial_sidebar_state", "expanded")
    )


def inject_custom_css():
    """Injects executive-level dark UI styling & hides native radio dots."""
    theme = DASHBOARD_CFG.get("theme", {})
    primary_color = theme.get("primary_color", "#3b82f6")
    secondary_color = theme.get("secondary_color", "#8b5cf6")
    card_bg = theme.get("card_background", "#1e293b")

    st.markdown(f"""
    <style>
        /* Hide default Streamlit multi-page auto-generated sidebar navigation */
        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}

        /* Top Header Styling */
        .main-header {{
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, {primary_color} 0%, {secondary_color} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.0rem;
        }}

        /* Executive Metric Cards */
        .metric-card {{
            background-color: {card_bg};
            border-radius: 8px;
            padding: 16px;
            border-left: 4px solid {primary_color};
        }}

        /* Sub-tab bar height & multi-line text fix */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: auto !important;
            min-height: 44px;
            white-space: normal !important;
            border-radius: 6px 6px 0px 0px;
            padding-top: 8px;
            padding-bottom: 8px;
            padding-left: 16px;
            padding-right: 16px;
            font-weight: 600;
        }}

        /* Hide childish radio circle dots in sidebar */
        [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {{
            display: none !important;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {{
            gap: 6px;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label {{
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 8px 12px !important;
            margin-bottom: 2px;
            cursor: pointer;
            transition: all 0.15s ease-in-out;
            width: 100%;
            display: flex;
            align-items: center;
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
            background-color: #334155;
            border-color: {primary_color};
        }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
            background: linear-gradient(90deg, #1d4ed8 0%, {primary_color} 100%) !important;
            border-color: #60a5fa !important;
            color: #ffffff !important;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Renders sleek top dashboard project title."""
    st.markdown("<div class='main-header'>Q-RiskNet India</div>", unsafe_allow_html=True)
