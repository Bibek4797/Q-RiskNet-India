import streamlit as st
from src.config.settings import DASHBOARD_CFG

def setup_page_config():
    """
    Initializes Streamlit page configuration.
    """
    st.set_page_config(
        page_title=DASHBOARD_CFG.get("page_title", "Q-RiskNet India | Volatility Spillover Dashboard"),
        page_icon=DASHBOARD_CFG.get("page_icon", "📈"),
        layout=DASHBOARD_CFG.get("layout", "wide"),
        initial_sidebar_state=DASHBOARD_CFG.get("initial_sidebar_state", "expanded")
    )

def inject_custom_css():
    """
    Injects custom CSS theme for executive-level dark UI styling.
    """
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
        .main-header {{
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, {primary_color} 0%, {secondary_color} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .sub-header {{
            font-size: 1.05rem;
            color: #94a3b8;
            margin-bottom: 1.5rem;
        }}
        .metric-card {{
            background-color: {card_bg};
            border-radius: 8px;
            padding: 16px;
            border-left: 4px solid {primary_color};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 45px;
            white-space: pre-wrap;
            border-radius: 6px 6px 0px 0px;
            padding-left: 16px;
            padding-right: 16px;
            font-weight: 600;
        }}
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """
    Renders top dashboard title banner.
    """
    st.markdown("<div class='main-header'>Q-RiskNet-India: Sectoral Spillover Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Quantile-LSTM Deep Learning & Financial Network Topology Analysis for Indian Stock Markets</div>", unsafe_allow_html=True)
