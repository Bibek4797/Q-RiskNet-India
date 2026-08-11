"""
Q-RiskNet India — Home Page (Alias to Overview)
Copyright (c) 2026 Bibek Rout
"""
import dashboard.pages.overview as overview

def render_page(returns_df=None, cfg=None):
    """Renders the Overview page."""
    if cfg is None:
        cfg = {"lags": 2, "forecast_horizon": 10}
    overview.render_page(returns_df, cfg)
