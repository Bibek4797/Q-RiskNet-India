"""
Q-RiskNet India — About Page (Alias to Overview)
Copyright (c) 2026 Bibek Rout
"""
import dashboard.pages.overview as overview

def render_page(returns_df=None, cfg=None):
    """Renders the Overview section."""
    if cfg is None:
        cfg = {"selected_sectors": []}
    overview.render_page(returns_df, cfg)
