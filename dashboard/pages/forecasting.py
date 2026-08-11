"""
Q-RiskNet India — Forecasting Page (Alias to Portfolio & Validation)
Copyright (c) 2026 Bibek Rout
"""
import dashboard.pages.portfolio_validation as pv

def render_page(returns_df=None):
    """Renders the Portfolio & Validation section."""
    cfg = {"selected_sectors": list(returns_df.columns) if returns_df is not None else []}
    pv.render_page(returns_df, cfg)
