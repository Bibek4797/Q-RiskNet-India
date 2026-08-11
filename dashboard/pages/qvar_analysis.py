"""
Q-RiskNet India — QVAR Analysis Page (Alias to Connectedness)
Copyright (c) 2026 Bibek Rout
"""
import dashboard.pages.connectedness as connectedness

def render_page(returns_df=None, qvar_res=None):
    """Renders the Connectedness section."""
    cfg = {"selected_sectors": list(returns_df.columns) if returns_df is not None else [], "lags": 2, "forecast_horizon": 10}
    connectedness.render_page(returns_df, returns_df, cfg)
