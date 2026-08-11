"""
Q-RiskNet India — Data Center Page (Alias to Market & Risk)
Copyright (c) 2026 Bibek Rout
"""
import dashboard.pages.market_risk as market_risk

def render_page(prices_df=None, returns_df=None, features_dict=None, val_report=None, cfg=None):
    """Renders the Market & Risk section."""
    if cfg is None:
        cfg = {"selected_sectors": list(prices_df.columns) if prices_df is not None else []}
    diag_res = None
    vol_res = None
    market_risk.render_page(prices_df, returns_df, features_dict, val_report, diag_res, vol_res, cfg)
