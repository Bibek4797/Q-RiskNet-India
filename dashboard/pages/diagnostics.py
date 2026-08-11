"""
Q-RiskNet India — Diagnostics Page (Alias to Market & Risk)
Copyright (c) 2026 Bibek Rout
"""
import dashboard.pages.market_risk as market_risk

def render_page(returns_df=None, diag_res=None):
    """Renders the Market & Risk section."""
    prices_df = returns_df
    features_dict = {"drawdowns": returns_df, "volatility_20d": returns_df} if returns_df is not None else {}
    val_report = {"total_rows": len(returns_df) if returns_df is not None else 0, "is_valid": True}
    cfg = {"selected_sectors": list(returns_df.columns) if returns_df is not None else []}
    vol_res = None
    market_risk.render_page(prices_df, returns_df, features_dict, val_report, diag_res, vol_res, cfg)
