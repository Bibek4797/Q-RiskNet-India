import numpy as np
import pandas as pd
import src.diagnostics.logger as diag

def compute_log_returns(prices_df):
    """
    Computes percentage log returns: r_t = ln(P_t / P_{t-1}) * 100
    """
    with diag.DiagnosticTimer("Log Returns Computation"):
        cleaned = prices_df.ffill().bfill().dropna()
        log_ret = np.log(cleaned / cleaned.shift(1)) * 100
        log_ret = log_ret.dropna()
        return log_ret

def compute_daily_simple_returns(prices_df):
    """
    Computes daily percentage returns: R_t = (P_t - P_{t-1}) / P_{t-1} * 100
    """
    cleaned = prices_df.ffill().bfill().dropna()
    simple_ret = cleaned.pct_change() * 100
    return simple_ret.dropna()

def compute_rolling_volatility(returns_df, window=20, annualize=True):
    """
    Computes rolling volatility over specified window size.
    """
    scale = np.sqrt(252) if annualize else 1.0
    rolling_vol = returns_df.rolling(window=window).std() * scale
    return rolling_vol

def compute_drawdowns(prices_df):
    """
    Computes peak-to-trough percentage drawdown series for each sector.
    """
    rolling_max = prices_df.cummax()
    drawdowns = (prices_df - rolling_max) / rolling_max * 100.0
    return drawdowns

def compute_comprehensive_features(prices_df, short_window=20, long_window=60):
    """
    Generates a dictionary of clean features:
    - log_returns
    - daily_returns
    - rolling_vol_20d
    - rolling_vol_60d
    - rolling_mean_20d
    - drawdowns
    """
    with diag.DiagnosticTimer("Comprehensive Feature Engineering"):
        cleaned_prices = prices_df.ffill().bfill().dropna()
        log_ret = compute_log_returns(cleaned_prices)
        simple_ret = compute_daily_simple_returns(cleaned_prices)
        vol_20 = compute_rolling_volatility(log_ret, window=short_window, annualize=True)
        vol_60 = compute_rolling_volatility(log_ret, window=long_window, annualize=True)
        mean_20 = log_ret.rolling(window=short_window).mean()
        drawdowns = compute_drawdowns(cleaned_prices)

        return {
            "prices": cleaned_prices,
            "log_returns": log_ret,
            "daily_returns": simple_ret,
            "volatility_20d": vol_20,
            "volatility_60d": vol_60,
            "rolling_mean_20d": mean_20,
            "drawdowns": drawdowns
        }
