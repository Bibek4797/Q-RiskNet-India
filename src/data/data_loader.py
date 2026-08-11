"""
Q-RiskNet India — Data Loader Module (Re-exporter & Wrapper)
Copyright (c) 2026 Bibek Rout
"""
import pandas as pd
from src.data.download import fetch_raw_market_data
from src.data.preprocessing import compute_log_returns

def download_data(sectors, start_date, end_date):
    """
    Downloads historical close price data for selected sectors via Yahoo Finance or local fallback.
    Canonical implementation delegated to src.data.download.fetch_raw_market_data.
    """
    return fetch_raw_market_data(sectors, start_date, end_date, save_raw=True)

def calculate_log_returns(prices_df):
    """
    Calculates percentage log returns: r_t = ln(P_t / P_{t-1}) * 100
    Canonical implementation delegated to src.data.preprocessing.compute_log_returns.
    """
    return compute_log_returns(prices_df)
