import os
import tempfile
import numpy as np
import pandas as pd
import yfinance as yf

from src.config.settings import TICKER_MAP
import src.diagnostics.logger as diag

# Disable yfinance timezone cache to prevent SQLite database lock issues by using a local temp path
try:
    temp_dir = os.path.join(tempfile.gettempdir(), "yf_cache")
    os.makedirs(temp_dir, exist_ok=True)
    yf.set_tz_cache_location(temp_dir)
except Exception:
    pass

def download_data(sectors, start_date, end_date):
    """
    Downloads historical close price data for selected sectors via Yahoo Finance.
    """
    with diag.DiagnosticTimer("yfinance Tickers Download"):
        tickers = [TICKER_MAP[s] for s in sectors if s in TICKER_MAP]
        if not tickers:
            diag.log_warning("No valid tickers matched selected sectors.")
            return pd.DataFrame()
        
        diag.log_info(f"Target Tickers: {tickers} from {start_date} to {end_date}")
        
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        except Exception as e:
            diag.log_error("Failed to execute yf.download", e)
            raise ValueError(f"Failed to download data from Yahoo Finance: {str(e)}")
            
        if data.empty:
            diag.log_error("Yahoo Finance returned an empty dataset.")
            raise ValueError("Yahoo Finance returned an empty dataset. Try selecting a different date range.")
            
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close']
        else:
            close_prices = pd.DataFrame(data['Close'])
            close_prices.columns = tickers
            
        reverse_map = {v: k for k, v in TICKER_MAP.items()}
        close_prices = close_prices.rename(columns=reverse_map)
        
        empty_cols = [col for col in close_prices.columns if close_prices[col].isna().all()]
        if empty_cols:
            diag.log_warning(f"Sectors returned all NaNs and will be dropped: {empty_cols}")
            close_prices = close_prices.drop(columns=empty_cols)
            
        if close_prices.empty or len(close_prices.columns) < 2:
            diag.log_error(f"Fewer than 2 valid sectors remaining after cleaning empty columns: {list(close_prices.columns)}")
            raise ValueError("Fewer than 2 valid sectors downloaded. Please select other sectors or a wider date range.")
            
        close_prices = close_prices.ffill().bfill()
        
        nan_sums = close_prices.isna().sum().sum()
        if nan_sums > 0:
            diag.log_warning(f"Remaining NaNs found after ffill/bfill: {nan_sums}. Dropping rows with NaNs.")
            close_prices = close_prices.dropna()
            
        diag.log_info(f"Final cleaned prices shape: {close_prices.shape} for sectors {list(close_prices.columns)}")
        return close_prices

def calculate_log_returns(prices_df):
    """
    Calculates percentage log returns: r_t = ln(P_t / P_{t-1}) * 100
    """
    with diag.DiagnosticTimer("Log Returns Calculation"):
        returns_df = np.log(prices_df / prices_df.shift(1)) * 100
        returns_df = returns_df.dropna()
        diag.log_info(f"Log returns shape: {returns_df.shape}")
        if returns_df.empty:
            diag.log_error("Log returns dataset is empty after dropna()!")
            raise ValueError("Returns calculation resulted in an empty dataset. Check your date range or data quality.")
        return returns_df
