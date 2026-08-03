import os
import tempfile
import pandas as pd
import yfinance as yf
from src.config.settings import TICKER_MAP, PATHS, ROOT_DIR
import src.diagnostics.logger as diag

# Disable yfinance timezone cache to prevent SQLite database lock issues
try:
    temp_dir = os.path.join(tempfile.gettempdir(), "yf_cache")
    os.makedirs(temp_dir, exist_ok=True)
    yf.set_tz_cache_location(temp_dir)
except Exception:
    pass

def fetch_raw_market_data(sectors, start_date, end_date, save_raw=True):
    """
    Downloads raw index data from Yahoo Finance and optionally saves to data/raw/.
    """
    with diag.DiagnosticTimer("Raw Market Data Ingestion"):
        tickers = [TICKER_MAP[s] for s in sectors if s in TICKER_MAP]
        if not tickers:
            diag.log_warning("No valid tickers matched selected sectors.")
            return pd.DataFrame()

        diag.log_info(f"Downloading tickers: {tickers} from {start_date} to {end_date}")

        raw_dir = os.path.join(ROOT_DIR, PATHS.get("raw_data", "data/raw"))
        raw_filepath = os.path.join(raw_dir, "raw_prices.csv")

        try:
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        except Exception as e:
            diag.log_warning(f"yfinance download failed: {e}. Attempting local fallback.")
            raw_data = pd.DataFrame()

        if raw_data.empty:
            if os.path.exists(raw_filepath):
                diag.log_info(f"Loading local raw data fallback from {raw_filepath}")
                close_prices = pd.read_csv(raw_filepath, index_col=0, parse_dates=True)
                valid_cols = [s for s in sectors if s in close_prices.columns]
                if valid_cols:
                    close_prices = close_prices[valid_cols]
                    close_prices = close_prices.loc[str(start_date):str(end_date)]
                    if not close_prices.empty:
                        return close_prices
            diag.log_error("Yahoo Finance returned empty dataset and no local fallback available.")
            raise ValueError("Yahoo Finance returned an empty dataset. Try selecting a different date range.")

        if isinstance(raw_data.columns, pd.MultiIndex):
            close_prices = raw_data['Close']
        else:
            close_prices = pd.DataFrame(raw_data['Close'])
            close_prices.columns = tickers

        reverse_map = {v: k for k, v in TICKER_MAP.items()}
        close_prices = close_prices.rename(columns=reverse_map)

        if save_raw:
            os.makedirs(raw_dir, exist_ok=True)
            close_prices.to_csv(raw_filepath)
            diag.log_info(f"Saved raw prices to {raw_filepath}")

        return close_prices

