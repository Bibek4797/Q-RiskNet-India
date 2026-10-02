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

        use_fallback = False
        close_prices = pd.DataFrame()

        try:
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            if raw_data.empty:
                use_fallback = True
            else:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    close_prices = raw_data['Close']
                else:
                    close_prices = pd.DataFrame(raw_data['Close'])
                    close_prices.columns = tickers

                reverse_map = {v: k for k, v in TICKER_MAP.items()}
                close_prices = close_prices.rename(columns=reverse_map)

                # Validate downloaded sectors and completeness
                missing_sectors = [s for s in sectors if s not in close_prices.columns]
                if missing_sectors:
                    diag.log_warning(f"Live download missing sectors: {missing_sectors}. Triggering local fallback.")
                    use_fallback = True
                else:
                    high_nan = any(close_prices[s].isna().mean() > 0.10 for s in sectors if s in close_prices.columns)
                    if high_nan or len(close_prices.dropna()) < 30:
                        diag.log_warning("Live download incomplete or contains excessive missing values. Triggering local fallback.")
                        use_fallback = True
        except Exception as e:
            diag.log_warning(f"yfinance download failed: {e}. Attempting local fallback.")
            use_fallback = True

        if use_fallback or close_prices.empty:
            if os.path.exists(raw_filepath):
                diag.log_info(f"Loading local raw data fallback from {raw_filepath}")
                fallback_prices = pd.read_csv(raw_filepath, index_col=0, parse_dates=True)
                valid_cols = [s for s in sectors if s in fallback_prices.columns]
                if len(valid_cols) >= 2:
                    fallback_prices = fallback_prices[valid_cols]
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    sliced = fallback_prices.loc[(fallback_prices.index >= start_dt) & (fallback_prices.index <= end_dt)]
                    if len(sliced.dropna()) >= 30:
                        close_prices = sliced
                    else:
                        diag.log_info("Selected date slice yielded fewer than 30 observations; using complete historical dataset.")
                        close_prices = fallback_prices
                    return close_prices.ffill().bfill()

            proc_filepath = os.path.join(ROOT_DIR, PATHS.get("processed_data", "data/processed"), "prices.csv")
            if os.path.exists(proc_filepath):
                diag.log_info(f"Loading processed data fallback from {proc_filepath}")
                proc_prices = pd.read_csv(proc_filepath, index_col=0, parse_dates=True)
                valid_cols = [s for s in sectors if s in proc_prices.columns]
                if len(valid_cols) >= 2:
                    return proc_prices[valid_cols].ffill().bfill()

            diag.log_error("Yahoo Finance returned empty dataset and no local fallback available.")
            raise ValueError("Yahoo Finance returned an empty dataset. Try selecting a different date range.")

        if save_raw and not use_fallback:
            os.makedirs(raw_dir, exist_ok=True)
            close_prices.to_csv(raw_filepath)
            diag.log_info(f"Saved raw prices to {raw_filepath}")

        return close_prices.ffill().bfill()

