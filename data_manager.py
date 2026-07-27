import yfinance as yf
import pandas as pd
import numpy as np
import os
import tempfile
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera

# Import diagnostics logger
import diagnostics as diag

# Disable yfinance timezone cache to prevent SQLite database lock issues by using a local temp path
try:
    temp_dir = os.path.join(tempfile.gettempdir(), "yf_cache")
    os.makedirs(temp_dir, exist_ok=True)
    yf.set_tz_cache_location(temp_dir)
except Exception as e:
    pass

# Default Sector Mapping to Yahoo Finance Tickers
TICKER_MAP = {
    "Nifty 50": "^NSEI",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Auto": "^CNXAUTO",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Energy": "^CNXENERGY",
    "Nifty Realty": "^CNXREALTY",
    "Nifty Financial Services": "NIFTY_FIN_SERVICE.NS"
}

def download_data(sectors, start_date, end_date):
    """
    Downloads historical close price data for selected sectors.
    Validates that the downloaded data is not empty and contains valid tickers.
    """
    with diag.DiagnosticTimer("yfinance Tickers Download"):
        tickers = [TICKER_MAP[s] for s in sectors if s in TICKER_MAP]
        if not tickers:
            diag.log_warning("No valid tickers matched selected sectors.")
            return pd.DataFrame()
        
        diag.log_info(f"Target Tickers: {tickers} from {start_date} to {end_date}")
        
        # Download data
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        except Exception as e:
            diag.log_error("Failed to execute yf.download", e)
            raise ValueError(f"Failed to download data from Yahoo Finance: {str(e)}")
            
        if data.empty:
            diag.log_error("Yahoo Finance returned an empty dataset.")
            raise ValueError("Yahoo Finance returned an empty dataset. Try selecting a different date range.")
            
        # Extract 'Close' prices
        if isinstance(data.columns, pd.MultiIndex):
            close_prices = data['Close']
        else:
            close_prices = pd.DataFrame(data['Close'])
            close_prices.columns = tickers
            
        # Rename columns to human-readable names
        reverse_map = {v: k for k, v in TICKER_MAP.items()}
        close_prices = close_prices.rename(columns=reverse_map)
        
        # Check for completely empty columns (failed downloads or inactive sectors in date range)
        empty_cols = []
        for col in close_prices.columns:
            if close_prices[col].isna().all():
                empty_cols.append(col)
                
        if empty_cols:
            diag.log_warning(f"Sectors returned all NaNs and will be dropped: {empty_cols}")
            close_prices = close_prices.drop(columns=empty_cols)
            
        if close_prices.empty or len(close_prices.columns) < 2:
            diag.log_error(f"Fewer than 2 valid sectors remaining after cleaning empty columns: {list(close_prices.columns)}")
            raise ValueError("Fewer than 2 valid sectors downloaded. Please select other sectors or a wider date range.")
            
        # Forward fill and backward fill missing values (e.g. holidays / missing index points)
        close_prices = close_prices.ffill().bfill()
        
        # Double check if any NaNs remain
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

def estimate_garch_volatility(returns_series):
    """
    Estimates conditional volatility using an asymmetric GJR-GARCH(1,1,1) model 
    to capture the leverage effect (negative shocks inducing higher volatility than positive shocks).
    """
    sector_name = returns_series.name
    with diag.DiagnosticTimer(f"GJR-GARCH(1,1,1) Estimation for {sector_name}"):
        try:
            # GJR-GARCH with p=1 (GARCH lag), o=1 (asymmetric ARCH lag), q=1 (ARCH lag)
            model = arch_model(returns_series, vol='Garch', p=1, o=1, q=1, dist='normal', rescale=True)
            res = model.fit(disp='off')
            scale = res.scale if res.scale else 1.0
            if scale == 0:
                scale = 1.0
            vol = res.conditional_volatility / scale
            diag.log_info(f"GJR-GARCH converged for {sector_name}. Rescale scale factor: {scale:.4f}")
            return vol
        except Exception as e:
            # Fallback to standard GARCH(1,1) or rolling std dev if GJR-GARCH fails
            diag.log_warning(f"GJR-GARCH failed to converge for {sector_name}. Trying standard GARCH(1,1)...")
            try:
                model_std = arch_model(returns_series, vol='Garch', p=1, q=1, dist='normal', rescale=True)
                res_std = model_std.fit(disp='off')
                scale = res_std.scale if res_std.scale else 1.0
                return res_std.conditional_volatility / scale
            except Exception:
                diag.log_warning(f"Standard GARCH failed for {sector_name}. Fallback to rolling std dev.")
                return returns_series.rolling(window=10).std().fillna(returns_series.std())

def get_descriptive_stats(returns_df):
    """
    Computes summary descriptive stats: Mean, Min, Max, Std, Skew, Kurtosis, Jarque-Bera, ADF
    """
    with diag.DiagnosticTimer("Econometric Descriptive Statistics"):
        stats_list = []
        for col in returns_df.columns:
            series = returns_df[col].dropna()
            if len(series) < 10:
                diag.log_warning(f"Sector {col} has too few data points for descriptive stats ({len(series)}).")
                continue
                
            mean_val = series.mean()
            std_val = series.std()
            min_val = series.min()
            max_val = series.max()
            skew_val = series.skew()
            kurt_val = series.kurtosis() + 3  # Excess kurtosis + 3 = Pearson Kurtosis
            
            # Jarque-Bera Normality Test
            try:
                jb_stat, jb_p = jarque_bera(series)
            except Exception as e:
                jb_stat, jb_p = np.nan, np.nan
                
            # Augmented Dickey-Fuller Unit Root Test
            try:
                adf_res = adfuller(series)
                adf_stat = adf_res[0]
                adf_p = adf_res[1]
            except Exception as e:
                adf_stat, adf_p = np.nan, np.nan
                
            stats_list.append({
                "Sector": col,
                "Observations": len(series),
                "Mean (%)": round(mean_val, 4),
                "Std Dev (%)": round(std_val, 4),
                "Min (%)": round(min_val, 4),
                "Max (%)": round(max_val, 4),
                "Skewness": round(skew_val, 4),
                "Kurtosis": round(kurt_val, 4),
                "JB Stat": round(jb_stat, 2) if not np.isnan(jb_stat) else "N/A",
                "JB p-val": round(jb_p, 4) if not np.isnan(jb_p) else "N/A",
                "ADF Stat": round(adf_stat, 4) if not np.isnan(adf_stat) else "N/A",
                "ADF p-val": round(adf_p, 4) if not np.isnan(adf_p) else "N/A"
            })
            
        return pd.DataFrame(stats_list)
