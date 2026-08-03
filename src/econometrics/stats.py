import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera
import src.diagnostics.logger as diag

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
            kurt_val = series.kurtosis() + 3
            
            try:
                jb_stat, jb_p = jarque_bera(series)
            except Exception:
                jb_stat, jb_p = np.nan, np.nan
                
            try:
                adf_res = adfuller(series)
                adf_stat = adf_res[0]
                adf_p = adf_res[1]
            except Exception:
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
