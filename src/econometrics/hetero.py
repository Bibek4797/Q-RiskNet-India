import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch
import src.diagnostics.logger as diag

def run_arch_lm_test(series, lags=5):
    """
    Executes Engle's Lagrange Multiplier (LM) Test for Autoregressive Conditional Heteroskedasticity (ARCH).
    """
    cleaned = series.dropna()
    try:
        res = het_arch(cleaned, nlags=lags)
        lm_stat, p_val, f_stat, f_pval = res
        has_arch = p_val <= 0.05
        return {
            "Sector": series.name,
            "Test": "ARCH-LM",
            "Lag": lags,
            "LM_Statistic": round(lm_stat, 4),
            "p_value": round(p_val, 4),
            "F_Statistic": round(f_stat, 4),
            "F_pval": round(f_pval, 4),
            "ARCH_Effects_Present": has_arch,
            "Interpretation": "Significant ARCH volatility clustering present" if has_arch else "No significant ARCH volatility clustering"
        }
    except Exception as e:
        diag.log_warning(f"ARCH-LM test failed for {series.name}: {e}")
        return {
            "Sector": series.name, "Test": "ARCH-LM", "Lag": lags,
            "LM_Statistic": np.nan, "p_value": np.nan, "F_Statistic": np.nan, "F_pval": np.nan,
            "ARCH_Effects_Present": False, "Interpretation": "Test failed"
        }

def compute_rolling_variance(series, window=20):
    """
    Computes rolling sample variance to visualize volatility clustering.
    """
    return series.rolling(window=window).var()

def run_full_hetero_suite(returns_df, lags=5):
    """
    Runs ARCH-LM test across all sectors.
    """
    with diag.DiagnosticTimer("Full Heteroskedasticity & Volatility Clustering Suite"):
        results = []
        for col in returns_df.columns:
            s = returns_df[col]
            results.append(run_arch_lm_test(s, lags=lags))
        return pd.DataFrame(results)
