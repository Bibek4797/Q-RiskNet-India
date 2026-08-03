import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import src.diagnostics.logger as diag

def compute_acf_pacf(series, nlags=20):
    """
    Computes ACF and PACF values up to nlags.
    """
    cleaned = series.dropna()
    acf_vals = acf(cleaned, nlags=nlags, fft=True)
    pacf_vals = pacf(cleaned, nlags=nlags, method='ywm')
    return {
        "lags": list(range(nlags + 1)),
        "acf": acf_vals,
        "pacf": pacf_vals
    }

def run_ljung_box_test(series, lags=10):
    """
    Executes Ljung-Box test for serial correlation.
    """
    cleaned = series.dropna()
    lb_df = acorr_ljungbox(cleaned, lags=[lags], return_df=True)
    stat = lb_df.loc[lags, "lb_stat"]
    p_val = lb_df.loc[lags, "lb_pvalue"]
    has_autocorr = p_val <= 0.05
    return {
        "Sector": series.name,
        "Test": "Ljung-Box",
        "Lag": lags,
        "Statistic": round(stat, 4),
        "p_value": round(p_val, 4),
        "Autocorrelation_Present": has_autocorr,
        "Interpretation": "Significant serial correlation detected" if has_autocorr else "No significant serial correlation (White Noise)"
    }

def compute_durbin_watson(series):
    """
    Computes Durbin-Watson statistic (2 = no autocorrelation, <2 = positive, >2 = negative).
    """
    cleaned = series.dropna()
    dw_stat = durbin_watson(cleaned)
    return round(dw_stat, 4)

def run_full_autocorrelation_suite(returns_df, lags=10):
    """
    Runs Ljung-Box and Durbin-Watson tests for all sectors.
    """
    with diag.DiagnosticTimer("Full Autocorrelation Analysis Suite"):
        results = []
        for col in returns_df.columns:
            s = returns_df[col]
            lb_res = run_ljung_box_test(s, lags=lags)
            dw_val = compute_durbin_watson(s)
            lb_res["Durbin_Watson"] = dw_val
            results.append(lb_res)
        return pd.DataFrame(results)
