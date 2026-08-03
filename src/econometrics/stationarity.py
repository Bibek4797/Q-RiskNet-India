import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
import src.diagnostics.logger as diag

def run_adf_test(series, maxlag=None):
    """
    Executes Augmented Dickey-Fuller (ADF) Unit Root Test.
    """
    cleaned = series.dropna()
    res = adfuller(cleaned, maxlag=maxlag, autolag='AIC')
    stat, p_val, usedlag, nobs, crit, _ = res
    is_stationary = p_val <= 0.05
    return {
        "Sector": series.name,
        "Test": "ADF",
        "Statistic": round(stat, 4),
        "p_value": round(p_val, 4),
        "Lags": usedlag,
        "Crit_1%": round(crit['1%'], 4),
        "Crit_5%": round(crit['5%'], 4),
        "Decision": "Stationary" if is_stationary else "Non-Stationary",
        "Interpretation": "Reject H0: Series has no unit root" if is_stationary else "Fail to reject H0: Series contains unit root"
    }

def run_kpss_test(series, regression='c'):
    """
    Executes KPSS Trend-Stationarity Test.
    """
    cleaned = series.dropna()
    try:
        stat, p_val, lags, crit = kpss(cleaned, regression=regression, nlags='auto')
        is_stationary = p_val > 0.05
        return {
            "Sector": series.name,
            "Test": "KPSS",
            "Statistic": round(stat, 4),
            "p_value": round(p_val, 4),
            "Lags": lags,
            "Crit_1%": round(crit['1%'], 4),
            "Crit_5%": round(crit['5%'], 4),
            "Decision": "Stationary" if is_stationary else "Non-Stationary",
            "Interpretation": "Fail to reject H0: Trend-stationary" if is_stationary else "Reject H0: Non-stationary"
        }
    except Exception as e:
        diag.log_warning(f"KPSS test failed for {series.name}: {e}")
        return {
            "Sector": series.name, "Test": "KPSS", "Statistic": np.nan, "p_value": np.nan,
            "Lags": 0, "Crit_1%": np.nan, "Crit_5%": np.nan, "Decision": "N/A", "Interpretation": "Test failed"
        }

def run_zivot_andrews_test(series):
    """
    Executes Zivot-Andrews Unit Root Test with single structural break.
    """
    cleaned = series.dropna()
    try:
        stat, p_val, crit, bvar, bdate = zivot_andrews(cleaned)
        break_date_str = str(cleaned.index[bdate].date()) if isinstance(bdate, int) and bdate < len(cleaned) else "N/A"
        is_stationary = p_val <= 0.05
        return {
            "Sector": series.name,
            "Test": "Zivot-Andrews",
            "Statistic": round(stat, 4),
            "p_value": round(p_val, 4),
            "Break_Date": break_date_str,
            "Crit_1%": round(crit['1%'], 4),
            "Crit_5%": round(crit['5%'], 4),
            "Decision": "Stationary w/ Break" if is_stationary else "Unit Root w/ Break",
            "Interpretation": f"Detected potential structural break near {break_date_str}"
        }
    except Exception as e:
        diag.log_warning(f"Zivot-Andrews test failed for {series.name}: {e}")
        return {
            "Sector": series.name, "Test": "Zivot-Andrews", "Statistic": np.nan, "p_value": np.nan,
            "Break_Date": "N/A", "Crit_1%": np.nan, "Crit_5%": np.nan, "Decision": "N/A", "Interpretation": "Test failed"
        }

def run_full_stationarity_suite(returns_df):
    """
    Runs ADF, KPSS, and Zivot-Andrews tests for all sectors in dataframe.
    """
    with diag.DiagnosticTimer("Full Stationarity Analysis Suite"):
        results = []
        for col in returns_df.columns:
            s = returns_df[col]
            results.append(run_adf_test(s))
            results.append(run_kpss_test(s))
            results.append(run_zivot_andrews_test(s))
        return pd.DataFrame(results)
