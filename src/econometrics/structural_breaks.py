import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import zivot_andrews
import src.diagnostics.logger as diag

def run_cusum_break_test(series):
    """
    Executes OLS CUSUM test for parameter constancy and structural stability.
    """
    cleaned = series.dropna()
    y = cleaned.values
    X = np.ones((len(y), 1))
    
    try:
        res = sm.OLS(y, X).fit()
        cusum_stat, p_val, crit = breaks_cusumolsresid(res.resid)
        has_break = p_val <= 0.05
        return {
            "Sector": series.name,
            "Test": "CUSUM",
            "Statistic": round(cusum_stat, 4),
            "p_value": round(p_val, 4),
            "Structural_Break_Present": has_break,
            "Interpretation": "Significant structural instability / parameter break" if has_break else "Stable parameters over time"
        }
    except Exception as e:
        diag.log_warning(f"CUSUM test failed for {series.name}: {e}")
        return {
            "Sector": series.name, "Test": "CUSUM", "Statistic": np.nan,
            "p_value": np.nan, "Structural_Break_Present": False, "Interpretation": "Test failed"
        }

def run_full_structural_breaks_suite(returns_df):
    """
    Runs CUSUM structural break test for all sectors.
    """
    with diag.DiagnosticTimer("Full Structural Breaks Analysis Suite"):
        results = [run_cusum_break_test(returns_df[col]) for col in returns_df.columns]
        return pd.DataFrame(results)
