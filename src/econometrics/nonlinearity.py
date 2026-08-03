import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import bds
import src.diagnostics.logger as diag

def run_bds_test(series, max_dim=3):
    """
    Executes Brock-Dechert-Scheinkman (BDS) Test for Non-linear Dependence.
    """
    cleaned = series.dropna()
    try:
        bds_stats, p_values = bds(cleaned, max_dim=max_dim)
        dim_results = []
        for d in range(2, max_dim + 1):
            idx = d - 2
            stat = bds_stats[idx]
            p_val = p_values[idx]
            dim_results.append({
                "Sector": series.name,
                "Test": "BDS",
                "Embedding_Dimension": d,
                "Statistic": round(stat, 4),
                "p_value": round(p_val, 4),
                "Nonlinear_Dependence": p_val <= 0.05,
                "Interpretation": "Significant non-linear dependence detected" if p_val <= 0.05 else "No significant non-linear dependence (i.i.d.)"
            })
        return pd.DataFrame(dim_results)
    except Exception as e:
        diag.log_warning(f"BDS test failed for {series.name}: {e}")
        return pd.DataFrame([{
            "Sector": series.name, "Test": "BDS", "Embedding_Dimension": 2,
            "Statistic": np.nan, "p_value": np.nan, "Nonlinear_Dependence": False,
            "Interpretation": "BDS Test failed or missing dependencies"
        }])

def run_full_nonlinearity_suite(returns_df, max_dim=3):
    """
    Runs BDS non-linearity test across all sectors.
    """
    with diag.DiagnosticTimer("Full Non-Linearity Analysis Suite"):
        frames = [run_bds_test(returns_df[col], max_dim=max_dim) for col in returns_df.columns]
        return pd.concat(frames, ignore_index=True)
