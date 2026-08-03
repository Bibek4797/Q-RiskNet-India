import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm, skew, kurtosis
import src.diagnostics.logger as diag

def compute_distribution_metrics(series):
    """
    Computes comprehensive empirical distribution summary stats and Jarque-Bera normality test.
    """
    cleaned = series.dropna()
    mean_val = float(cleaned.mean())
    median_val = float(cleaned.median())
    var_val = float(cleaned.var())
    std_val = float(cleaned.std())
    min_val = float(cleaned.min())
    max_val = float(cleaned.max())
    skew_val = float(skew(cleaned))
    kurt_val = float(kurtosis(cleaned, fisher=False)) # Excess Kurtosis = Kurt - 3

    try:
        jb_stat, jb_p = jarque_bera(cleaned)
        is_normal = jb_p > 0.05
    except Exception:
        jb_stat, jb_p = np.nan, np.nan
        is_normal = False

    return {
        "Sector": series.name,
        "Observations": len(cleaned),
        "Mean (%)": round(mean_val, 4),
        "Median (%)": round(median_val, 4),
        "Variance": round(var_val, 4),
        "Std_Dev (%)": round(std_val, 4),
        "Min (%)": round(min_val, 4),
        "Max (%)": round(max_val, 4),
        "Skewness": round(skew_val, 4),
        "Kurtosis": round(kurt_val, 4),
        "Excess_Kurtosis": round(kurt_val - 3.0, 4),
        "Jarque_Bera_Stat": round(jb_stat, 2) if not np.isnan(jb_stat) else np.nan,
        "JB_p_value": round(jb_p, 4) if not np.isnan(jb_p) else np.nan,
        "Is_Normal": is_normal,
        "Tail_Behavior": "Heavy-Tailed (Fat Tails)" if (kurt_val > 3.0 or jb_p <= 0.05) else "Normal Tails"
    }

def get_kde_comparison(series, num_points=100):
    """
    Computes empirical Kernel Density Estimate (KDE) and reference Gaussian PDF.
    """
    cleaned = series.dropna()
    mu, std = cleaned.mean(), cleaned.std()
    x_grid = np.linspace(cleaned.min(), cleaned.max(), num_points)
    norm_pdf = norm.pdf(x_grid, loc=mu, scale=std)
    return {
        "x": x_grid,
        "gaussian_pdf": norm_pdf
    }

def run_full_distribution_suite(returns_df):
    """
    Runs distribution metrics and Jarque-Bera normality tests for all sectors.
    """
    with diag.DiagnosticTimer("Full Distribution & Normality Suite"):
        results = [compute_distribution_metrics(returns_df[col]) for col in returns_df.columns]
        return pd.DataFrame(results)
