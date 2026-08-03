import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import jarque_bera
import src.diagnostics.logger as diag

def _extract_model_metrics(res, model_type, series_name):
    """
    Extracts persistence, half-life, long-run variance, AIC, BIC, LogLik, and residual diagnostics.
    """
    params = res.params
    scale = res.scale if res.scale else 1.0
    if scale == 0:
        scale = 1.0

    omega = params.get('omega', np.nan)
    alpha = params.get('alpha[1]', np.nan)
    beta = params.get('beta[1]', np.nan)
    gamma = params.get('gamma[1]', np.nan)

    # Persistence calculation
    if model_type == "ARCH(1)":
        persistence = alpha if not np.isnan(alpha) else np.nan
    elif model_type == "GARCH(1,1)":
        persistence = (alpha + beta) if (not np.isnan(alpha) and not np.isnan(beta)) else np.nan
    elif model_type == "GJR-GARCH(1,1,1)":
        gamma_val = gamma if not np.isnan(gamma) else 0.0
        persistence = (alpha + beta + 0.5 * gamma_val) if (not np.isnan(alpha) and not np.isnan(beta)) else np.nan
    elif model_type == "EGARCH(1,1,1)":
        persistence = beta if not np.isnan(beta) else np.nan
    else:
        persistence = np.nan

    # Half-life calculation in days: HL = ln(0.5) / ln(persistence)
    if not np.isnan(persistence) and 0 < persistence < 1:
        half_life = np.log(0.5) / np.log(persistence)
    else:
        half_life = np.nan

    # Long-run unconditional variance
    if not np.isnan(persistence) and 0 < persistence < 1 and not np.isnan(omega):
        long_run_var = (omega / (1.0 - persistence)) / scale
        long_run_vol = np.sqrt(max(0, long_run_var)) * np.sqrt(252)
    else:
        long_run_var = np.nan
        long_run_vol = np.nan

    # Standardized Residual Diagnostics
    std_resid = res.std_resid.dropna()
    try:
        arch_lm_res = het_arch(std_resid, nlags=5)
        arch_lm_pval = round(arch_lm_res[1], 4)
    except Exception:
        arch_lm_pval = np.nan

    try:
        jb_stat, jb_pval = jarque_bera(std_resid)
        jb_pval = round(jb_pval, 4)
    except Exception:
        jb_pval = np.nan

    return {
        "Sector": series_name,
        "Model": model_type,
        "Log_Likelihood": round(res.loglikelihood, 2),
        "AIC": round(res.aic, 2),
        "BIC": round(res.bic, 2),
        "Omega": round(omega, 6) if not np.isnan(omega) else np.nan,
        "Alpha": round(alpha, 4) if not np.isnan(alpha) else np.nan,
        "Gamma_Asymmetry": round(gamma, 4) if not np.isnan(gamma) else np.nan,
        "Beta": round(beta, 4) if not np.isnan(beta) else np.nan,
        "Persistence": round(persistence, 4) if not np.isnan(persistence) else np.nan,
        "Half_Life_Days": round(half_life, 2) if not np.isnan(half_life) else np.nan,
        "Long_Run_Vol_Pct": round(long_run_vol, 2) if not np.isnan(long_run_vol) else np.nan,
        "Resid_ARCH_LM_pVal": arch_lm_pval,
        "Resid_JB_pVal": jb_pval,
        "fit_result": res
    }

def fit_arch_model(returns_series):
    """
    Fits ARCH(1) model.
    """
    am = arch_model(returns_series, vol='Garch', p=1, q=0, dist='normal', rescale=True)
    res = am.fit(disp='off')
    return _extract_model_metrics(res, "ARCH(1)", returns_series.name)

def fit_garch_model(returns_series):
    """
    Fits standard symmetric GARCH(1,1) model.
    """
    am = arch_model(returns_series, vol='Garch', p=1, q=1, dist='normal', rescale=True)
    res = am.fit(disp='off')
    return _extract_model_metrics(res, "GARCH(1,1)", returns_series.name)

def fit_egarch_model(returns_series):
    """
    Fits Nelson's Exponential EGARCH(1,1,1) model.
    """
    am = arch_model(returns_series, vol='EGARCH', p=1, o=1, q=1, dist='normal', rescale=True)
    res = am.fit(disp='off')
    return _extract_model_metrics(res, "EGARCH(1,1,1)", returns_series.name)

def fit_gjr_garch_model(returns_series):
    """
    Fits Glosten-Jagannathan-Runkle GJR-GARCH(1,1,1) Threshold Asymmetric model.
    """
    am = arch_model(returns_series, vol='Garch', p=1, o=1, q=1, dist='normal', rescale=True)
    res = am.fit(disp='off')
    return _extract_model_metrics(res, "GJR-GARCH(1,1,1)", returns_series.name)

def compare_volatility_models_for_sector(returns_series):
    """
    Fits ARCH(1), GARCH(1,1), EGARCH(1,1,1), and GJR-GARCH(1,1,1) for a single sector series.
    Returns comparison dataframe sorted by AIC.
    """
    models_list = []
    for fit_fn in [fit_arch_model, fit_garch_model, fit_egarch_model, fit_gjr_garch_model]:
        try:
            m_res = fit_fn(returns_series)
            models_list.append(m_res)
        except Exception as e:
            diag.log_warning(f"Volatility model fitting failed for {returns_series.name}: {e}")

    df = pd.DataFrame(models_list)
    if "AIC" in df.columns:
        df = df.sort_values(by="AIC")
    return df

def generate_multi_step_volatility_forecast(res_object, horizons=[1, 5, 20]):
    """
    Generates multi-step ahead conditional volatility forecasts (annualized %).
    Supports simulation / bootstrap fallback for EGARCH models when horizon > 1.
    """
    res = res_object
    scale = res.scale if res.scale else 1.0
    if scale == 0:
        scale = 1.0

    max_h = max(horizons)
    try:
        fc = res.forecast(horizon=max_h, method='analytic')
    except Exception:
        try:
            fc = res.forecast(horizon=max_h, method='simulation', simulations=500)
        except Exception:
            fc = res.forecast(horizon=max_h, method='bootstrap')

    variance_forecasts = fc.variance.iloc[-1].values / scale

    forecast_dict = {}
    for h in horizons:
        var_h = variance_forecasts[h - 1]
        vol_annualized = np.sqrt(max(0, var_h)) * np.sqrt(252)
        forecast_dict[f"Forecast_{h}d_Vol_Pct"] = round(vol_annualized, 2)

    return forecast_dict
