"""
Q-RiskNet India — Tail Risk & VaR Backtesting Module
Copyright (c) 2026 Bibek Rout
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

import src.diagnostics.logger as diag


def compute_historical_var(series, confidence=0.95):
    """
    Computes Historical Value-at-Risk (VaR) percentage at given confidence level (e.g. 0.95 or 0.99).
    VaR_alpha = -Quantile_{1 - confidence}(returns)
    """
    cleaned = series.dropna()
    alpha = 1.0 - confidence
    var_val = -np.quantile(cleaned, alpha)
    return round(float(var_val), 4)


def compute_historical_cvar(series, confidence=0.95):
    """
    Computes Historical Expected Shortfall (CVaR) percentage at given confidence level.
    CVaR_alpha = -Mean(returns | returns <= -VaR_alpha)
    """
    cleaned = series.dropna()
    alpha = 1.0 - confidence
    cutoff = np.quantile(cleaned, alpha)
    tail = cleaned[cleaned <= cutoff]
    if len(tail) == 0:
        cvar_val = -cutoff
    else:
        cvar_val = -tail.mean()
    return round(float(cvar_val), 4)


def run_kupiec_pof_test(returns, var_val, confidence=0.95):
    """
    Executes Kupiec Proportion of Failures (POF) Likelihood Ratio test for VaR backtesting.
    Null hypothesis H0: Actual exception rate p equals expected failure rate alpha = 1 - confidence.
    """
    cleaned = returns.dropna().values
    alpha = 1.0 - confidence
    T = len(cleaned)
    if T == 0:
        return {"Exceptions": 0, "Expected_Exceptions": 0, "Exception_Rate_Pct": 0.0, "LR_Stat": 0.0, "p_value": 1.0, "Decision": "PASS"}

    # Exception occurs when daily return is worse than -VaR
    exceptions = np.sum(cleaned < -abs(var_val))
    N = int(exceptions)
    p_hat = N / T

    if N == 0:
        lr_stat = -2.0 * T * np.log(1.0 - alpha)
    elif N == T:
        lr_stat = -2.0 * T * np.log(alpha)
    else:
        num = ((1.0 - alpha) ** (T - N)) * (alpha ** N)
        den = ((1.0 - p_hat) ** (T - N)) * (p_hat ** N)
        if den <= 0 or num <= 0:
            lr_stat = 0.0
        else:
            lr_stat = -2.0 * np.log(num / den)

    p_val = 1.0 - chi2.cdf(max(0.0, lr_stat), df=1)
    is_valid = p_val > 0.05

    return {
        "Total_Observations": T,
        "Exceptions_Count": N,
        "Expected_Exceptions": round(T * alpha, 1),
        "Exception_Rate_Pct": round(p_hat * 100.0, 2),
        "Kupiec_LR_Stat": round(float(lr_stat), 4),
        "p_value": round(float(p_val), 4),
        "Decision": "PASS (Valid VaR)" if is_valid else "REJECT (Uncalibrated VaR)"
    }


def run_christoffersen_test(returns, var_val, confidence=0.95):
    """
    Executes Christoffersen Independence Test for VaR exception clustering.
    Null hypothesis H0: VaR exceptions are independent across consecutive days.
    """
    cleaned = returns.dropna().values
    alpha = 1.0 - confidence
    T = len(cleaned)
    if T <= 1:
        return {"LR_Ind_Stat": 0.0, "p_value": 1.0, "Decision": "PASS"}

    hits = (cleaned < -abs(var_val)).astype(int)
    
    # Transition counts: n_ij = count of state i followed by state j
    n00, n01, n10, n11 = 0, 0, 0, 0
    for t in range(T - 1):
        if hits[t] == 0 and hits[t+1] == 0:
            n00 += 1
        elif hits[t] == 0 and hits[t+1] == 1:
            n01 += 1
        elif hits[t] == 1 and hits[t+1] == 0:
            n10 += 1
        elif hits[t] == 1 and hits[t+1] == 1:
            n11 += 1

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    p_hat = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    # Likelihood ratio under independence
    L_null = ((1.0 - p_hat) ** (n00 + n10)) * (p_hat ** (n01 + n11)) if 0 < p_hat < 1 else 1.0
    L_alt = ((1.0 - p01) ** n00) * (p01 ** n01) * ((1.0 - p11) ** n10) * (p11 ** n11) if (0 < p01 < 1 and 0 < p11 < 1) else L_null

    if L_null <= 0 or L_alt <= 0 or L_null == L_alt:
        lr_ind = 0.0
    else:
        lr_ind = -2.0 * np.log(L_null / L_alt)

    p_val = 1.0 - chi2.cdf(max(0.0, lr_ind), df=1)
    is_independent = p_val > 0.05

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "LR_Ind_Stat": round(float(lr_ind), 4),
        "p_value": round(float(p_val), 4),
        "Decision": "PASS (No Clustering)" if is_independent else "REJECT (Exceptions Clustered)"
    }


def run_full_tail_risk_suite(returns_df, confidence_levels=[0.95, 0.99]):
    """
    Master Tail Risk & VaR Backtesting Analysis across all sectors.
    """
    with diag.DiagnosticTimer("Full Tail Risk & VaR Analysis Suite"):
        records = []
        for col in returns_df.columns:
            s = returns_df[col]
            for conf in confidence_levels:
                var_val = compute_historical_var(s, confidence=conf)
                cvar_val = compute_historical_cvar(s, confidence=conf)
                kup_res = run_kupiec_pof_test(s, var_val, confidence=conf)
                chr_res = run_christoffersen_test(s, var_val, confidence=conf)
                records.append({
                    "Sector": col,
                    "Confidence": f"{int(conf * 100)}%",
                    "VaR (%)": var_val,
                    "CVaR (%)": cvar_val,
                    "Exceptions": kup_res["Exceptions_Count"],
                    "Expected": kup_res["Expected_Exceptions"],
                    "Exception_Rate": f"{kup_res['Exception_Rate_Pct']}%",
                    "Kupiec_pVal": kup_res["p_value"],
                    "Kupiec_Status": kup_res["Decision"],
                    "Christoffersen_pVal": chr_res["p_value"],
                    "Independence_Status": chr_res["Decision"]
                })
        return pd.DataFrame(records)
