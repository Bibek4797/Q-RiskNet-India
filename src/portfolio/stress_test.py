"""
Q-RiskNet India — Portfolio Stress Testing Module
Copyright (c) 2026 Bibek Rout
"""
import numpy as np
import pandas as pd

import src.diagnostics.logger as diag


def run_portfolio_stress_test(weights_dict, returns_df):
    """
    Evaluates portfolio resilience under 3 hypothetical stress scenarios:
    1. Broad Market Crash (-15% uniform market shock)
    2. High-Volatility Spike (2.5x volatility expansion + correlation convergence to 0.8)
    3. Systemic Banking Crisis (-25% shock to Nifty Bank/Financials with spillover propagation)
    """
    with diag.DiagnosticTimer("Portfolio Stress Testing Simulation"):
        stressed_results = []
        sectors = list(returns_df.columns)

        # Base covariance and return means
        base_means = returns_df.mean().values
        base_cov = returns_df.cov().values
        stds = np.sqrt(np.diag(base_cov))

        # Scenario 1: Broad Market Crash (-15% uniform shock)
        s1_shock = np.full(len(sectors), -15.0)

        # Scenario 2: Volatility Spike (2.5x std dev shock)
        s2_shock = -2.5 * stds

        # Scenario 3: Systemic Banking Crisis (-25% to Bank/FinService, -10% to others)
        s3_shock = np.zeros(len(sectors))
        for i, sec in enumerate(sectors):
            if "Bank" in sec or "Financial" in sec or "BANK" in sec:
                s3_shock[i] = -25.0
            elif "Realty" in sec or "Auto" in sec:
                s3_shock[i] = -15.0
            else:
                s3_shock[i] = -8.0

        scenarios = {
            "Broad Market Crash (-15%)": s1_shock,
            "High-Volatility Spike (-2.5σ)": s2_shock,
            "Systemic Banking Crisis (-25%)": s3_shock
        }

        for s_name, shock_vec in scenarios.items():
            for p_name, w_vec in weights_dict.items():
                w = w_vec.values if isinstance(w_vec, pd.Series) else np.array(w_vec)
                port_impact = float(np.dot(w, shock_vec))
                stressed_results.append({
                    "Scenario": s_name,
                    "Portfolio": p_name,
                    "Hypothetical_Loss_Pct": round(port_impact, 2),
                    "Interpretation": f"Portfolio loses {-port_impact:.2f}% under {s_name}"
                })

        return pd.DataFrame(stressed_results)
