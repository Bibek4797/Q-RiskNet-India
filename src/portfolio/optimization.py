"""
Q-RiskNet India — Portfolio Optimization Engine (MPT, Risk Parity, CVaR)
Copyright (c) 2026 Bibek Rout
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import src.diagnostics.logger as diag
from src.econometrics.tail_risk import compute_historical_var, compute_historical_cvar


def evaluate_portfolio_performance(weights, returns_df, rf=0.06):
    """
    Computes annualized return, annualized volatility, Sharpe ratio, Max Drawdown,
    Historical 95% VaR, and Historical 95% CVaR for a portfolio allocation vector.
    """
    weights = np.array(weights)
    port_daily = np.dot(returns_df.values, weights)
    port_series = pd.Series(port_daily, index=returns_df.index)

    ann_return = float(np.mean(port_daily) * 252.0)
    ann_vol = float(np.std(port_daily, ddof=1) * np.sqrt(252.0))
    sharpe = float((ann_return - rf) / ann_vol) if ann_vol > 0 else 0.0

    cum_returns = np.cumprod(1.0 + (port_daily / 100.0))
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max * 100.0
    max_dd = float(np.min(drawdowns))

    var_95 = compute_historical_var(port_series, confidence=0.95)
    cvar_95 = compute_historical_cvar(port_series, confidence=0.95)

    return {
        "Annualized_Return_Pct": round(ann_return, 2),
        "Annualized_Vol_Pct": round(ann_vol, 2),
        "Sharpe_Ratio": round(sharpe, 2),
        "Max_Drawdown_Pct": round(max_dd, 2),
        "VaR_95_Pct": var_95,
        "CVaR_95_Pct": cvar_95,
        "portfolio_returns": port_series
    }


def optimize_minimum_variance(returns_df, max_weight=0.40, min_weight=0.0):
    """
    Markowitz Minimum Variance Portfolio Optimization.
    Minimizes w^T Sigma w subject to sum(w) = 1, min_w <= w_i <= max_w.
    Uses covariance regularization for numerical stability.
    """
    K = returns_df.shape[1]
    cov = returns_df.cov().values
    # Small regularization matrix to ensure positive definiteness
    cov = cov + 1e-8 * np.eye(K)

    def _variance_obj(w):
        return float(np.dot(w.T, np.dot(cov, w)))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((min_weight, max_weight) for _ in range(K))
    w0 = np.ones(K) / K

    with diag.DiagnosticTimer("Markowitz Minimum Variance Portfolio Optimization"):
        res = minimize(_variance_obj, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        w_opt = res.x if res.success else w0
        w_opt = np.maximum(min_weight, np.minimum(max_weight, w_opt))
        w_opt = w_opt / np.sum(w_opt)
        return pd.Series(w_opt, index=returns_df.columns)


def optimize_risk_parity(returns_df, max_weight=0.40, min_weight=0.0):
    """
    Equal Risk Contribution (ERC) Risk Parity Portfolio Optimization.
    Equalizes Percentage Risk Contribution (PRC) across all sector positions.
    """
    K = returns_df.shape[1]
    cov = returns_df.cov().values
    cov = cov + 1e-8 * np.eye(K)

    def _risk_parity_obj(w):
        port_var = float(np.dot(w.T, np.dot(cov, w)))
        if port_var <= 0:
            return 1e6
        # Marginal Risk Contribution = (cov @ w) / port_vol
        # Risk Contribution RC_i = w_i * (cov @ w)_i / port_vol
        # Percentage Risk Contribution PRC_i = RC_i / port_vol = w_i * (cov @ w)_i / port_var
        prc = (w * np.dot(cov, w)) / port_var
        target_prc = 1.0 / K
        return float(np.sum((prc - target_prc) ** 2))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((min_weight, max_weight) for _ in range(K))
    w0 = np.ones(K) / K

    with diag.DiagnosticTimer("Equal Risk Contribution (ERC) Risk Parity Optimization"):
        res = minimize(_risk_parity_obj, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        w_opt = res.x if res.success else w0
        w_opt = np.maximum(min_weight, np.minimum(max_weight, w_opt))
        w_opt = w_opt / np.sum(w_opt)
        return pd.Series(w_opt, index=returns_df.columns)


def optimize_cvar_portfolio(returns_df, max_weight=0.40, min_weight=0.0, confidence=0.95):
    """
    CVaR (Expected Shortfall) Tail-Risk Portfolio Optimization.
    Uses Rockafellar-Uryasev formulation to minimize historical 95% Expected Shortfall.
    """
    K = returns_df.shape[1]
    R = returns_df.values
    alpha = 1.0 - confidence

    def _cvar_obj(w):
        losses = -np.dot(R, w)
        gamma = np.quantile(losses, confidence)
        excess = np.maximum(0.0, losses - gamma)
        return float(gamma + np.mean(excess) / alpha)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((min_weight, max_weight) for _ in range(K))
    w0 = np.ones(K) / K

    with diag.DiagnosticTimer("CVaR (Expected Shortfall) Portfolio Optimization"):
        res = minimize(_cvar_obj, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        w_opt = res.x if res.success else w0
        w_opt = np.maximum(min_weight, np.minimum(max_weight, w_opt))
        w_opt = w_opt / np.sum(w_opt)
        return pd.Series(w_opt, index=returns_df.columns)
