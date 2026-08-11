import pytest
import pandas as pd
import numpy as np

from src.econometrics.tail_risk import (
    compute_historical_var,
    compute_historical_cvar,
    run_kupiec_pof_test,
    run_christoffersen_test,
    run_full_tail_risk_suite
)
from src.portfolio.optimization import (
    optimize_minimum_variance,
    optimize_risk_parity,
    optimize_cvar_portfolio,
    evaluate_portfolio_performance
)
from src.portfolio.backtest import run_portfolio_backtest
from src.portfolio.stress_test import run_portfolio_stress_test


def test_tail_risk_metrics():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.05, 1.5, 500), name="Bank")
    
    var_95 = compute_historical_var(returns, confidence=0.95)
    cvar_95 = compute_historical_cvar(returns, confidence=0.95)
    
    assert var_95 > 0
    assert cvar_95 >= var_95

    kup_res = run_kupiec_pof_test(returns, var_95, confidence=0.95)
    assert "Decision" in kup_res
    assert kup_res["Exceptions_Count"] >= 0

    chr_res = run_christoffersen_test(returns, var_95, confidence=0.95)
    assert "Decision" in chr_res


def test_portfolio_optimization():
    np.random.seed(42)
    df = pd.DataFrame({
        "Bank": np.random.normal(0.05, 1.5, 200),
        "IT": np.random.normal(0.03, 1.2, 200),
        "Energy": np.random.normal(0.02, 1.8, 200)
    })

    w_mpt = optimize_minimum_variance(df, max_weight=0.50)
    assert np.isclose(np.sum(w_mpt), 1.0)
    assert (w_mpt <= 0.501).all()

    w_rp = optimize_risk_parity(df, max_weight=0.50)
    assert np.isclose(np.sum(w_rp), 1.0)

    w_cvar = optimize_cvar_portfolio(df, max_weight=0.50)
    assert np.isclose(np.sum(w_cvar), 1.0)

    eval_res = evaluate_portfolio_performance(w_mpt, df)
    assert "Sharpe_Ratio" in eval_res
    assert "VaR_95_Pct" in eval_res


def test_portfolio_backtest_and_stress_test():
    np.random.seed(42)
    df = pd.DataFrame({
        "Bank": np.random.normal(0.05, 1.5, 250),
        "IT": np.random.normal(0.03, 1.2, 250),
        "Energy": np.random.normal(0.02, 1.8, 250)
    })

    bt_res = run_portfolio_backtest(df)
    assert "summary_df" in bt_res
    assert "equity_df" in bt_res

    st_df = run_portfolio_stress_test(bt_res["weights_dict"], df)
    assert not st_df.empty
    assert "Hypothetical_Loss_Pct" in st_df.columns
