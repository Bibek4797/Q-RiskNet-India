import pytest
import pandas as pd
import numpy as np

from src.models.qvar import QVARModel, estimate_multi_quantile_qvar, compute_qvar_girf
from src.models.qvar_runner import run_all_qvar_diagnostics

def test_qvar_single_fit():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 100),
        "IT": np.random.normal(0, 1.2, 100)
    }, index=dates)

    model = QVARModel(p=2, quantile=0.05)
    model.fit(returns)

    assert len(model.models) == 2
    coeff_mat = model.get_coefficient_matrix(lag=1)
    assert coeff_mat.shape == (2, 2)
    assert not coeff_mat.isna().any().any()

def test_multi_quantile_qvar():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 100),
        "IT": np.random.normal(0, 1.2, 100)
    }, index=dates)

    res = estimate_multi_quantile_qvar(returns, p=2, quantiles=[0.05, 0.50, 0.95])
    assert "models" in res
    assert "summary_df" in res
    assert len(res["models"]) == 3

def test_qvar_girf_computation():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 100),
        "IT": np.random.normal(0, 1.2, 100)
    }, index=dates)

    model = QVARModel(p=2, quantile=0.50)
    model.fit(returns)

    girf_df = compute_qvar_girf(model, returns, shocked_sector="Bank", shock_size_std=2.0, horizon=10)
    assert girf_df.shape == (10, 2)
    assert "Bank" in girf_df.columns
    assert "IT" in girf_df.columns

def test_qvar_runner():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 100),
        "IT": np.random.normal(0, 1.2, 100)
    }, index=dates)

    res = run_all_qvar_diagnostics(returns, p=2, quantiles=[0.05, 0.50, 0.95], save_reports=True)
    assert "summary_df" in res
    assert "girf_df" in res
    assert not res["summary_df"].empty
