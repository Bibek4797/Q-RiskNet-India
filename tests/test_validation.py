import pytest
import pandas as pd
import numpy as np

from src.diagnostics.validation_runner import (
    run_window_sensitivity_analysis,
    run_horizon_sensitivity_analysis,
    run_threshold_sensitivity_analysis,
    run_master_validation_suite
)

def test_window_sensitivity_analysis():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=120)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 120),
        "IT": np.random.normal(0, 1.2, 120)
    }, index=dates)

    df_res = run_window_sensitivity_analysis(returns, windows=[50, 80], p=2, quantile=0.50, horizon=10)
    assert not df_res.empty
    assert "Window_Size_W" in df_res.columns
    assert "Mean_TCI_Pct" in df_res.columns

def test_horizon_sensitivity_analysis():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 100),
        "IT": np.random.normal(0, 1.2, 100)
    }, index=dates)

    df_res = run_horizon_sensitivity_analysis(returns, horizons=[5, 10], p=2, quantile=0.50)
    assert not df_res.empty
    assert "Forecast_Horizon_H" in df_res.columns
    assert "Static_TCI_Pct" in df_res.columns

def test_threshold_sensitivity_analysis():
    spill_df = pd.DataFrame([
        [70.0, 15.0, 15.0],
        [10.0, 75.0, 15.0],
        [5.0, 10.0, 85.0]
    ], index=["Bank", "IT", "Energy"], columns=["Bank", "IT", "Energy"])

    df_res = run_threshold_sensitivity_analysis(spill_df, thresholds=[1.0, 5.0])
    assert not df_res.empty
    assert "Edge_Threshold_Pct" in df_res.columns
    assert "Network_Density" in df_res.columns

def test_master_validation_suite():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=120)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 120),
        "IT": np.random.normal(0, 1.2, 120)
    }, index=dates)

    res = run_master_validation_suite(returns, save_reports=True)
    assert "window_df" in res
    assert "horizon_df" in res
    assert "threshold_df" in res
