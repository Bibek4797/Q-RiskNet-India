import pytest
import pandas as pd
import numpy as np

from src.models.qvar import QVARModel
from src.forecasting.girf import compute_spillover_matrix, calculate_connectedness_metrics
from src.forecasting.connectedness_runner import run_static_connectedness, run_rolling_connectedness, run_all_connectedness_reports

def test_static_connectedness_metrics():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 100),
        "IT": np.random.normal(0, 1.2, 100),
        "Energy": np.random.normal(0, 1.1, 100)
    }, index=dates)

    model = QVARModel(p=2, quantile=0.50)
    model.fit(returns)

    spill_df = compute_spillover_matrix(model, returns, horizon=10)
    assert spill_df.shape == (3, 3)
    assert np.isclose(spill_df.sum(axis=1).values, 100.0).all()

    metrics = calculate_connectedness_metrics(spill_df)
    assert "TO" in metrics
    assert "FROM" in metrics
    assert "NET" in metrics
    assert "TCI" in metrics
    assert 0 <= metrics["TCI"] <= 100.0

def test_rolling_connectedness():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=120)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 120),
        "IT": np.random.normal(0, 1.2, 120)
    }, index=dates)

    roll_res = run_rolling_connectedness(returns, window_size=50, step_size=10, p=2, quantile=0.50, horizon=10)
    assert "tci_df" in roll_res
    assert "net_df" in roll_res
    assert not roll_res["tci_df"].empty

def test_connectedness_runner_reports():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 100),
        "IT": np.random.normal(0, 1.2, 100)
    }, index=dates)

    res = run_all_connectedness_reports(returns, p=2, quantile=0.50, horizon=10, save_reports=True)
    assert "static" in res
    assert "directional" in res
    assert "rolling" in res
