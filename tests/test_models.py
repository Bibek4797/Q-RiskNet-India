import pytest
import pandas as pd
import numpy as np
from src.models.qvar import QVARModel
from src.models.quantile_lstm import LSTMQuantileModel
from src.forecasting.girf import compute_spillover_matrix, calculate_connectedness_metrics

def test_qvar_fitting_and_spillover():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    data = pd.DataFrame({
        "Bank": np.random.normal(0, 1, 50),
        "IT": np.random.normal(0, 1, 50),
        "Pharma": np.random.normal(0, 1, 50)
    }, index=dates)
    
    model = QVARModel(p=2, quantile=0.5)
    model.fit(data)
    
    spill = compute_spillover_matrix(model, data, horizon=5)
    assert spill.shape == (3, 3)
    
    metrics = calculate_connectedness_metrics(spill)
    assert "TCI" in metrics
    assert 0 <= metrics["TCI"] <= 100

def test_quantile_lstm_fitting():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
    data = pd.DataFrame({
        "Bank": np.random.normal(0, 1, 50),
        "IT": np.random.normal(0, 1, 50)
    }, index=dates)
    
    model = LSTMQuantileModel(seq_len=3, epochs=5, early_stopping=False)
    model.fit(data)
    
    spill = compute_spillover_matrix(model, data, horizon=5)
    assert spill.shape == (2, 2)
