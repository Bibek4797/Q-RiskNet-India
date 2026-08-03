import pytest
import pandas as pd
import numpy as np
from src.data.validation import validate_dataset
from src.data.preprocessing import (
    compute_log_returns, 
    compute_daily_simple_returns, 
    compute_rolling_volatility, 
    compute_drawdowns, 
    compute_comprehensive_features
)

def test_data_validation():
    data = pd.DataFrame({
        "Bank": [100.0, 102.0, 101.0, 105.0],
        "IT": [200.0, 204.0, 202.0, 210.0]
    }, index=pd.date_range("2024-01-01", periods=4))
    
    report = validate_dataset(data)
    assert report["is_valid"] is True
    assert report["total_rows"] == 4
    assert report["duplicate_timestamps"] == 0

def test_feature_generation():
    prices = pd.DataFrame({
        "Bank": [100.0, 105.0, 102.0, 108.0, 110.0],
        "IT": [200.0, 195.0, 205.0, 210.0, 215.0]
    }, index=pd.date_range("2024-01-01", periods=5))
    
    features = compute_comprehensive_features(prices, short_window=3, long_window=4)
    assert "prices" in features
    assert "log_returns" in features
    assert "daily_returns" in features
    assert "volatility_20d" in features
    assert "drawdowns" in features
    
    # Check drawdowns non-positive property
    drawdowns = features["drawdowns"]
    assert (drawdowns <= 0).all().all()
