import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import calculate_log_returns

def test_calculate_log_returns():
    data = pd.DataFrame({
        "Nifty 50": [100.0, 102.0, 101.0, 105.0],
        "Nifty Bank": [200.0, 204.0, 202.0, 210.0]
    })
    returns = calculate_log_returns(data)
    assert not returns.empty
    assert len(returns) == 3
    assert "Nifty 50" in returns.columns
    assert "Nifty Bank" in returns.columns
