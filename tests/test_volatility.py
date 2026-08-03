import pytest
import pandas as pd
import numpy as np

from src.econometrics.volatility import (
    fit_arch_model,
    fit_garch_model,
    fit_egarch_model,
    fit_gjr_garch_model,
    compare_volatility_models_for_sector,
    generate_multi_step_volatility_forecast
)
from src.econometrics.volatility_runner import run_all_volatility_models

def test_volatility_model_fitting():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 1.5, 150), name="TestSector")

    arch_res = fit_arch_model(returns)
    assert arch_res["Model"] == "ARCH(1)"
    assert "AIC" in arch_res

    garch_res = fit_garch_model(returns)
    assert garch_res["Model"] == "GARCH(1,1)"
    assert "Persistence" in garch_res

    gjr_res = fit_gjr_garch_model(returns)
    assert gjr_res["Model"] == "GJR-GARCH(1,1,1)"

def test_volatility_forecasts():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 1.5, 150), name="TestSector")
    garch_res = fit_garch_model(returns)

    fc_dict = generate_multi_step_volatility_forecast(garch_res["fit_result"], horizons=[1, 5, 20])
    assert "Forecast_1d_Vol_Pct" in fc_dict
    assert "Forecast_5d_Vol_Pct" in fc_dict
    assert "Forecast_20d_Vol_Pct" in fc_dict
    assert fc_dict["Forecast_1d_Vol_Pct"] > 0

def test_volatility_runner():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns_df = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 100),
        "IT": np.random.normal(0, 1.2, 100)
    }, index=dates)

    res = run_all_volatility_models(returns_df, save_reports=True)
    assert "comparison" in res
    assert "forecasts" in res
    assert not res["comparison"].empty
