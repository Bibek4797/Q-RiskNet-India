import pytest
import pandas as pd
import numpy as np

from src.forecasting.benchmarks import (
    RandomWalkModel,
    HistoricalMeanModel,
    ARIMABenchmarkModel,
    RandomForestBenchmarkModel,
    GradientBoostingBenchmarkModel,
    SVRBenchmarkModel,
    calculate_forecast_metrics,
    diebold_mariano_test
)
from src.forecasting.evaluator import run_all_forecast_benchmarks

def test_forecast_metrics_and_dm_test():
    np.random.seed(42)
    y_true = np.random.normal(0, 1, 50)
    y_pred1 = y_true + np.random.normal(0, 0.5, 50)
    y_pred2 = y_true + np.random.normal(0, 1.0, 50)

    m1 = calculate_forecast_metrics(y_true, y_pred1)
    assert "RMSE" in m1
    assert "MAE" in m1
    assert "Directional_Accuracy_Pct" in m1
    assert 0.0 <= m1["Directional_Accuracy_Pct"] <= 100.0

    e1 = y_true - y_pred1
    e2 = y_true - y_pred2
    dm_res = diebold_mariano_test(e1, e2)
    assert "dm_stat" in dm_res
    assert "p_value" in dm_res
    assert 0.0 <= dm_res["p_value"] <= 1.0

def test_ml_and_classical_benchmark_models():
    np.random.seed(42)
    y_train = np.random.normal(0, 1, 80)
    X_train = np.random.normal(0, 1, (80, 5))
    X_test = np.random.normal(0, 1, (20, 5))

    rf = RandomForestBenchmarkModel(n_estimators=10)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    assert len(rf_preds) == 20

    gb = GradientBoostingBenchmarkModel(n_estimators=10)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    assert len(gb_preds) == 20

    svr = SVRBenchmarkModel()
    svr.fit(X_train, y_train)
    svr_preds = svr.predict(X_test)
    assert len(svr_preds) == 20

def test_master_forecast_evaluator():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1.5, 100),
        "IT": np.random.normal(0, 1.2, 100)
    }, index=dates)

    res = run_all_forecast_benchmarks(returns, target_sector="Bank", train_ratio=0.80, save_reports=True)
    assert "summary_df" in res
    assert "predictions_df" in res
    assert "dm_df" in res
    assert not res["summary_df"].empty
