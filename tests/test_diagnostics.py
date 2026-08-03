import pytest
import pandas as pd
import numpy as np

from src.econometrics.stationarity import run_adf_test, run_kpss_test, run_zivot_andrews_test, run_full_stationarity_suite
from src.econometrics.autocorr import compute_acf_pacf, run_ljung_box_test, compute_durbin_watson
from src.econometrics.hetero import run_arch_lm_test, compute_rolling_variance
from src.econometrics.distribution import compute_distribution_metrics, get_kde_comparison
from src.econometrics.nonlinearity import run_bds_test
from src.econometrics.structural_breaks import run_cusum_break_test
from src.econometrics.diagnostics_runner import run_all_econometric_diagnostics

def test_stationarity_suite():
    np.random.seed(42)
    series = pd.Series(np.random.normal(0, 1, 100), name="TestSector")
    adf_res = run_adf_test(series)
    assert adf_res["Sector"] == "TestSector"
    assert adf_res["Test"] == "ADF"
    assert adf_res["Decision"] in ["Stationary", "Non-Stationary"]

def test_autocorrelation_suite():
    np.random.seed(42)
    series = pd.Series(np.random.normal(0, 1, 100), name="TestSector")
    lb_res = run_ljung_box_test(series, lags=5)
    assert lb_res["Test"] == "Ljung-Box"
    dw_val = compute_durbin_watson(series)
    assert 1.0 <= dw_val <= 3.0

def test_arch_lm_suite():
    np.random.seed(42)
    series = pd.Series(np.random.normal(0, 1, 100), name="TestSector")
    arch_res = run_arch_lm_test(series, lags=5)
    assert arch_res["Test"] == "ARCH-LM"

def test_distribution_suite():
    np.random.seed(42)
    series = pd.Series(np.random.normal(0, 1, 100), name="TestSector")
    dist_res = compute_distribution_metrics(series)
    assert "Jarque_Bera_Stat" in dist_res
    assert "Tail_Behavior" in dist_res

def test_master_diagnostics_runner():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=50)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1, 50),
        "IT": np.random.normal(0, 1, 50)
    }, index=dates)
    
    diag_results = run_all_econometric_diagnostics(returns, save_reports=True)
    assert "stationarity" in diag_results
    assert "autocorrelation" in diag_results
    assert "heteroskedasticity" in diag_results
    assert "distribution" in diag_results
    assert "nonlinearity" in diag_results
    assert "structural_breaks" in diag_results
