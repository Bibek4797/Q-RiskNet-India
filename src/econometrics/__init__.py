"""
Q-RiskNet India — Econometric Diagnostics & Volatility Package
Copyright (c) 2026 Bibek Rout
"""
from .stationarity import run_full_stationarity_suite, run_adf_test, run_kpss_test, run_zivot_andrews_test
from .autocorr import run_full_autocorrelation_suite, compute_acf_pacf, run_ljung_box_test, compute_durbin_watson
from .hetero import run_full_hetero_suite, run_arch_lm_test, compute_rolling_variance
from .distribution import run_full_distribution_suite, compute_distribution_metrics, get_kde_comparison
from .nonlinearity import run_full_nonlinearity_suite, run_bds_test
from .structural_breaks import run_full_structural_breaks_suite, run_cusum_break_test
from .garch import estimate_garch_volatility
from .volatility import (
    fit_arch_model,
    fit_garch_model,
    fit_egarch_model,
    fit_gjr_garch_model,
    compare_volatility_models_for_sector,
    generate_multi_step_volatility_forecast
)
from .tail_risk import (
    compute_historical_var,
    compute_historical_cvar,
    run_kupiec_pof_test,
    run_christoffersen_test,
    run_full_tail_risk_suite
)

__all__ = [
    "run_full_stationarity_suite",
    "run_adf_test",
    "run_kpss_test",
    "run_zivot_andrews_test",
    "run_full_autocorrelation_suite",
    "compute_acf_pacf",
    "run_ljung_box_test",
    "compute_durbin_watson",
    "run_full_hetero_suite",
    "run_arch_lm_test",
    "compute_rolling_variance",
    "run_full_distribution_suite",
    "compute_distribution_metrics",
    "get_kde_comparison",
    "run_full_nonlinearity_suite",
    "run_bds_test",
    "run_full_structural_breaks_suite",
    "run_cusum_break_test",
    "estimate_garch_volatility",
    "fit_arch_model",
    "fit_garch_model",
    "fit_egarch_model",
    "fit_gjr_garch_model",
    "compare_volatility_models_for_sector",
    "generate_multi_step_volatility_forecast",
    "compute_historical_var",
    "compute_historical_cvar",
    "run_kupiec_pof_test",
    "run_christoffersen_test",
    "run_full_tail_risk_suite"
]
