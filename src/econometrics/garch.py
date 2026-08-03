import numpy as np
import pandas as pd
from arch import arch_model
import src.diagnostics.logger as diag

def estimate_garch_volatility(returns_series):
    """
    Estimates conditional volatility using an asymmetric GJR-GARCH(1,1,1) model.
    """
    sector_name = returns_series.name
    with diag.DiagnosticTimer(f"GJR-GARCH(1,1,1) Estimation for {sector_name}"):
        try:
            model = arch_model(returns_series, vol='Garch', p=1, o=1, q=1, dist='normal', rescale=True)
            res = model.fit(disp='off')
            scale = res.scale if res.scale else 1.0
            if scale == 0:
                scale = 1.0
            vol = res.conditional_volatility / scale
            diag.log_info(f"GJR-GARCH converged for {sector_name}. Rescale scale factor: {scale:.4f}")
            return vol
        except Exception as e:
            diag.log_warning(f"GJR-GARCH failed to converge for {sector_name}. Trying standard GARCH(1,1)...")
            try:
                model_std = arch_model(returns_series, vol='Garch', p=1, q=1, dist='normal', rescale=True)
                res_std = model_std.fit(disp='off')
                scale = res_std.scale if res_std.scale else 1.0
                return res_std.conditional_volatility / scale
            except Exception:
                diag.log_warning(f"Standard GARCH failed for {sector_name}. Fallback to rolling std dev.")
                return returns_series.rolling(window=10).std().fillna(returns_series.std())
