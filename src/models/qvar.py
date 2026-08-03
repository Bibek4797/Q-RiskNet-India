import numpy as np
import pandas as pd
import statsmodels.api as sm
import src.diagnostics.logger as diag

class QVARModel:
    """
    Quantile Vector Autoregression (QVAR) Model using Statsmodels QuantReg.
    """
    def __init__(self, p=2, quantile=0.5):
        self.p = p
        self.quantile = quantile
        self.models = {}
        self.columns = []
        
    def fit(self, df):
        with diag.DiagnosticTimer(f"QVAR(p={self.p}, q={self.quantile}) Model Fitting"):
            self.columns = list(df.columns)
            K = len(self.columns)
            
            if len(df) <= self.p:
                diag.log_error(f"Length of data ({len(df)}) is too short for lag p={self.p}")
                raise ValueError(f"Data length ({len(df)}) must be greater than lag length (p={self.p})")
            
            X_lags = []
            for lag in range(1, self.p + 1):
                lagged = df.shift(lag)
                lagged.columns = [f"{col}_lag{lag}" for col in self.columns]
                X_lags.append(lagged)
                
            X_df = pd.concat(X_lags, axis=1)
            X_df['const'] = 1.0
            
            valid_idx = df.index[self.p:]
            X_clean = X_df.loc[valid_idx]
            
            if X_clean.isna().any().any():
                diag.log_warning("Exogenous variables contain NaNs. Attempting to fill them.")
                X_clean = X_clean.ffill().bfill().fillna(0.0)
            
            for col in self.columns:
                y = df.loc[valid_idx, col]
                try:
                    quant_reg = sm.QuantReg(y, X_clean)
                    res = quant_reg.fit(q=self.quantile)
                    self.models[col] = res.params
                except Exception as e:
                    diag.log_error(f"QuantReg fit failed for sector {col}", e)
                    raise ValueError(f"Failed to fit Quantile Regression for sector {col}: {str(e)}")
                    
            diag.log_info(f"QVAR fitted successfully. Variables: {K}, Equation count: {K}")

    def predict_next(self, history):
        input_dict = {'const': 1.0}
        for lag in range(1, self.p + 1):
            row = history.iloc[-lag]
            for col in self.columns:
                input_dict[f"{col}_lag{lag}"] = row[col]
                
        input_series = pd.Series(input_dict)
        pred = {}
        for col in self.columns:
            params = self.models[col]
            aligned_input = input_series[params.index]
            pred[col] = np.dot(params.values, aligned_input.values)
        return pd.Series(pred)
        
    def forecast(self, history, steps=10):
        current_hist = history.copy()
        forecasts = []
        for h in range(steps):
            next_pred = self.predict_next(current_hist.iloc[-self.p:])
            forecasts.append(next_pred)
            next_df = pd.DataFrame([next_pred])
            next_df.index = [current_hist.index[-1] + pd.Timedelta(days=1)]
            current_hist = pd.concat([current_hist, next_df])
        return pd.DataFrame(forecasts)
