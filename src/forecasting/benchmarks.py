import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from scipy.stats import norm

import src.diagnostics.logger as diag
from src.models.quantile_lstm import LSTMQuantileModel

class RandomWalkModel:
    """Naive Random Walk forecast: return = 0.0"""
    def __init__(self):
        pass
    def fit(self, y):
        pass
    def predict(self, X_eval):
        return np.zeros(len(X_eval))

class HistoricalMeanModel:
    """Historical Mean forecast: return = mean(y_train)"""
    def __init__(self):
        self.mean_val = 0.0
    def fit(self, y):
        self.mean_val = float(np.mean(y))
    def predict(self, X_eval):
        return np.full(len(X_eval), self.mean_val)

class ARIMABenchmarkModel:
    """ARIMA(1,0,1) classical econometric benchmark model"""
    def __init__(self, order=(1, 0, 1)):
        self.order = order
        self.mean_val = 0.0
    def fit(self, y):
        self.mean_val = float(np.mean(y))
        try:
            mod = ARIMA(y, order=self.order)
            self.res = mod.fit()
        except Exception:
            self.res = None
    def predict(self, X_eval):
        if self.res is not None:
            try:
                fc = self.res.forecast(steps=len(X_eval))
                return np.array(fc)
            except Exception:
                pass
        return np.full(len(X_eval), self.mean_val)

class RandomForestBenchmarkModel:
    """Random Forest Regressor ML benchmark"""
    def __init__(self, n_estimators=100, max_depth=5):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def feature_importances(self, feature_names):
        return pd.Series(self.model.feature_importances_, index=feature_names)

class GradientBoostingBenchmarkModel:
    """Gradient Boosting Regressor ML benchmark"""
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.05):
        self.model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def feature_importances(self, feature_names):
        return pd.Series(self.model.feature_importances_, index=feature_names)

class SVRBenchmarkModel:
    """Support Vector Regression (RBF Kernel) ML benchmark"""
    def __init__(self, C=1.0, epsilon=0.1):
        self.model = SVR(C=C, epsilon=epsilon, kernel='rbf')
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

def calculate_forecast_metrics(y_true, y_pred):
    """
    Computes RMSE, MAE, and Directional Accuracy (%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    dir_acc = np.mean(sign_true == sign_pred) * 100.0

    return {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "Directional_Accuracy_Pct": round(float(dir_acc), 2)
    }

def diebold_mariano_test(e1, e2, h=1):
    """
    Computes Diebold-Mariano test statistic and p-value comparing forecast errors e1 and e2.
    """
    e1 = np.array(e1)
    e2 = np.array(e2)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n <= 1:
        return {"dm_stat": 0.0, "p_value": 1.0}

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0:
        return {"dm_stat": 0.0, "p_value": 1.0}

    dm_stat = mean_d / np.sqrt(var_d / n)
    p_val = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return {
        "dm_stat": round(float(dm_stat), 4),
        "p_value": round(float(p_val), 4)
    }

def create_lagged_features(returns_df, target_sector, lags=5):
    """
    Creates feature matrix X (lagged returns of all sectors) and target y.
    """
    df = returns_df.copy()
    feature_cols = []

    for lag in range(1, lags + 1):
        for col in df.columns:
            feat_name = f"{col}_lag{lag}"
            df[feat_name] = df[col].shift(lag)
            feature_cols.append(feat_name)

    df = df.dropna()
    X = df[feature_cols]
    y = df[target_sector]
    return X, y, feature_cols
