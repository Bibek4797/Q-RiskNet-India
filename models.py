import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import diagnostics
import diagnostics as diag

# =====================================================================
# 1. Quantile Vector Autoregression (QVAR) Model
# =====================================================================
class QVARModel:
    def __init__(self, p=2, quantile=0.5):
        self.p = p
        self.quantile = quantile
        self.models = {}
        self.columns = []
        
    def fit(self, df):
        with diag.DiagnosticTimer(f"QVAR(p={self.p}, q={self.quantile}) Model Fitting"):
            self.columns = list(df.columns)
            K = len(self.columns)
            
            # Check length constraint
            if len(df) <= self.p:
                diag.log_error(f"Length of data ({len(df)}) is too short for lag p={self.p}")
                raise ValueError(f"Data length ({len(df)}) must be greater than lag length (p={self.p})")
            
            # Create lag variables
            X_lags = []
            for lag in range(1, self.p + 1):
                lagged = df.shift(lag)
                lagged.columns = [f"{col}_lag{lag}" for col in self.columns]
                X_lags.append(lagged)
                
            X_df = pd.concat(X_lags, axis=1)
            X_df['const'] = 1.0
            
            # Drop rows with NaN due to lagging
            valid_idx = df.index[self.p:]
            X_clean = X_df.loc[valid_idx]
            
            # Check for NaNs or Infinite values
            if X_clean.isna().any().any():
                diag.log_warning("Exogenous variables contain NaNs. Attempting to fill them.")
                X_clean = X_clean.ffill().bfill().fillna(0.0)
            
            for col in self.columns:
                y = df.loc[valid_idx, col]
                
                try:
                    # Quantile regression using statsmodels
                    quant_reg = sm.QuantReg(y, X_clean)
                    res = quant_reg.fit(q=self.quantile)
                    self.models[col] = res.params
                except Exception as e:
                    diag.log_error(f"QuantReg fit failed for sector {col}", e)
                    raise ValueError(f"Failed to fit Quantile Regression for sector {col}: {str(e)}")
                    
            diag.log_info(f"QVAR fitted successfully. Variables: {K}, Equation count: {K}")

    def predict_next(self, history):
        """
        history: DataFrame with shape (P, K) containing the last P observations
        """
        input_dict = {'const': 1.0}
        for lag in range(1, self.p + 1):
            row = history.iloc[-lag]
            for col in self.columns:
                input_dict[f"{col}_lag{lag}"] = row[col]
                
        input_series = pd.Series(input_dict)
        pred = {}
        for col in self.columns:
            params = self.models[col]
            # Ensure index alignment
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


# =====================================================================
# 2. Quantile LSTM Model (PyTorch)
# =====================================================================
class PinballLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile
        
    def forward(self, pred, target):
        error = target - pred
        loss = torch.max((self.quantile - 1) * error, self.quantile * error)
        return loss.mean()

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]  # Output from final time step
        preds = self.fc(last_out)
        return preds

class LSTMQuantileModel:
    def __init__(self, seq_len=5, hidden_dim=16, quantile=0.5, epochs=30, lr=0.01):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.quantile = quantile
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.columns = []
        self.means = None
        self.stds = None
        
    def fit(self, df, progress_callback=None):
        with diag.DiagnosticTimer(f"Quantile LSTM Fitting (seq={self.seq_len}, epochs={self.epochs}, hidden={self.hidden_dim})"):
            self.columns = list(df.columns)
            K = len(self.columns)
            
            if len(df) <= self.seq_len:
                diag.log_error(f"Length of data ({len(df)}) is too short for seq_len={self.seq_len}")
                raise ValueError(f"Data length ({len(df)}) must be greater than sequence length (Lags={self.seq_len})")
            
            self.means = df.mean()
            self.stds = df.std().replace(0, 1.0)
            
            # Scale data for deep learning stability
            norm_df = (df - self.means) / self.stds
            
            # Double check for NaN values in normalized dataframe
            if norm_df.isna().any().any():
                diag.log_warning("Normalized data contains NaNs. Filling them to prevent model weight corruption.")
                norm_df = norm_df.ffill().bfill().fillna(0.0)
            
            # Create sliding window sequences
            X, Y = [], []
            for i in range(len(norm_df) - self.seq_len):
                X.append(norm_df.iloc[i : i+self.seq_len].values)
                Y.append(norm_df.iloc[i+self.seq_len].values)
                
            X_t = torch.tensor(np.array(X), dtype=torch.float32)
            Y_t = torch.tensor(np.array(Y), dtype=torch.float32)
            
            input_dim = len(self.columns)
            output_dim = input_dim
            
            self.model = LSTMNet(input_dim, self.hidden_dim, output_dim)
            criterion = PinballLoss(self.quantile)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            dataset = TensorDataset(X_t, Y_t)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            self.model.train()
            final_loss = 0.0
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                batch_count = 0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    pred = self.model(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1
                
                final_loss = epoch_loss / max(batch_count, 1)
                
                if progress_callback:
                    progress_callback(epoch + 1, self.epochs)
                    
            diag.log_info(f"LSTM Training finished. Final epoch loss: {final_loss:.6f}")
                
    def predict_next(self, history):
        self.model.eval()
        norm_hist = (history - self.means) / self.stds
        input_x = torch.tensor(norm_hist.values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_norm = self.model(input_x).squeeze(0).numpy()
        # Scale back to original units
        pred = pred_norm * self.stds.values + self.means.values
        return pd.Series(pred, index=self.columns)
        
    def forecast(self, history, steps=10):
        current_hist = history.copy()
        forecasts = []
        for h in range(steps):
            next_pred = self.predict_next(current_hist.iloc[-self.seq_len:])
            forecasts.append(next_pred)
            next_df = pd.DataFrame([next_pred])
            next_df.index = [current_hist.index[-1] + pd.Timedelta(days=1)]
            current_hist = pd.concat([current_hist, next_df])
        return pd.DataFrame(forecasts)


# =====================================================================
# 3. GIRF Connectedness Engine (Non-linear Spillover)
# =====================================================================
def compute_spillover_matrix(model, data, horizon=10):
    """
    Computes directional risk spillovers using Generalized Impulse Response simulation.
    
    Returns:
        DataFrame: A K x K matrix where entry (i, j) is the spillover from j to i.
    """
    sectors = list(data.columns)
    K = len(sectors)
    stds = data.std()
    
    # Required lag depth for baseline forecasting
    lag_depth = model.p if hasattr(model, 'p') else model.seq_len
    history = data.iloc[-max(lag_depth, 10):]
    
    with diag.DiagnosticTimer(f"GIRF Spillover Simulation (K={K}, H={horizon})"):
        # 1. Baseline multi-step ahead forecast
        base_fc = model.forecast(history, steps=horizon)
        
        # Verify forecast is not empty
        if base_fc.isna().any().any():
            diag.log_warning("Baseline forecast contains NaNs. Filling NaNs to prevent GFEVD failure.")
            base_fc = base_fc.ffill().bfill().fillna(0.0)
            
        raw_spillovers = np.zeros((K, K))
        
        # 2. Shock each transmitting sector 'j'
        for j, trans_sector in enumerate(sectors):
            shocked_history = history.copy()
            # Inject standard deviation shock (e.g. +2 std dev)
            shock_val = 2.0 * stds[trans_sector]
            # Use explicit .loc to avoid index matching bugs or Copy warnings
            last_date = shocked_history.index[-1]
            shocked_history.loc[last_date, trans_sector] = shocked_history.loc[last_date, trans_sector] + shock_val
            
            # Forecast with the shocked data
            shocked_fc = model.forecast(shocked_history, steps=horizon)
            
            if shocked_fc.isna().any().any():
                shocked_fc = shocked_fc.ffill().bfill().fillna(0.0)
            
            # Measure response in receiving sector 'i'
            for i, rec_sector in enumerate(sectors):
                sq_dev = (shocked_fc[rec_sector].values - base_fc[rec_sector].values) ** 2
                raw_spillovers[i, j] = np.sum(sq_dev)
                
        # 3. Normalize rows to build the Diebold-Yilmaz spillover table
        row_sums = np.sum(raw_spillovers, axis=1, keepdims=True)
        # Avoid dividing by zero
        row_sums[row_sums == 0] = 1e-8
        norm_spillovers = (raw_spillovers / row_sums) * 100
        
        spillover_df = pd.DataFrame(norm_spillovers, index=sectors, columns=sectors)
        diag.log_info("Completed GIRF spillover matrix calculations successfully.")
        return spillover_df

def calculate_connectedness_metrics(spillover_df):
    """
    Computes TO, FROM, NET connectedness and the Total Connectedness Index (TCI).
    """
    sectors = spillover_df.columns
    K = len(sectors)
    
    # FROM others (row sum excluding diagonal)
    from_metrics = spillover_df.apply(lambda row: sum(row[col] for col in sectors if col != row.name), axis=1)
    
    # TO others (column sum excluding diagonal)
    to_metrics = pd.Series(0.0, index=sectors)
    for col in sectors:
        to_metrics[col] = sum(spillover_df.loc[row, col] for row in sectors if row != col)
        
    # NET connectedness
    net_metrics = to_metrics - from_metrics
    
    # Total Connectedness Index (TCI)
    # Sum of all non-diagonal spillovers divided by K (as K-1 in traditional VAR, but K is fine if normalized)
    total_non_diag = 0.0
    for row in sectors:
        for col in sectors:
            if row != col:
                total_non_diag += spillover_df.loc[row, col]
    
    # Paper 1 divides by K-1, let's use K-1 for standard QVAR/DY matching
    tci = total_non_diag / (K - 1) if K > 1 else 0.0
    
    return {
        "TO": to_metrics,
        "FROM": from_metrics,
        "NET": net_metrics,
        "TCI": tci
    }
