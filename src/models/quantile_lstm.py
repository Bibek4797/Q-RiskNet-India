import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import src.diagnostics.logger as diag

class PinballLoss(nn.Module):
    """
    Quantile Pinball Loss function for PyTorch.
    """
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
        last_out = out[:, -1, :]
        preds = self.fc(last_out)
        return preds

class LSTMQuantileModel:
    """
    Quantile LSTM Neural Network with Pinball Loss and Early Stopping.
    """
    def __init__(self, seq_len=5, hidden_dim="auto", quantile=0.5, epochs=50, lr=0.01, early_stopping=True, patience=5):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.quantile = quantile
        self.epochs = epochs
        self.lr = lr
        self.early_stopping = early_stopping
        self.patience = patience
        self.model = None
        self.columns = []
        self.means = None
        self.stds = None
        
    def fit(self, df, progress_callback=None):
        torch.manual_seed(42)
        np.random.seed(42)
        self.columns = list(df.columns)
        K = len(self.columns)
        
        if self.hidden_dim == "auto" or self.hidden_dim is None:
            hidden_dim_val = max(16, min(64, int(2 ** np.ceil(np.log2(K * 3)))))
            diag.log_info(f"Auto-selected hidden dimension {hidden_dim_val} for K={K} sector features.")
        else:
            hidden_dim_val = int(self.hidden_dim)
            
        with diag.DiagnosticTimer(f"Quantile LSTM Fitting (seq={self.seq_len}, max_epochs={self.epochs}, hidden={hidden_dim_val})"):
            if len(df) <= self.seq_len:
                diag.log_error(f"Length of data ({len(df)}) is too short for seq_len={self.seq_len}")
                raise ValueError(f"Data length ({len(df)}) must be greater than sequence length (Lags={self.seq_len})")
            
            self.means = df.mean()
            self.stds = df.std().replace(0, 1.0)
            
            norm_df = (df - self.means) / self.stds
            
            if norm_df.isna().any().any():
                diag.log_warning("Normalized data contains NaNs. Filling them to prevent weight corruption.")
                norm_df = norm_df.ffill().bfill().fillna(0.0)
            
            X, Y = [], []
            for i in range(len(norm_df) - self.seq_len):
                X.append(norm_df.iloc[i : i+self.seq_len].values)
                Y.append(norm_df.iloc[i+self.seq_len].values)
                
            X_t = torch.tensor(np.array(X), dtype=torch.float32)
            Y_t = torch.tensor(np.array(Y), dtype=torch.float32)
            
            input_dim = len(self.columns)
            output_dim = input_dim
            
            self.model = LSTMNet(input_dim, hidden_dim_val, output_dim)
            criterion = PinballLoss(self.quantile)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            dataset = TensorDataset(X_t, Y_t)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            self.model.train()
            best_loss = float('inf')
            no_improve_count = 0
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
                    
                if self.early_stopping:
                    if final_loss < best_loss - 1e-4:
                        best_loss = final_loss
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count >= self.patience:
                            diag.log_info(f"Early stopping triggered at epoch {epoch+1}/{self.epochs}. Loss converged at {best_loss:.6f}")
                            if progress_callback:
                                progress_callback(self.epochs, self.epochs)
                            break
                    
            diag.log_info(f"LSTM Training finished. Final loss: {final_loss:.6f}")
                
    def predict_next(self, history):
        self.model.eval()
        norm_hist = (history - self.means) / self.stds
        input_x = torch.tensor(norm_hist.values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_norm = self.model(input_x).squeeze(0).numpy()
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
