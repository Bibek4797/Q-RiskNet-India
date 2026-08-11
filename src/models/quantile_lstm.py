"""
Q-RiskNet India — PyTorch Quantile LSTM Neural Network
Copyright (c) 2026 Bibek Rout
"""
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
    L_tau(y, y_hat) = max( (tau - 1)*(y - y_hat), tau*(y - y_hat) )
    """
    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile
        
    def forward(self, pred, target):
        error = target - pred
        loss = torch.max((self.quantile - 1.0) * error, self.quantile * error)
        return loss.mean()


class LSTMNet(nn.Module):
    """Single-layer LSTM network with linear output mapping."""
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
    Quantile LSTM Neural Network with Pinball Loss, Chronological Validation Split,
    and Early Stopping on Validation Loss.
    """
    def __init__(self, seq_len=5, hidden_dim="auto", quantile=0.5, epochs=40, lr=0.01,
                 early_stopping=True, patience=5, val_ratio=0.15):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.quantile = quantile
        self.epochs = epochs
        self.lr = lr
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_ratio = val_ratio
        self.model = None
        self.columns = []
        self.means = None
        self.stds = None
        
    def fit(self, df, progress_callback=None):
        """
        Fits the Quantile LSTM model using chronological train/validation split.
        Early stopping strictly monitors validation Pinball Loss to prevent overfitting.
        """
        torch.manual_seed(42)
        np.random.seed(42)
        self.columns = list(df.columns)
        K = len(self.columns)
        
        if self.hidden_dim == "auto" or self.hidden_dim is None:
            hidden_dim_val = max(16, min(64, int(2 ** np.ceil(np.log2(K * 3)))))
        else:
            hidden_dim_val = int(self.hidden_dim)
            
        with diag.DiagnosticTimer(f"Quantile LSTM Fitting (seq={self.seq_len}, hidden={hidden_dim_val}, q={self.quantile:.2f})"):
            if len(df) <= self.seq_len + 10:
                diag.log_error(f"Length of data ({len(df)}) is too short for seq_len={self.seq_len}")
                raise ValueError(f"Data length ({len(df)}) must be greater than sequence length + 10")
            
            # Chronological train / validation split inside training portion
            val_size = int(len(df) * self.val_ratio) if (self.val_ratio > 0 and len(df) > 40) else 0
            if val_size > 0:
                train_df = df.iloc[:-val_size]
                val_df = df.iloc[-val_size:]
            else:
                train_df = df
                val_df = None

            # Compute z-score scaling parameters ONLY on training set to prevent data leakage
            self.means = train_df.mean()
            self.stds = train_df.std().replace(0, 1.0)
            
            norm_train = (train_df - self.means) / self.stds
            norm_train = norm_train.ffill().bfill().fillna(0.0)
            
            X_tr, Y_tr = [], []
            for i in range(len(norm_train) - self.seq_len):
                X_tr.append(norm_train.iloc[i : i+self.seq_len].values)
                Y_tr.append(norm_train.iloc[i+self.seq_len].values)
                
            X_tr_t = torch.tensor(np.array(X_tr), dtype=torch.float32)
            Y_tr_t = torch.tensor(np.array(Y_tr), dtype=torch.float32)
            
            # Build validation tensors if validation set exists
            if val_df is not None and len(val_df) > self.seq_len:
                norm_val = (val_df - self.means) / self.stds
                norm_val = norm_val.ffill().bfill().fillna(0.0)
                X_v, Y_v = [], []
                for i in range(len(norm_val) - self.seq_len):
                    X_v.append(norm_val.iloc[i : i+self.seq_len].values)
                    Y_v.append(norm_val.iloc[i+self.seq_len].values)
                X_v_t = torch.tensor(np.array(X_v), dtype=torch.float32)
                Y_v_t = torch.tensor(np.array(Y_v), dtype=torch.float32)
            else:
                X_v_t, Y_v_t = None, None

            input_dim = len(self.columns)
            output_dim = input_dim
            
            self.model = LSTMNet(input_dim, hidden_dim_val, output_dim)
            criterion = PinballLoss(self.quantile)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            dataset = TensorDataset(X_tr_t, Y_tr_t)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            best_val_loss = float('inf')
            best_weights = None
            no_improve_count = 0
            
            for epoch in range(self.epochs):
                self.model.train()
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    pred = self.model(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # Evaluate Validation Pinball Loss for early stopping
                self.model.eval()
                with torch.no_grad():
                    if X_v_t is not None:
                        val_pred = self.model(X_v_t)
                        current_val_loss = criterion(val_pred, Y_v_t).item()
                    else:
                        tr_pred = self.model(X_tr_t)
                        current_val_loss = criterion(tr_pred, Y_tr_t).item()

                if progress_callback:
                    progress_callback(epoch + 1, self.epochs)
                    
                if self.early_stopping:
                    if current_val_loss < best_val_loss - 1e-4:
                        best_val_loss = current_val_loss
                        best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count >= self.patience:
                            diag.log_info(f"Early stopping triggered at epoch {epoch+1}/{self.epochs} (Val Loss: {best_val_loss:.6f})")
                            if progress_callback:
                                progress_callback(self.epochs, self.epochs)
                            break
            
            if best_weights is not None:
                self.model.load_state_dict(best_weights)
            diag.log_info(f"Quantile LSTM training finished. Best Val Pinball Loss: {best_val_loss:.6f}")
                
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
