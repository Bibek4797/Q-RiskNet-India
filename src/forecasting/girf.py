import numpy as np
import pandas as pd
import src.diagnostics.logger as diag

def compute_spillover_matrix(model, data, horizon=10):
    """
    Computes directional risk spillovers using Generalized Impulse Response Simulation (GIRF).
    """
    sectors = list(data.columns)
    K = len(sectors)
    stds = data.std()
    
    lag_depth = model.p if hasattr(model, 'p') else model.seq_len
    history = data.iloc[-max(lag_depth, 10):]
    
    with diag.DiagnosticTimer(f"GIRF Spillover Simulation (K={K}, H={horizon})"):
        base_fc = model.forecast(history, steps=horizon)
        
        if base_fc.isna().any().any():
            diag.log_warning("Baseline forecast contains NaNs. Filling NaNs to prevent GFEVD failure.")
            base_fc = base_fc.ffill().bfill().fillna(0.0)
            
        raw_spillovers = np.zeros((K, K))
        
        for j, trans_sector in enumerate(sectors):
            shocked_history = history.copy()
            shock_val = 2.0 * stds[trans_sector]
            last_date = shocked_history.index[-1]
            shocked_history.loc[last_date, trans_sector] = shocked_history.loc[last_date, trans_sector] + shock_val
            
            shocked_fc = model.forecast(shocked_history, steps=horizon)
            
            if shocked_fc.isna().any().any():
                shocked_fc = shocked_fc.ffill().bfill().fillna(0.0)
            
            for i, rec_sector in enumerate(sectors):
                sq_dev = (shocked_fc[rec_sector].values - base_fc[rec_sector].values) ** 2
                raw_spillovers[i, j] = np.sum(sq_dev)
                
        row_sums = np.sum(raw_spillovers, axis=1, keepdims=True)
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
    
    from_metrics = spillover_df.apply(lambda row: sum(row[col] for col in sectors if col != row.name), axis=1)
    
    to_metrics = pd.Series(0.0, index=sectors)
    for col in sectors:
        to_metrics[col] = sum(spillover_df.loc[row, col] for row in sectors if row != col)
        
    net_metrics = to_metrics - from_metrics
    
    total_non_diag = 0.0
    for row in sectors:
        for col in sectors:
            if row != col:
                total_non_diag += spillover_df.loc[row, col]
    
    tci = total_non_diag / K if K > 0 else 0.0
    
    return {
        "TO": to_metrics,
        "FROM": from_metrics,
        "NET": net_metrics,
        "TCI": tci
    }
