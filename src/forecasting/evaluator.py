"""
Q-RiskNet India — Master Walk-Forward & Out-of-Sample Forecasting Evaluator
Copyright (c) 2026 Bibek Rout
"""
import os
import json
import pandas as pd
import numpy as np

from src.config.settings import PATHS, ROOT_DIR
import src.diagnostics.logger as diag
from src.forecasting.benchmarks import (
    RandomWalkModel,
    HistoricalMeanModel,
    ARIMABenchmarkModel,
    SVRBenchmarkModel,
    RandomForestBenchmarkModel,
    calculate_forecast_metrics,
    calculate_pinball_loss,
    diebold_mariano_test,
    create_lagged_features
)
from src.models.quantile_lstm import LSTMQuantileModel


def run_walk_forward_evaluation(returns_df, target_sector, initial_ratio=0.70, step=10, quantile=0.50):
    """
    Executes chronological expanding-window walk-forward forecast evaluation.
    No random cross-validation. Out-of-sample predictions are collected step by step.
    """
    X, y, feat_names = create_lagged_features(returns_df, target_sector=target_sector, lags=5)
    N = len(X)
    start_idx = int(N * initial_ratio)

    models = {
        "Random Walk (Naive)": RandomWalkModel(),
        "Historical Mean": HistoricalMeanModel(),
        "ARIMA(1,0,1)": ARIMABenchmarkModel(),
        "Support Vector Regression": SVRBenchmarkModel()
    }

    preds_records = {m: [] for m in models}
    preds_records["Quantile LSTM"] = []
    actuals = []

    with diag.DiagnosticTimer(f"Walk-Forward Forecast Evaluation for {target_sector} (N={N}, start={start_idx})"):
        # Expand training window chronologically
        for t in range(start_idx, N, step):
            t_end = min(t + step, N)
            X_tr, y_tr = X.iloc[:t], y.iloc[:t]
            X_te, y_te = X.iloc[t:t_end], y.iloc[t:t_end]
            actuals.extend(y_te.values)

            # Fit classical / ML models
            for name, m in models.items():
                try:
                    if isinstance(m, (RandomWalkModel, HistoricalMeanModel, ARIMABenchmarkModel)):
                        m.fit(y_tr.values)
                        fc = m.predict(y_te.values)
                    else:
                        m.fit(X_tr.values, y_tr.values)
                        fc = m.predict(X_te.values)
                    preds_records[name].extend(fc)
                except Exception:
                    preds_records[name].extend(np.full(len(y_te), float(y_tr.mean())))

            # Fit PyTorch Quantile LSTM on expanding history
            try:
                sub_returns = returns_df.iloc[:t]
                lstm_m = LSTMQuantileModel(seq_len=5, hidden_dim=16, quantile=quantile, epochs=15, early_stopping=True, patience=3)
                lstm_m.fit(sub_returns)
                lstm_fc = lstm_m.forecast(sub_returns, steps=len(y_te))
                if target_sector in lstm_fc.columns:
                    preds_records["Quantile LSTM"].extend(lstm_fc[target_sector].values[:len(y_te)])
                else:
                    preds_records["Quantile LSTM"].extend(np.full(len(y_te), float(y_tr.mean())))
            except Exception:
                preds_records["Quantile LSTM"].extend(np.full(len(y_te), float(y_tr.mean())))

        eval_len = len(actuals)
        results_list = []
        errors_dict = {}

        for name, p_list in preds_records.items():
            p_arr = np.array(p_list[:eval_len])
            y_arr = np.array(actuals[:eval_len])
            errors_dict[name] = y_arr - p_arr
            m_dict = calculate_forecast_metrics(y_arr, p_arr, quantile=quantile)
            results_list.append({
                "Target_Sector": target_sector,
                "Model": name,
                "Evaluation": "Chronological Walk-Forward",
                **m_dict
            })

        summary_df = pd.DataFrame(results_list).sort_values(by="RMSE")
        
        # Diebold-Mariano test vs Random Walk
        rw_err = errors_dict.get("Random Walk (Naive)")
        dm_list = []
        if rw_err is not None:
            for name, err in errors_dict.items():
                if name != "Random Walk (Naive)":
                    dm_res = diebold_mariano_test(rw_err, err)
                    dm_list.append({
                        "Model": name,
                        "DM_Statistic": dm_res["dm_stat"],
                        "DM_p_Value": dm_res["p_value"],
                        "Significantly_Superior": dm_res["p_value"] <= 0.05
                    })

        return summary_df, pd.DataFrame(dm_list)


def run_all_forecast_benchmarks(returns_df, target_sector, train_ratio=0.80, save_reports=True):
    """
    Master Forecasting Benchmark Evaluator.
    Runs both out-of-sample split and walk-forward evaluations across benchmarks and Quantile LSTM.
    """
    with diag.DiagnosticTimer(f"Master Forecasting Benchmark Suite for {target_sector}"):
        summary_df, dm_df = run_walk_forward_evaluation(returns_df, target_sector=target_sector, initial_ratio=0.70, step=15, quantile=0.50)

        # Single out-of-sample split predictions for overlay charting
        X, y, feat_names = create_lagged_features(returns_df, target_sector=target_sector, lags=5)
        split_idx = int(len(X) * train_ratio)
        y_test = y.iloc[split_idx:]
        
        # Generate baseline predictions for display chart
        rw = RandomWalkModel()
        rw.fit(y.iloc[:split_idx].values)
        rw_p = rw.predict(y_test.values)

        ar = ARIMABenchmarkModel()
        ar.fit(y.iloc[:split_idx].values)
        ar_p = ar.predict(y_test.values)

        svr = SVRBenchmarkModel()
        svr.fit(X.iloc[:split_idx].values, y.iloc[:split_idx].values)
        svr_p = svr.predict(X.iloc[split_idx:].values)

        lstm_m = LSTMQuantileModel(seq_len=5, hidden_dim=16, quantile=0.50, epochs=20, early_stopping=True, patience=3)
        lstm_m.fit(returns_df.iloc[:split_idx])
        lstm_fc = lstm_m.forecast(returns_df.iloc[:split_idx], steps=len(y_test))
        lstm_p = lstm_fc[target_sector].values[:len(y_test)] if target_sector in lstm_fc.columns else np.zeros(len(y_test))

        preds_df = pd.DataFrame({
            "Actual": y_test,
            "Random Walk": rw_p,
            "ARIMA(1,0,1)": ar_p,
            "SVR": svr_p,
            "Quantile LSTM": lstm_p
        }, index=y_test.index)

        reports_dir = os.path.join(ROOT_DIR, PATHS.get("reports_dir", "reports"))
        if save_reports:
            os.makedirs(reports_dir, exist_ok=True)
            summary_df.to_csv(os.path.join(reports_dir, "forecast_benchmark_summary.csv"), index=False)
            preds_df.to_csv(os.path.join(reports_dir, "forecast_accuracy_comparison.csv"))

            best_row = summary_df.iloc[0]
            summary_json = {
                "target_sector": target_sector,
                "evaluation_method": "Chronological Walk-Forward Expanding Window",
                "best_performing_model": str(best_row["Model"]),
                "best_model_rmse": float(best_row["RMSE"]),
                "best_model_pinball_loss": float(best_row["Pinball_Loss"]),
                "best_model_directional_accuracy": float(best_row["Directional_Accuracy_Pct"])
            }
            with open(os.path.join(reports_dir, "forecast_benchmark_report.json"), "w", encoding="utf-8") as f:
                json.dump(summary_json, f, indent=4)

            diag.log_info(f"Saved walk-forward forecasting benchmark reports to {reports_dir}")

        return {
            "summary_df": summary_df,
            "predictions_df": preds_df,
            "dm_df": dm_df
        }
