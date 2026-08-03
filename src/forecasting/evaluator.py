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
    RandomForestBenchmarkModel,
    GradientBoostingBenchmarkModel,
    SVRBenchmarkModel,
    calculate_forecast_metrics,
    diebold_mariano_test,
    create_lagged_features
)
from src.models.quantile_lstm import LSTMQuantileModel

def run_all_forecast_benchmarks(returns_df, target_sector, train_ratio=0.80, save_reports=True):
    """
    Master Forecasting Benchmark Evaluator.
    Fits Naive, ARIMA, Random Forest, Gradient Boosting, SVR, and LSTM models out-of-sample.
    Computes RMSE, MAE, Directional Accuracy, and Diebold-Mariano tests against Naive Random Walk.
    """
    with diag.DiagnosticTimer(f"Master Forecasting Benchmark for {target_sector}"):
        X, y, feat_names = create_lagged_features(returns_df, target_sector=target_sector, lags=5)

        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        models = {
            "Random Walk (Naive)": RandomWalkModel(),
            "Historical Mean": HistoricalMeanModel(),
            "ARIMA(1,0,1)": ARIMABenchmarkModel(),
            "Random Forest Regressor": RandomForestBenchmarkModel(),
            "Gradient Boosting Regressor": GradientBoostingBenchmarkModel(),
            "Support Vector Regression": SVRBenchmarkModel()
        }

        results_list = []
        predictions_dict = {"Actual": y_test}
        errors_dict = {}

        # 1. Fit & Evaluate Classical & ML Models
        for name, m in models.items():
            try:
                if isinstance(m, (RandomWalkModel, HistoricalMeanModel, ARIMABenchmarkModel)):
                    m.fit(y_train.values)
                    preds = m.predict(y_test.values)
                else:
                    m.fit(X_train.values, y_train.values)
                    preds = m.predict(X_test.values)

                preds_s = pd.Series(preds, index=y_test.index)
                predictions_dict[name] = preds_s
                errors_dict[name] = y_test.values - preds

                metrics = calculate_forecast_metrics(y_test.values, preds)
                results_list.append({
                    "Target_Sector": target_sector,
                    "Model": name,
                    "Class": "Classical / ML",
                    **metrics
                })
            except Exception as e:
                diag.log_warning(f"Benchmark fitting failed for {name}: {e}")

        # 2. Fit & Evaluate PyTorch Quantile LSTM Model
        try:
            lstm_m = LSTMQuantileModel(seq_len=5, hidden_dim=16, quantile=0.50, epochs=15, early_stopping=True, patience=3)
            lstm_m.fit(returns_df.iloc[:split_idx])
            lstm_preds = lstm_m.forecast(returns_df.iloc[:split_idx], steps=len(y_test))
            if target_sector in lstm_preds.columns:
                l_preds_val = lstm_preds[target_sector].values[:len(y_test)]
                l_preds_s = pd.Series(l_preds_val, index=y_test.index)
                predictions_dict["Quantile LSTM"] = l_preds_s
                errors_dict["Quantile LSTM"] = y_test.values - l_preds_val

                l_metrics = calculate_forecast_metrics(y_test.values, l_preds_val)
                results_list.append({
                    "Target_Sector": target_sector,
                    "Model": "Quantile LSTM",
                    "Class": "Deep Learning",
                    **l_metrics
                })
        except Exception as e:
            diag.log_warning(f"Quantile LSTM benchmark failed: {e}")

        comp_df = pd.DataFrame(results_list).sort_values(by="RMSE")
        preds_df = pd.DataFrame(predictions_dict)

        # Compute Diebold-Mariano Test against Random Walk
        rw_errors = errors_dict.get("Random Walk (Naive)")
        dm_list = []
        if rw_errors is not None:
            for name, errs in errors_dict.items():
                if name != "Random Walk (Naive)":
                    dm_res = diebold_mariano_test(rw_errors, errs)
                    dm_list.append({
                        "Model": name,
                        "DM_Statistic": dm_res["dm_stat"],
                        "DM_p_Value": dm_res["p_value"],
                        "Significantly_Superior": dm_res["p_value"] <= 0.05
                    })

        dm_df = pd.DataFrame(dm_list)

        reports_dir = os.path.join(ROOT_DIR, PATHS.get("reports_dir", "reports"))
        if save_reports:
            os.makedirs(reports_dir, exist_ok=True)
            comp_df.to_csv(os.path.join(reports_dir, "forecast_benchmark_summary.csv"), index=False)
            preds_df.to_csv(os.path.join(reports_dir, "forecast_accuracy_comparison.csv"))

            best_model_name = str(comp_df.iloc[0]["Model"])
            summary_json = {
                "target_sector": target_sector,
                "eval_observations": len(y_test),
                "best_performing_model": best_model_name,
                "best_model_rmse": float(comp_df.iloc[0]["RMSE"]),
                "best_model_directional_accuracy": float(comp_df.iloc[0]["Directional_Accuracy_Pct"]),
                "naive_random_walk_rmse": float(comp_df[comp_df["Model"] == "Random Walk (Naive)"]["RMSE"].iloc[0]) if "Random Walk (Naive)" in comp_df["Model"].values else np.nan
            }
            with open(os.path.join(reports_dir, "forecast_benchmark_report.json"), "w", encoding="utf-8") as f:
                json.dump(summary_json, f, indent=4)

            diag.log_info(f"Saved forecasting benchmark reports to {reports_dir}")

        return {
            "summary_df": comp_df,
            "predictions_df": preds_df,
            "dm_df": dm_df
        }
