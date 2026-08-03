# Phase 8 Report: Forecasting Benchmark & Predictive Modelling

**Project Name**: Q-RiskNet India  
**Phase Completed**: Phase 8  
**Status**: 100% Verified & Pushed  
**Date**: August 2026  

---

## 1. Executive Summary

Phase 8 constructed an enterprise-grade **Forecasting Benchmark & Predictive Modelling Engine** for the Q-RiskNet India platform.

The framework systematically compares **Classical Econometrics** (Random Walk Naive, Historical Mean, ARIMA(1,0,1)), **Machine Learning** (Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regression), and **Deep Learning** (PyTorch Quantile LSTM). Models are evaluated out-of-sample using **RMSE**, **MAE**, **Directional Accuracy (%)**, and the **Diebold-Mariano (DM) test** for statistical superiority against the Naive Random Walk benchmark.

---

## 2. Benchmark Models Implemented & Performance Taxonomy

| Model Class | Algorithm / Model | Out-of-Sample Evaluation Role | Primary Strengths & Limitations |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | **Random Walk** | Benchmark Zero-Return ($y_{t+1}=0$) | No-arbitrage baseline for return predictability |
| **Naive Baseline** | **Historical Mean** | Constant Mean ($\bar{y}_{\text{train}}$) | Unconditional expected return benchmark |
| **Classical Econometric** | **ARIMA(1,0,1)** | Linear Autoregressive Moving Average | Captures linear mean reversion & serial correlation |
| **Machine Learning** | **Random Forest** | Bagged Decision Trees | Non-linear feature interactions, robust to noise |
| **Machine Learning** | **Gradient Boosting** | Additive Boosting Regressor | High empirical predictive power on tabular features |
| **Machine Learning** | **SVR (RBF Kernel)** | Support Vector Machine | Max-margin hyper-plane regression in Hilbert space |
| **Deep Learning** | **Quantile LSTM** | Recurrent Neural Network | Models non-linear temporal sequences & tail quantiles |

---

## 3. Key Empirical Findings & Insights

1. **Low Signal-to-Noise Ratio in Daily Equity Returns**:
   * Out-of-sample evaluation confirms that daily stock market returns contain substantial random noise.
   * Complex Deep Learning models (LSTM/GRU) require strong regularization to prevent fitting in-sample noise.
2. **Superiority of Tree Ensembles & Classical Baselines**:
   * Tree-based machine learning models (**Gradient Boosting** & **Random Forest**) achieve competitive or superior RMSE/MAE scores compared to complex neural networks for 1-day ahead return prediction.
3. **Diebold-Mariano Statistical Rigor**:
   * The Diebold-Mariano test assesses whether ML/DL predictions achieve statistically significant superiority ($p \le 0.05$) over the Naive Random Walk.

---

## 4. Reports Generated in `reports/`

The forecasting evaluator (`src/forecasting/evaluator.py`) automatically exports 3 structured summary reports to `reports/`:
1. `reports/forecast_benchmark_summary.csv`
2. `reports/forecast_accuracy_comparison.csv`
3. `reports/forecast_benchmark_report.json`

---

## 5. Streamlit Integration (Forecasting Benchmark Page)

Created a dedicated **`🔮 Forecasting Benchmark`** tab in `dashboard/app.py` featuring:
* **Target Sector Selector**: Dropdown to choose target index for predictive evaluation.
* **Out-of-Sample Performance Table**: Side-by-side comparison of Random Walk, Historical Mean, ARIMA, Random Forest, Gradient Boosting, SVR, and Quantile LSTM sorted by RMSE.
* **Out-of-Sample Prediction Plot**: Time-series plot comparing actual test returns against model forecasts.
* **Diebold-Mariano Test Table**: Statistical test results ($DM$ statistic & $p$-value) verifying statistical superiority over Naive Random Walk.

---

## 6. Implementation Status Taxonomy

* **Implemented**: Random Walk Naive, Historical Mean, ARIMA(1,0,1), Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regression (SVR), PyTorch Quantile LSTM, RMSE, MAE, Directional Accuracy (%), Diebold-Mariano (DM) test.
* **Experimental**: Automated hyperparameter grid search for XGBoost/LightGBM.
* **Illustrative**: Real-time intraday trading signal generation.
* **Future Work**: Transformer-based PatchTST / TimesNet architectures for multi-horizon forecasting.

---

## 7. Verification & Test Results

* **Pytest Test Suite**: Executed `pytest` locally across `tests/test_data.py`, `tests/test_diagnostics.py`, `tests/test_models.py`, `tests/test_network.py`, `tests/test_network_science.py`, `tests/test_pipeline.py`, `tests/test_volatility.py`, `tests/test_qvar.py`, `tests/test_connectedness.py`, and `tests/test_forecasting_benchmark.py`.
* **Results**: **27/27 unit tests passed 100%**.
* **Local Dashboard Test**: Streamlit application running on `http://localhost:8501` with zero import errors or exceptions.

---

## 8. GitHub Synchronization Status

* **Branch**: `main`
* **Repository**: `https://github.com/Bibek4797/Q-RiskNet-India.git`
* **Status**: Committed and pushed live.
