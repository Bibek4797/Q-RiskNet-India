# Forecasting Benchmark & Predictive Modelling Methodology

**Document Version**: 1.0.0  
**Project**: Q-RiskNet India  
**Date**: August 2026  

---

## 1. Executive Summary

Predicting financial asset returns and volatility is one of the most challenging problems in quantitative finance due to the low signal-to-noise ratio, non-stationarity, and market efficiency (Fama, 1970).

While structural econometrics focuses on **explanation** (reconstructing true data-generating processes), predictive modeling focuses strictly on **out-of-sample forecast accuracy**.

**Phase 8** constructs a rigorous multi-model forecasting benchmark comparing **Classical Econometrics** (Random Walk, Historical Mean, ARIMA), **Machine Learning** (Random Forest, Gradient Boosting, SVR), and **Deep Learning** (PyTorch Quantile LSTM/GRU) across Indian sectoral indices.

---

## 2. Benchmark Model Specifications

| Model Class | Algorithm | Model Specification | Key Assumptions & Characteristics |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | **Random Walk** | $\hat{y}_{t+h} = 0$ (Returns) / $y_t$ (Prices) | Efficient market benchmark |
| **Naive Baseline** | **Historical Mean** | $\hat{y}_{t+h} = \frac{1}{T} \sum_{\tau=1}^T y_\tau$ | Constant expected return assumption |
| **Classical Econometric** | **ARIMA($p,d,q$)** | $(1 - \sum \phi_i L^i)(1-L)^d y_t = (1 + \sum \theta_i L^i) \epsilon_t$ | Linear autoregressive integrated moving average |
| **Machine Learning** | **Random Forest** | Ensemble of $N$ decision trees via bagging | Non-linear feature interactions, robust to overfitting |
| **Machine Learning** | **Gradient Boosting** | Sequential additive boosting of weak learners | Optimizes loss function gradient iteratively |
| **Machine Learning** | **SVR (RBF Kernel)** | $\min \frac{1}{2} \|w\|^2 + C \sum \xi_i$ with $\epsilon$-insensitive loss | Support vector regression in high-dimensional kernel space |
| **Deep Learning** | **Quantile LSTM / GRU** | Recurrent neural network with gated memory cells | Sequential temporal dependencies & non-linear tail quantiles |

---

## 3. Forecast Evaluation Metrics

Let $y_t$ be actual out-of-sample values and $\hat{y}_t$ be model predictions over $N_{eval}$ forecast periods.

### 3.1 Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N_{eval}} \sum_{t=1}^{N_{eval}} \left( y_t - \hat{y}_t \right)^2}$$
Penalizes larger forecast errors more heavily than smaller errors.

### 3.2 Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N_{eval}} \sum_{t=1}^{N_{eval}} \left| y_t - \hat{y}_t \right|$$
Linear loss metric, robust to isolated extreme outlier returns.

### 3.3 Directional Accuracy (%)
$$\text{DA} = \frac{1}{N_{eval}} \sum_{t=1}^{N_{eval}} \mathbf{I} \left( \text{sign}(y_t) == \text{sign}(\hat{y}_t) \right) \times 100\%$$
Measures percentage of correctly predicted return signs (trading signal accuracy).

### 3.4 Diebold-Mariano (DM) Test
Tests whether two competing model forecasts have statistically different predictive accuracy:
$$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d}) / N_{eval}}} \sim \mathcal{N}(0, 1)$$
where $d_t = e_{1,t}^2 - e_{2,t}^2$ is the loss differential.

---

## 4. Empirical Philosophy & Findings

1. **Low Signal-to-Noise Ratio**: Daily stock returns are dominated by unpredictable market noise. Complex deep learning models (LSTM/GRU) often overfit historical noise unless heavily regularized.
2. **Superiority of Ensemble Tree & Classical Models**: Simple historical mean or tree ensembles (Random Forest, Gradient Boosting) frequently match or outperform complex deep neural networks in 1-day ahead return forecasting.
3. **Value of Deep Learning**: Deep sequence models provide clear value in multi-step ahead conditional volatility and non-linear quantile risk forecasting.
