# Econometric Methodology & Statistical Assumption Validation

**Document Version**: 1.0.0  
**Project**: Q-RiskNet India  
**Date**: August 2026  

---

## 1. Executive Summary

Before specifying or estimating advanced quantitative risk models (e.g. QVAR, Quantile LSTM), empirical financial time series must undergo rigorous statistical assumption validation. 

Financial returns violate classical Gaussian assumptions in predictable ways:
1. **Fat Tails & Excess Kurtosis**: Extreme losses occur far more frequently than predicted by normal distribution theory ($K > 3$).
2. **Asymmetric Skewness**: Negative returns exhibit larger magnitudes than positive returns ($\text{Skew} < 0$).
3. **Volatility Clustering**: High-volatility days cluster together, inducing conditional heteroskedasticity ($\text{ARCH}$ effects).
4. **Structural Breaks**: Macroeconomic shocks (e.g. COVID-19 pandemic, interest rate shifts) create regime shifts in time-series dynamics.

This document details the mathematical formulation, statistical hypotheses, and research interpretations for all diagnostic tests implemented in **Phase 3**.

---

## 2. Stationarity Analysis

Stationarity is a fundamental requirement for time-series modeling to avoid spurious regression.

### 2.1 Augmented Dickey-Fuller (ADF) Test
* **Null Hypothesis ($H_0$)**: The time series contains a unit root ($\rho = 1$, Non-Stationary).
* **Alternative ($H_1$)**: The time series is stationary ($\rho < 1$).
* **Regression Model**:
  $$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$$
* **Decision Rule**: Reject $H_0$ if Test Statistic < Critical Value (or $p \le 0.05$).

### 2.2 Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
* **Null Hypothesis ($H_0$)**: The time series is trend-stationary.
* **Alternative ($H_1$)**: The time series contains a unit root (Non-Stationary).
* **Research Rationale**: Combining ADF and KPSS confirms **Confirmatory Stationarity** when ADF rejects $H_0$ and KPSS fails to reject $H_0$.

### 2.3 Zivot-Andrews Test for Unit Root with Structural Break
* **Null Hypothesis ($H_0$)**: The series has a unit root without a structural break.
* **Alternative ($H_1$)**: The series is trend-stationary with a single endogenously determined structural break in intercept or trend.

---

## 3. Autocorrelation & Serial Dependence

### 3.1 Autocorrelation Function (ACF) & Partial Autocorrelation Function (PACF)
* **ACF ($\rho_k$)**: Measures linear dependence between $y_t$ and $y_{t-k}$.
* **PACF ($\phi_{kk}$)**: Measures dependence between $y_t$ and $y_{t-k}$ after controlling for intermediate lags $y_{t-1}, \dots, y_{t-k+1}$.

### 3.2 Ljung-Box Test for Serial Correlation
* **Null Hypothesis ($H_0$)**: The autocorrelation coefficients up to lag $m$ are jointly equal to zero (White Noise).
* **Test Statistic**:
  $$Q = n(n+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(m)$$

---

## 4. Heteroskedasticity & Volatility Clustering

### 4.1 Engle’s ARCH-LM Test
* **Null Hypothesis ($H_0$)**: No autoregressive conditional heteroskedasticity ($\alpha_1 = \dots = \alpha_q = 0$).
* **Alternative ($H_1$)**: ARCH effects are present in squared residuals ($\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \epsilon_{t-i}^2$).
* **Research Rationale**: Rejection of $H_0$ justifies conditional volatility modeling (e.g. GJR-GARCH) and non-linear deep learning.

---

## 5. Distribution Analysis & Tail Behavior

### 5.1 Jarque-Bera Test for Normality
* **Null Hypothesis ($H_0$)**: The data is normally distributed ($\text{Skewness}=0, \text{Kurtosis}=3$).
* **Test Statistic**:
  $$JB = \frac{n}{6} \left( S^2 + \frac{(K-3)^2}{4} \right) \sim \chi^2(2)$$
* **Research Rationale**: Rejection of $H_0$ confirms heavy tails, justifying **Quantile Regressions** over traditional Ordinary Least Squares (OLS).

---

## 6. Non-Linearity Testing

### 6.1 Brock-Dechert-Scheinkman (BDS) Test
* **Null Hypothesis ($H_0$)**: The time series observations are independent and identically distributed (i.i.d.).
* **Alternative ($H_1$)**: The time series exhibits non-linear dependence or chaos.

---

## 7. Structural Break Analysis

### 7.1 OLS-CUSUM Test for Parameter Stability
* **Null Hypothesis ($H_0$)**: Model parameters are constant over time.
* **Alternative ($H_1$)**: Parameters exhibit structural instability (regime shifts).
* **Research Rationale**: Identifies macroeconomic event dates (e.g. COVID-19 pandemic March 2020, Global Monetary Tightening 2022).
