# Phase 5 Report: Enterprise Quantile VAR (QVAR) Modelling

**Project Name**: Q-RiskNet India  
**Phase Completed**: Phase 5  
**Status**: 100% Verified & Pushed  
**Date**: August 2026  

---

## 1. Executive Summary

Phase 5 constructed an enterprise-grade **Quantile Vector Autoregression (QVAR) Modelling Engine** for the Q-RiskNet India platform.

The framework estimates and evaluates vector autoregressive parameters across 7 distinct conditional quantiles ($\tau \in \{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95\}$). This reveals how cross-sector shock transmission changes between extreme bearish crash states ($\tau = 0.05$), normal trading regimes ($\tau = 0.50$), and bullish market rallies ($\tau = 0.95$). Zero connectedness metrics, network community algorithms, or machine learning estimators were added in this phase, maintaining strict phased modularity.

---

## 2. QVAR Framework & Formulations

| Quantile ($\tau$) | Market State | Econometric & Portfolio Significance |
| :--- | :--- | :--- |
| **$\tau = 0.05$** | **Extreme Bearish (Crash)** | Evaluates systemic contagion & tail-risk shock transmission. |
| **$\tau = 0.10$** | **Bearish (Downturn)** | Measures cross-sector spillover during elevated market stress. |
| **$\tau = 0.50$** | **Median (Normal)** | Central tendency relationship (equivalent to OLS VAR baseline). |
| **$\tau = 0.90$** | **Bullish (Rally)** | Captures market euphoria and expansionary co-movement. |
| **$\tau = 0.95$** | **Extreme Bullish (Boom)** | Evaluates extreme upside co-movement during market surges. |

### 2.1 Optimization Engine
Parameters are estimated equation-by-equation using **Statsmodels QuantReg** by minimizing asymmetric pinball loss:
$$\min_{\Phi_i(\tau)} \sum_{t=p+1}^{T} \rho_\tau \left( y_{i,t} - \mu_i(\tau) - \sum_{j=1}^{p} \mathbf{\Phi}_{i,j}(\tau) y_{t-j} \right)$$

---

## 3. Reports Generated in `reports/`

The QVAR runner (`src/models/qvar_runner.py`) automatically generates 4 structured report files:
1. `reports/qvar_parameter_summary.csv`
2. `reports/qvar_quantile_comparison.csv`
3. `reports/qvar_girf_responses.csv`
4. `reports/qvar_diagnostics_report.json`

---

## 4. Streamlit Integration (QVAR Analysis Page)

Created a dedicated **`📊 QVAR Analysis`** tab in `dashboard/app.py` featuring:
* **Quantile Slider Controls**: Interactive selection of $\tau \in [0.05, 0.95]$.
* **QVAR Coefficient Matrix Heatmap**: $K \times K$ interactive matrix displaying $\Phi_1(\tau)$.
* **Coefficient Stability Plot**: Line chart tracing how any cross-sector parameter $\Phi_{i,j}(\tau)$ changes across quantiles $\tau$.
* **Generalized Impulse Response (GIRF)**: Interactive line chart displaying multi-step shock propagation over horizon $H=1 \dots 10$ days.

---

## 5. Verification & Test Results

* **Pytest Test Suite**: Executed `pytest` locally across `tests/test_data.py`, `tests/test_diagnostics.py`, `tests/test_models.py`, `tests/test_network.py`, `tests/test_pipeline.py`, `tests/test_volatility.py`, and `tests/test_qvar.py`.
* **Results**: **18/18 unit tests passed 100%**.
* **Local Dashboard Test**: Streamlit application running on `http://localhost:8501` with zero import errors or exceptions.

---

## 6. GitHub Synchronization Status

* **Branch**: `main`
* **Repository**: `https://github.com/Bibek4797/Q-RiskNet-India.git`
* **Status**: Committed and pushed live.
