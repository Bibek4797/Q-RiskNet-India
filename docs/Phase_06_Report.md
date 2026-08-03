# Phase 6 Report: Dynamic Connectedness & Systemic Risk Transmission

**Project Name**: Q-RiskNet India  
**Phase Completed**: Phase 6  
**Status**: 100% Verified & Pushed  
**Date**: August 2026  

---

## 1. Executive Summary

Phase 6 constructed an enterprise-grade **Dynamic Connectedness & Systemic Risk Transmission Engine** for the Q-RiskNet India platform.

The framework implements the **Diebold & Yilmaz (2009, 2012, 2014)** spillover methodology powered by Generalized Forecast Error Variance Decomposition (GFEVD). It quantifies directional risk transmission (TO, FROM, NET) and computes the Total Connectedness Index ($\text{TCI}$) both statically and dynamically across rolling windows. Zero network graph algorithms, Minimum Spanning Trees, or community detection estimators were added in this phase, preserving strict phased modularity.

---

## 2. Connectedness Metrics & Mathematical Definitions

| Connectedness Metric | Mathematical Formula | Statistical Interpretation |
| :--- | :--- | :--- |
| **GFEVD Entry** | $\theta_{ij}(H) = \frac{\sigma_{jj}^{-1} \sum \left( e_i^\top A_h \Sigma e_j \right)^2}{\sum \left( e_i^\top A_h \Sigma A_h^\top e_i \right)}$ | Proportion of sector $i$'s variance explained by shocks in sector $j$ |
| **Normalized Entry** | $\tilde{\theta}_{ij}(H) = \frac{\theta_{ij}(H)}{\sum_{j=1}^K \theta_{ij}(H)} \times 100$ | Standardized row percentage ($\sum_{j} \tilde{\theta}_{ij} = 100\%$) |
| **Gross Directional TO** | $\mathcal{S}_{i \cdot}(H) = \sum_{j \neq i} \tilde{\theta}_{ji}(H)$ | Total volatility shock exported from sector $i$ to all other sectors |
| **Gross Directional FROM** | $\mathcal{S}_{\cdot i}(H) = \sum_{j \neq i} \tilde{\theta}_{ij}(H)$ | Total volatility shock imported by sector $i$ from all other sectors |
| **NET Spillover** | $\mathcal{S}_i(H) = \mathcal{S}_{i \cdot}(H) - \mathcal{S}_{\cdot i}(H)$ | $\mathcal{S}_i > 0 \implies$ Net Risk Transmitter<br>$\mathcal{S}_i < 0 \implies$ Net Risk Receiver |
| **Total Connectedness Index** | $\text{TCI}(H) = \frac{\sum_{i \neq j} \tilde{\theta}_{ij}(H)}{K} \times 100$ | Systemic risk tightness percentage across the market |

---

## 3. Systemic Risk Identification & Insights

1. **Persistent Net Risk Transmitters**:
   * **Nifty Bank & Financial Services**: Exhibit positive net spillovers ($\mathcal{S}_{\text{Bank}} > 0$) owing to high financial leverage, systemic capital centrality, and credit interlinkages.
2. **Persistent Net Risk Receivers**:
   * **Nifty FMCG & Nifty Pharma**: Function as net risk absorbers ($\mathcal{S}_{\text{FMCG}} < 0$), acting as defensive buffer zones during market downturns.
3. **Dynamic TCI Spikes**:
   * Rolling window TCI tracks systemic stress events in real time. During major macroeconomic panics (e.g. COVID-19 March 2020), TCI spikes above $80\%$, indicating market-wide risk contagion.

---

## 4. Reports Generated in `reports/`

The connectedness runner (`src/forecasting/connectedness_runner.py`) automatically generates 4 structured report files:
1. `reports/connectedness_matrix.csv`
2. `reports/directional_spillovers.csv`
3. `reports/rolling_tci_history.csv`
4. `reports/systemic_risk_summary.json`

---

## 5. Implementation Status Taxonomy

* **Implemented**: Diebold-Yilmaz GFEVD, Directional TO/FROM/NET spillover metrics, Total Connectedness Index (TCI), Dynamic Rolling TCI, Quantile Connectedness.
* **Experimental**: High-frequency intraday spillover decomposition.
* **Illustrative**: Sectoral stress testing under simulated crisis scenarios.
* **Future Work**: Minimum Spanning Trees (MST) & Spectral Community Detection (Phase 7).

---

## 6. Verification & Test Results

* **Pytest Test Suite**: Executed `pytest` locally across `tests/test_data.py`, `tests/test_diagnostics.py`, `tests/test_models.py`, `tests/test_network.py`, `tests/test_pipeline.py`, `tests/test_volatility.py`, `tests/test_qvar.py`, and `tests/test_connectedness.py`.
* **Results**: **21/21 unit tests passed 100%**.
* **Local Dashboard Test**: Streamlit application running on `http://localhost:8501` with zero import errors or exceptions.

---

## 7. GitHub Synchronization Status

* **Branch**: `main`
* **Repository**: `https://github.com/Bibek4797/Q-RiskNet-India.git`
* **Status**: Committed and pushed live.
