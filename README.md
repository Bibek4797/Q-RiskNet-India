# 🕸️ Q-RiskNet India: Quantile LSTM & Network Analytics for NSE Volatility Spillovers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Q-RiskNet India** is an end-to-end econometric and deep-learning framework designed to analyze **systemic risk, non-linear volatility spillovers, and network topology** across key sectoral indices of the National Stock Exchange (NSE) of India (e.g., Nifty Bank, Nifty IT, Nifty Pharma, Nifty Realty, Nifty Auto, etc.).

Building on the Diebold-Yilmaz connectedness methodology, this platform upgrades standard linear econometric models (QVAR) to a non-linear **Quantile LSTM Neural Network** optimized via custom **Pinball Loss**. It allows portfolio managers, risk officers, and researchers to map tail-risk contagion during normal markets ($\tau = 0.50$) vs. extreme bear markets ($\tau = 0.05$).

---

## 🌟 Key Features

1. **Econometric Data Engine & GJR-GARCH Filtering** (`data_manager.py`):
   * Dynamic automated data ingestion via `yfinance` with automated missing-value handling and stationarity checks.
   * **GJR-GARCH(1,1,1)** conditional volatility estimation to capture the asymmetric **leverage effect** (negative market shocks inducing higher volatility than positive shocks).
   * **Jarque-Bera Normality Test** and **Augmented Dickey-Fuller (ADF) Unit Root Test** diagnostic pipeline.

2. **Dual-Model Connectedness Engine** (`models.py`):
   * **Quantile VAR (QVAR)**: Linear benchmark utilizing Statsmodels `QuantReg`.
   * **Quantile LSTM**: Deep Recurrent Neural Network built with PyTorch using custom **Pinball Loss** ($\rho_\tau(u)$) to estimate non-linear conditional quantiles without assuming Gaussian normality.
   * **Generalized Impulse Response Functions (GIRF)**: Simulation-based multi-step variance decomposition to capture directional shock transmission.
   * **Diebold-Yilmaz Spillover Metrics**: Calculates Directional $\text{TO}$, Directional $\text{FROM}$, $\text{NET}$ Connectedness, and the Total Connectedness Index ($\text{TCI}$).

3. **Graph Topology & Network Analytics** (`network_utils.py`):
   * **Interactive Directed Risk Networks**: Built with Plotly, visualizing directional spillover edges and color-coded node roles (Net Transmitters vs. Net Receivers).
   * **Minimum Spanning Tree (MST)**: Computes correlation distance matrices ($d_{ij} = \sqrt{2(1 - \rho_{ij})}$) and extracts Kruskal's MST backbone.
   * **Spectral Clustering**: Partitioning market networks into contagion communities using Normalized Graph Laplacians ($L_{sym} = D^{-1/2} L D^{-1/2}$).

4. **Time-Varying Dynamic Spillovers**:
   * Rolling-window evaluation to track the historical evolution of the Total Connectedness Index ($\text{TCI}$) across macroeconomic cycles and crisis events.

5. **Centralized Diagnostics & Exception Monitor** (`diagnostics.py`):
   * Real-time backend execution logging, operation timers, data dimension tracking, and full traceback reporting.

---

## 📐 Mathematical Framework

### 1. Asymmetric GJR-GARCH(1,1,1) Volatility
$$\sigma_{i,t}^2 = \omega + \alpha \epsilon_{i,t-1}^2 + \gamma I_{t-1} \epsilon_{i,t-1}^2 + \beta \sigma_{i,t-1}^2$$
*Where $I_{t-1} = 1$ if $\epsilon_{t-1} < 0$ (bad news / leverage shock), and $0$ otherwise.*

### 2. PyTorch Quantile Pinball Loss
$$\mathcal{L}_\tau(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N \max \Big( (\tau - 1)(y_i - \hat{y}_i), \; \tau (y_i - \hat{y}_i) \Big)$$

### 3. Generalized Forecast Error Variance Response (GIRF)
$$d_{ij}^\tau(H) = \sum_{h=1}^H \left( \hat{y}_{i, t+h}^{(j)} - \hat{y}_{i, t+h} \right)^2$$

### 4. Total Connectedness Index ($\text{TCI}$)
$$\text{TCI}^\tau = \frac{1}{K - 1} \sum_{i=1}^K \sum_{\substack{j=1 \\ j \neq i}}^K \tilde{d}_{ij}^\tau(H)$$

---

## 📁 Repository Architecture

```text
Indian_Stock_Market_Analysis/
├── app.py              # Main Streamlit web application & tabbed UI layout
├── data_manager.py     # Yahoo Finance scraper, log returns, GJR-GARCH & stats
├── models.py           # PyTorch Quantile LSTM, QVAR, GIRF & Diebold-Yilmaz engine
├── network_utils.py    # Plotly directed graphs, Minimum Spanning Tree & Spectral Clustering
├── diagnostics.py      # Real-time backend logging, timers & traceback monitoring
└── README.md           # Project documentation
```

---

## 🚀 Quick Start & Installation

### Prerequisites
* Python 3.10 or higher
* `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/Bibek4797/Q-RiskNet-India.git
cd Q-RiskNet-India
```

### 2. Install Dependencies
```bash
pip install streamlit yfinance torch statsmodels arch networkx plotly scikit-learn pandas numpy
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your default web browser and navigate to **`http://localhost:8501`**.

---

## 📊 Dashboard Overview

| Tab | Feature Description |
| :--- | :--- |
| **📊 Data Center** | Normalized index prices, Pearson correlation heatmaps, ADF stationarity, and Jarque-Bera normality tests. |
| **📈 Volatility Spillover** | Select Quantile ($\tau=0.05, 0.50, 0.95$), train QVAR/Quantile LSTM, view Net Spillover bar charts and DY tables. |
| **🕸️ Network Topology** | Interactive Directed Plotly Risk Graph, Minimum Spanning Tree (MST) backbone, and Spectral Clustering groups. |
| **🕒 Dynamic TCI** | Rolling-window time-varying TCI evaluation across historical timeframes. |
| **🔧 Diagnostics & Logs** | Real-time console tracking execution timers, data shapes, and full stack tracebacks. |

---

## 📚 Academic References

1. **Kapar, B., Buigut, S., & Billah, S. M. (2025)**. *The short-term reaction of financial markets to the U.S. trade tariff announcement*. Finance Research Letters, 86, 108452.
2. **Lin, Z. L., Ouyang, W. P., & Yu, Q. R. (2024)**. *Risk spillover effects of the Israel–Hamas War on global financial and commodity markets: A time–frequency and network analysis*. Finance Research Letters, 66, 105618.
3. **Diebold, F. X., & Yılmaz, K. (2014)**. *On the network topology of variance decompositions: Measuring the connectedness of financial firms*. Journal of Econometrics, 182(1), 119-134.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
