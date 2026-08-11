# 🇮🇳 Q-RiskNet India: Quantile Risk & Systemic Network Topology Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/Bibek4797/Q-RiskNet-India)
[![Tests: 37 Passed](https://img.shields.io/badge/Tests-37%20Passed-success.svg)](tests/)

> **An interactive quantitative risk analytics platform for measuring, forecasting, and visualizing sectoral tail-risk spillovers, asymmetric volatility, financial network topology, and out-of-sample portfolio risk across National Stock Exchange (NSE) indices in India.**

---

## 📌 Executive Summary

**Q-RiskNet India** is a quantitative finance research platform designed to analyze systemic risk transmission across Indian equity market sectors. Traditional linear mean-variance models often fail during market crises because cross-sector return correlations increase non-linearly during severe downturns.

By integrating econometric models (**QVAR**, **GJR-GARCH(1,1,1)**), deep learning (**PyTorch Quantile LSTM** under Pinball Loss), graph theory (**Minimum Spanning Trees**, **Spectral Community Detection**), and portfolio optimization (**Markowitz Minimum Variance**, **Risk Parity / ERC**, **Rockafellar-Uryasev CVaR**), the platform quantifies tail-risk transmission dynamics during both normal market conditions ($\tau = 0.50$) and extreme bearish market regimes ($\tau = 0.05$).

---

## 🔬 Key Empirical Capabilities

- **Multi-Quantile Risk Modeling**: Captures tail-risk transmission across normal ($\tau=0.50$) and crisis ($\tau=0.05$) quantile states.
- **Asymmetric Volatility Estimation**: Fits GJR-GARCH(1,1,1) models to quantify leverage effects ($\gamma > 0$) following negative return shocks.
- **Directional Risk Spillovers**: Computes gross (Transmitted/Received) and Net Risk Flow across sector indices.
- **Financial Network Topology**: Prunes spillover matrices into directed graphs and extracts Minimum Spanning Tree (MST) risk backbones.
- **Tail-Risk Portfolio Optimization**: Constructs Minimum Variance, Equal Risk Contribution (ERC) Risk Parity, and CVaR portfolios under position constraints ($w_i \le 0.40$).
- **Chronological Out-of-Sample Backtesting**: Evaluates monthly rebalanced portfolio performance without look-ahead bias.
- **Econometric Validation & Testing**: Conducts Kupiec POF and Christoffersen VaR backtesting, Diebold-Mariano forecasting tests, and parameter robustness checks.

---

## 📐 Methodological Framework

### 1. Data & Log Returns
Log returns are calculated as percentage changes:
$$r_{i,t} = \ln \left( \frac{P_{i,t}}{P_{i,t-1}} \right) \times 100$$

### 2. Quantile VAR (QVAR) Framework
The core systemic spillover engine uses an **equation-by-equation multi-quantile VAR framework** estimated via quantile regression:
$$Q_{\tau}(r_{i,t} \mid \mathcal{F}_{t-1}) = \alpha_i(\tau) + \sum_{p=1}^P \sum_{j=1}^K \phi_{ij,p}(\tau) r_{j,t-p}$$

### 3. Asymmetric Volatility (GJR-GARCH)
Conditional volatility is estimated using GJR-GARCH(1,1,1) to account for asymmetric news responses:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \gamma \epsilon_{t-1}^2 I(\epsilon_{t-1} < 0) + \beta \sigma_{t-1}^2$$

### 4. Directional Connectedness (GIRF)
Spillover transmission utilizes a **simulation-based spillover and connectedness framework inspired by the Diebold-Yılmaz methodology** via Generalized Impulse Response Functions (GIRF) under $+2\sigma$ shocks over horizon $H=10$:
- **FROM Index**: $\text{FROM}_i = \sum_{j \neq i} S_{i,j}$ (Total risk imported)
- **TO Index**: $\text{TO}_j = \sum_{i \neq j} S_{i,j}$ (Total risk exported)
- **Net Risk Flow**: $\text{NET}_i = \text{TO}_i - \text{FROM}_i$
- **Total Connectedness Index (TCI)**: $\text{TCI} = \frac{1}{K} \sum_{i=1}^K \text{FROM}_i$

### 5. Network Science & Backbone Pruning
Adjacency matrices are constructed by thresholding spillovers ($\tau_{\text{edge}}$). Spectral clustering via graph Laplacian eigengap identifies sector communities, while the Minimum Spanning Tree (MST) isolates the core risk backbone using Kruskal's algorithm on correlation distance $d_{ij} = \sqrt{2(1 - \rho_{ij})}$.

### 6. Tail-Risk Portfolio Optimization
- **Minimum Variance**: $\min_w w^T \Sigma w$ with positive definite covariance regularization.
- **Risk Parity (ERC)**: Equalizes Percentage Risk Contribution $\text{PRC}_i = \frac{w_i (\Sigma w)_i}{w^T \Sigma w} = \frac{1}{K}$.
- **CVaR Tail Risk**: Minimizes Rockafellar-Uryasev CVaR: $\min_{\gamma, w} \left[ \gamma + \frac{1}{\alpha N} \sum_{n=1}^N \max(0, -w^T r_n - \gamma) \right]$.

---

## 🖥️ Dashboard Structure

The Streamlit dashboard is organized into five executive modules:

1. **Overview**: Executive risk metrics (TCI, Top Transmitter, Top Receiver), auto-computed baselines, and methodology guide.
2. **Market & Risk**: Sector price trends, percentage drawdowns, rolling/GARCH volatility models, and stationarity/normality diagnostics.
3. **Connectedness**: Pruned spillover transmission matrices, gross/net risk flow charts, and dynamic rolling TCI timelines.
4. **Network**: Interactive directed spillover graph, centrality rankings (PageRank, Betweenness), and MST backbone.
5. **Portfolio & Validation**: Chronological out-of-sample backtests, stress test scenarios, walk-forward forecast comparisons, VaR backtests, and hyperparameter sensitivity tables.

---

## 🛠️ Technology Stack

- **Core**: Python 3.10+
- **Frontend / Dashboard**: Streamlit, Plotly Express, Plotly Graph Objects
- **Econometrics & Statistics**: Statsmodels, arch, SciPy, NumPy, Pandas
- **Deep Learning**: PyTorch
- **Machine Learning**: Scikit-Learn
- **Network Science**: NetworkX

---

## ⚡ Quick Start & Running Locally

### 1. Clone Repository
```bash
git clone https://github.com/Bibek4797/Q-RiskNet-India.git
cd Q-RiskNet-India
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 4. Run Unit Tests
```bash
pytest -v
```

---

## 📁 Repository Structure

```text
Q-RiskNet-India/
├── app.py                      # Root Streamlit entrypoint
├── README.md                   # Project documentation
├── requirements.txt            # Python package dependencies
├── pyproject.toml              # Pytest & package configuration
├── configs/
│   └── config.yaml             # Application configuration
├── src/                        # Domain logic package
│   ├── data/                   # Data loader, preprocessing & validation
│   ├── econometrics/           # Stationarity, volatility & tail risk
│   ├── models/                 # QVARModel & PyTorch Quantile LSTM
│   ├── forecasting/            # GIRF spillovers & walk-forward evaluator
│   ├── network/                # Centrality, spectral clustering & MST
│   ├── portfolio/              # MPT, Risk Parity, CVaR & backtesting
│   └── visualization/          # Plotly network graph layout renderers
├── dashboard/                  # Streamlit view controllers
│   ├── app.py                  # Main dashboard router
│   ├── components/             # Reusable UI components & Plotly wrappers
│   └── pages/                  # Page modules (Overview, Market Risk, etc.)
└── tests/                      # Automated unit test suite (37 tests)
```

---

## ⚠️ Disclaimer

*This platform is designed exclusively for academic research, educational, and quantitative demonstration purposes. Content and model outputs do not constitute investment, financial, or trading advice.*
