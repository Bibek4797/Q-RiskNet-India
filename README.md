# Q-RiskNet India: Quantile-LSTM & Financial Network Topology Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![CI Pipeline](https://github.com/Bibek4797/Q-RiskNet-India/actions/workflows/ci.yml/badge.svg)](https://github.com/Bibek4797/Q-RiskNet-India/actions/workflows/ci.yml)

## 📌 Executive Summary

**Q-RiskNet India** is an enterprise-grade quantitative risk modeling and network topology platform designed to measure, forecast, and visualize **sectoral risk spillovers** across the National Stock Exchange (NSE) of India. 

Combining classic econometrics (**QVAR**, **GJR-GARCH(1,1,1)**) with deep learning (**Quantile LSTM** under Pinball Loss and Early Stopping) and graph theory (**Minimum Spanning Trees**, **Eigengap Spectral Clustering**), the platform quantifies tail-risk transmission dynamics during both normal market regimes and crisis periods.

---

## 🏗️ Enterprise Architecture & Project Structure

```text
Q-RiskNet-India/
│
├── dashboard/                  # Pure Streamlit Presentation Layer
│   ├── app.py                  # Master entry point, routing & data orchestration
│   ├── pages/                  # Modular View Controllers (11 pages)
│   │   ├── home.py             # Overview, research questions & status
│   │   ├── data_center.py      # Ingestion, validation & price/return views
│   │   ├── diagnostics.py      # Stationarity, autocorrelation, ARCH-LM & KDE
│   │   ├── volatility.py       # Comparative ARCH/GARCH/EGARCH/GJR-GARCH models
│   │   ├── qvar_analysis.py    # QVAR estimation, heatmaps, stability & GIRF
│   │   ├── connectedness.py    # Spillover matrix & dynamic rolling TCI
│   │   ├── network.py          # Directed network graph, centrality & MST
│   │   ├── forecasting.py      # Classical, ML & Quantile LSTM benchmarks
│   │   ├── validation.py       # Sensitivity analysis across W, H, τ_edge
│   │   ├── reports.py          # Centralized report catalog & export center
│   │   └── about.py            # Methodology, tech stack & author info
│   ├── components/             # Reusable Presentation Components
│   │   ├── sidebar.py          # Navigation menu & parameter controls
│   │   ├── kpi_cards.py        # Executive TCI & spillover KPI cards
│   │   ├── charts.py           # Plotly network, spillover & forecast plots
│   │   ├── tables.py           # Styled DataFrames & spillover matrices
│   │   ├── exports.py          # CSV/JSON download buttons
│   │   └── status.py           # Safe execution & status badges
│   └── utils/                  # UI Styling & Layout Configuration
│       └── theme.py            # CSS layout & dark mode configuration
│
├── src/                        # Modular Core Domain Logic Package
│   ├── data/                   # Data ingestion, preprocessing & validation
│   ├── econometrics/           # Stationarity, volatility, autocorrelation & ARCH-LM
│   ├── models/                 # QVARModel & LSTMQuantileModel
│   ├── forecasting/            # GIRF spillovers & Diebold-Yilmaz TCI
│   ├── network/                # Centrality, spectral clustering & MST
│   ├── visualization/          # Plotly network graph renderers
│   ├── diagnostics/            # Execution timers & logger
│   └── config/                 # YAML settings loader
│
├── configs/
│   └── config.yaml             # Centralized YAML Configuration File
│
├── tests/                      # Pytest Automated Test Suite (34 tests)
│   ├── test_connectedness.py   # Static & rolling TCI tests
│   ├── test_dashboard_structure.py # Modular dashboard import tests
│   ├── test_data.py            # Data pipeline & return calculation tests
│   ├── test_diagnostics.py     # Stationarity & heteroskedasticity tests
│   ├── test_forecasting_benchmark.py # Forecast metrics & DM tests
│   ├── test_models.py          # QVAR & Quantile LSTM tests
│   ├── test_network_science.py # Network centrality & MST tests
│   ├── test_qvar.py            # Multi-quantile QVAR tests
│   ├── test_validation.py      # Sensitivity analysis tests
│   └── test_volatility.py      # Volatility fitting & forecast tests
│
├── reports/                    # Generated CSV/JSON research outputs
├── docs/                       # Methodology documentation & phase logs
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Build & pytest configuration
├── Dockerfile                  # Container build instructions
├── LICENSE                     # Copyright (c) 2026 Bibek Rout
└── README.md                   # Project documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Bibek4797/Q-RiskNet-India.git
cd Q-RiskNet-India
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your web browser at **`http://localhost:8501`**.

---

## 🧪 Testing & Code Quality

Run the full automated test suite using `pytest`:
```bash
pytest -v
```
All **34 unit tests** pass cleanly.


Run the automated test suite locally:
```bash
pytest
```

---

## 📜 License
Distributed under the MIT License. Copyright (c) 2026 **Bibek Rout**. See [LICENSE](LICENSE) for details.
