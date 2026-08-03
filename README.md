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
│   ├── app.py                  # Main Streamlit declarative dashboard application
│   ├── components/             # Reusable UI components
│   │   ├── sidebar.py          # Data & model control panels
│   │   ├── kpi_cards.py        # Systemic risk TCI & transmitter cards
│   │   ├── charts.py           # Plotly price, network & spillover plots
│   │   └── tables.py           # Styler tables & descriptive statistics
│   ├── utils/                  # UI themes & styling
│   │   └── theme.py            # CSS layout & dark mode configuration
│   ├── assets/                 # Custom branding & static media
│   └── pages/                  # Additional view controllers
│
├── src/                        # Modular Core Domain Logic Package
│   ├── data/                   # data_loader.py (Data ingestion & log return processing)
│   ├── econometrics/           # garch.py (GJR-GARCH) & stats.py (ADF, Jarque-Bera)
│   ├── models/                 # qvar.py (QVAR) & quantile_lstm.py (PyTorch Pinball LSTM)
│   ├── forecasting/            # girf.py (GIRF shocks & Diebold-Yilmaz connectedness)
│   ├── network/                # mst.py (Correlation MST) & spectral.py (Eigengap Spectral)
│   ├── visualization/          # plotly_plots.py (Directed network graphs & heatmaps)
│   ├── diagnostics/            # logger.py (Execution timers & exception loggers)
│   ├── utils/                  # helpers.py (Formatting utilities)
│   └── config/                 # settings.py (Central YAML settings loader)
│
├── configs/
│   └── config.yaml             # Centralized YAML Configuration File
│
├── data/                       # Data storage layers
│   ├── raw/                    # Downloaded price series
│   ├── processed/              # Log returns & GARCH volatility estimates
│   └── external/               # Macro & benchmark data
│
├── tests/                      # Pytest Automated Test Suite
│   ├── test_data.py            # Data loading & returns tests
│   ├── test_models.py          # QVAR & Quantile LSTM fitting tests
│   └── test_network.py         # MST & Spectral Clustering tests
│
├── notebooks/                  # Exploratory research notebooks
├── reports/                    # Generated PDF/LaTeX research reports
├── docs/                       # Architecture documentation & phase logs
├── models/                     # Saved PyTorch model checkpoints (.pt)
├── logs/                       # Application runtime logs
├── outputs/                    # Exported risk matrices & CSVs
│
├── .github/
│   └── workflows/ci.yml        # GitHub Actions CI Automation
│
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Build & pytest configuration
├── Dockerfile                  # Container build instructions
├── docker-compose.yml          # Container orchestration
├── .gitignore                  # Git exclusion rules
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

Run the automated test suite locally:
```bash
pytest
```

---

## 📜 License
Distributed under the MIT License. Copyright (c) 2026 **Bibek Rout**. See [LICENSE](LICENSE) for details.
