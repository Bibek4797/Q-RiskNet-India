# 🇮🇳 Q-RiskNet India: Quantile-LSTM & Financial Network Topology Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![CI Pipeline](https://github.com/Bibek4797/Q-RiskNet-India/actions/workflows/ci.yml/badge.svg)](https://github.com/Bibek4797/Q-RiskNet-India/actions/workflows/ci.yml)
[![Tests: 34 Passed](https://img.shields.io/badge/Tests-34%20Passed-success.svg)](tests/)

> **An Enterprise Quantitative Finance Platform for Measuring, Forecasting, and Visualizing Sectoral Tail-Risk Spillovers, Asymmetric Volatility, and Financial Network Topology across National Stock Exchange (NSE) Indices in India.**

---

## 📌 Executive Summary & Research Motivation

**Q-RiskNet India** is a institutional quantitative risk research platform designed to analyze systemic risk transmission across Indian equity market sectors. Standard mean-based Vector Autoregressive (VAR) models fail during market panics because return correlations non-linearly spike during drawdowns.

By combining classical financial econometrics (**QVAR**, **GJR-GARCH(1,1,1)**), deep learning (**PyTorch Quantile LSTM** under Pinball Loss), and graph theory (**Minimum Spanning Trees**, **Eigengap Spectral Clustering**), the platform quantifies tail-risk transmission dynamics during both normal market regimes ($\tau=0.50$) and extreme bearish crisis states ($\tau=0.05$).

### 🔬 Core Empirical Hypotheses Confirmed:
1. **$H_1$ (Tail Risk Connectedness)**: Systemic risk connectedness during extreme bearish markets ($\tau=0.05$, $\text{TCI}=78.4\%$) significantly exceeds median market conditions ($\tau=0.50$, $\text{TCI}=42.1\%$).
2. **$H_2$ (Asymmetric Leverage Effect)**: Negative return shocks induce statistically significant asymmetric volatility reactions ($\gamma > 0$ in GJR-GARCH and EGARCH).
3. **$H_3$ (Banking Sector Dominance)**: **Nifty Bank** and **Nifty Financial Services** serve as the primary net systemic risk transmitters ($\text{NET} > +18.5\%$) and topological central hubs across Indian financial markets.

---

## 🏛️ System Architecture

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
├── reports/                    # Generated CSV/JSON research outputs
├── docs/                       # Research paper, architecture, guides & reports
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Build & pytest configuration
├── Dockerfile                  # Container build instructions
├── LICENSE                     # Copyright (c) 2026 Bibek Rout
└── README.md                   # Master documentation
```

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Bibek4797/Q-RiskNet-India.git
cd Q-RiskNet-India
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows: .\venv\Scripts\Activate.ps1
# On Linux/macOS: source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your web browser.

---

## 🧪 Testing & Quality Assurance

Run the automated test suite with `pytest`:
```bash
pytest -v
```
All **34 unit tests** pass cleanly in under 30 seconds.

---

## 📚 Documentation Directory

- 📄 [**IEEE Research Manuscript**](docs/Research_Paper.md)
- 🏗️ [**System Architecture Specification**](docs/System_Architecture.md)
- 🖥️ [**Dashboard Architecture Guide**](docs/Dashboard_Architecture.md)
- 🔬 [**Research Reproducibility Guide**](docs/Reproducibility_Guide.md)
- 🎤 [**Presentation & Demo Guide**](docs/Presentation_Guide.md)
- 💼 [**Interview Preparation Guide**](docs/Interview_Guide.md)
- 📄 [**Resume Bullets & Pitch Summaries**](docs/Resume_Bullets.md)
- 🟢 [**Release Candidate Checklist**](docs/Release_Candidate_Checklist.md)
- 📋 [**Final Project Report & Institutional Certification**](docs/Final_Project_Report.md)

---

## 📄 Citation & License

If you use this repository in academic research or quantitative projects, please cite:

```bibtex
@misc{rout2026qrisknet,
  author = {Rout, Bibek},
  title = {Q-RiskNet India: Quantile-LSTM Deep Learning and Financial Network Topology Platform},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Bibek4797/Q-RiskNet-India}}
}
```

This project is released under the **MIT License**. Copyright (c) 2026 Bibek Rout.
