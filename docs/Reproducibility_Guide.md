# Q-RiskNet India — Research Reproducibility Guide

**Author**: Bibek Rout  
**Version**: 1.0.0 (Phase 11 Release Candidate)  
**License**: MIT License  

---

## 📌 Overview

This guide provides step-by-step instructions to reproduce the complete **Q-RiskNet India** econometric, volatility, Quantile VAR (QVAR), financial network topology, and forecasting benchmark pipeline from a completely clean environment.

---

## 💻 Environment Requirements

### 1. Operating System
- **Windows**: Windows 10 / 11 64-bit (PowerShell / Command Prompt / WSL2)
- **Linux**: Ubuntu 20.04 / 22.04 LTS
- **macOS**: macOS 12+ (Intel / Apple Silicon)

### 2. Python Runtime
- **Supported Versions**: Python `3.10.x`, `3.11.x`, `3.12.x`, `3.13.x`
- **Recommended**: Python `3.11.x`

---

## ⚙️ Step-by-Step Reproduction Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/Bibek4797/Q-RiskNet-India.git
cd Q-RiskNet-India
```

### Step 2: Create a Clean Virtual Environment
```bash
# On Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Automated Unit Test Suite
```bash
pytest -v
```
**Expected Output**: All 34 tests pass cleanly in under 30 seconds.

### Step 5: Execute the Quantitative Pipeline via Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🔬 Deterministic Random Seeds

To guarantee deterministic, 100% reproducible results:
- **PyTorch Quantile LSTM**: Fixed seed `torch.manual_seed(42)` and `np.random.seed(42)` in `LSTMQuantileModel.fit()`.
- **Random Forest / Gradient Boosting Benchmarks**: Seed set to `random_state=42`.
- **NetworkX Spring Layout**: Seed set to `seed=42`.

---

## 📁 Expected Output Artifacts

Running the pipeline populates the `reports/` directory with the following CSV and JSON artifacts:

| Output File | Description |
|:---|:---|
| `reports/forecast_benchmark_summary.csv` | Out-of-sample RMSE, MAE, and Directional Accuracy across 7 models |
| `reports/forecast_accuracy_comparison.csv` | Out-of-sample daily prediction series for target sector |
| `reports/forecast_benchmark_report.json` | JSON summary report of best performing model |
| `reports/robustness_window_sensitivity.csv` | TCI mean/std stability across rolling windows ($W \in \{100, 150, 200, 250\}$) |
| `reports/robustness_horizon_sensitivity.csv` | Static TCI stability across forecast horizons ($H \in \{5, 10, 15, 20\}$) |
| `reports/robustness_threshold_sensitivity.csv` | Graph edge threshold sensitivity ($\tau_{\text{edge}} \in \{1.0, 2.0, 5.0\}$) |
| `reports/research_validation_report.json` | Overall hypothesis confirmation summary ($H_1, H_2, H_3$) |
| `reports/network_centrality.csv` | Out-degree, In-degree, Betweenness, Closeness, PageRank rankings |

---

## ⚠️ Known Limitations & Troubleshooting

1. **Internet Dependency**: `yfinance` requires an active internet connection to fetch real-time or updated market data. If offline, existing cached datasets in `data/` or offline mocks will be used.
2. **First Run Model Training**: PyTorch Quantile LSTM training takes ~5–10 seconds per run depending on CPU hardware acceleration.
