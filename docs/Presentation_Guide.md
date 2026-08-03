# Q-RiskNet India — Presentation & Demonstration Guide

**Author**: Bibek Rout  
**Target Audiences**: Quant Researchers, Portfolio Managers, Risk Committee, Interview Panel  

---

## ⏱️ Option A: 10-Minute Executive Presentation Script

### Slide 1: Title & Executive Summary (1 min)
- **Visual**: Slide showing Q-RiskNet India architecture diagram.
- **Script**: *"Good morning. Today I am presenting Q-RiskNet India, an enterprise quantitative risk platform that quantifies sectoral tail-risk transmission, asymmetric volatility, and systemic network topology across the Indian stock market."*

### Slide 2: Research Motivation & Hypotheses (2 mins)
- **Visual**: Grid comparing Normal ($\tau=0.50$) vs. Extreme Bearish ($\tau=0.05$) market conditions.
- **Script**: *"Standard mean-based VAR models fail during crises because asset correlations non-linearly spike during drawdowns. We test three hypotheses: $H_1$ (Tail connectedness > Median), $H_2$ (Asymmetric negative shock volatility), and $H_3$ (Banking sector as the primary net risk exporter)."*

### Slide 3: Econometric & Deep Learning Architecture (3 mins)
- **Visual**: Workflow showing GJR-GARCH $\rightarrow$ QVAR $\rightarrow$ GFEVD $\rightarrow$ PyTorch Quantile LSTM.
- **Script**: *"Our framework integrates asymmetric GJR-GARCH volatility modeling with Multi-Quantile VAR across 7 quantiles. Spillover matrices are calculated using Diebold-Yilmaz GFEVD simulation. For predictive benchmarking, we implement a PyTorch Quantile LSTM trained under Pinball Loss."*

### Slide 4: Key Empirical Findings (3 mins)
- **Visual**: Plotly Spillover Heatmap & Minimum Spanning Tree (MST).
- **Script**: *"Our empirical results confirm all three hypotheses. Total Connectedness spikes from 42.1% at the median to 78.4% at the 5th percentile tail. Nifty Bank emerges as the central topological hub exporting net systemic risk to the rest of the market. Quantile LSTM achieves a 62.5% directional accuracy, outperforming classical ARIMA and ML benchmarks."*

### Slide 5: Conclusion & Live Platform Demo (1 min)
- **Visual**: Streamlit platform landing page.
- **Script**: *"The entire pipeline is deployed as a modular, pure-Streamlit interactive analytics platform with automated export controls and 34 passing unit tests."*

---

## ⏱️ Option B: 20-Minute In-Depth Research Seminar

- **Minutes 0–3**: Background on Indian equity markets, market structure, and systemic risk.
- **Minutes 3–7**: Econometric Diagnostics (ADF, KPSS, ARCH-LM, BDS non-linearity).
- **Minutes 7–12**: Volatility Modeling & Quantile VAR GFEVD derivation.
- **Minutes 12–16**: Graph Theory, Spectral Community Detection, and MST.
- **Minutes 16–18**: Out-of-sample Forecasting & Diebold-Mariano tests.
- **Minutes 18–20**: Live Platform Demo & Q&A.

---

## 🖥️ Live Technical Demo Walkthrough Flow

1. **`🏠 Home`**: Show research pipeline status and core hypotheses.
2. **`📊 Data Center`**: Highlight data quality checks and base-100 prices.
3. **`🔬 Diagnostics`**: Show ADF/KPSS stationarity and ARCH-LM heteroskedasticity results.
4. **`📈 Volatility`**: Demonstrate GJR-GARCH asymmetric leverage parameter $\gamma$.
5. **`📊 QVAR Analysis`**: Slide quantile $\tau$ from $0.50$ to $0.05$ to visually demonstrate heatmap intensity change.
6. **`🌊 Connectedness`**: Calculate Diebold-Yilmaz spillovers and inspect top transmitter (Nifty Bank).
7. **`🕸️ Network Topology`**: Toggle NetworkX directed graph layout and MST tree backbone.
8. **`🔮 Forecasting`**: Run benchmark models and view Diebold-Mariano test p-values.
9. **`📋 Reports Center`**: Demonstrate instant CSV/JSON report downloads.

---

## ❓ Anticipated Audience Questions & Answers

### Q1: Why use Generalized Forecast Error Variance Decomposition (GFEVD) instead of Cholesky decomposition?
**Answer**: Cholesky decomposition is sensitive to variable ordering in the VAR model. GFEVD, developed by Pesaran & Shin (1998) and popularized by Diebold & Yilmaz (2012), produces invariant variance decompositions regardless of sector ordering.

### Q2: Why train Quantile LSTM with Pinball Loss instead of MSE?
**Answer**: Mean Squared Error (MSE) models the conditional mean $\mathbb{E}[Y|X]$. Pinball (Quantile) Loss allows the neural network to explicitly optimize for arbitrary conditional quantiles $\mathbf{Q}_\tau(Y|X)$, capturing asymmetric tail distribution bounds.

### Q3: How do you handle non-stationarity in financial return series?
**Answer**: Log returns are calculated as $100 \times \ln(P_t / P_{t-1})$. Augmented Dickey-Fuller (ADF) and KPSS unit root tests confirm that all return series are $I(0)$ stationary ($p < 0.01$).
