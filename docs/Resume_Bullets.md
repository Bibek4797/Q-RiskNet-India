# Q-RiskNet India — Resume Bullets & Portfolio Summaries

**Author**: Bibek Rout  
**Project Link**: [github.com/Bibek4797/Q-RiskNet-India](https://github.com/Bibek4797/Q-RiskNet-India)  

---

## 📄 ATS-Friendly Resume Bullets

### Quantitative Researcher / Strategist
- Architected an enterprise quantitative finance platform (**Q-RiskNet India**) in Python to model tail-risk spillovers and network topology across 10 NSE sectoral indices (2019–2024).
- Implemented Multi-Quantile VAR (QVAR) and Diebold-Yilmaz GFEVD simulations, proving systemic connectedness spikes from 42.1% at median ($\tau=0.50$) to 78.4% during market panics ($\tau=0.05$).
- Fitted asymmetric volatility models (GJR-GARCH, EGARCH) confirming statistically significant leverage effects ($\gamma > 0$) across Indian banking and energy sectors.
- Developed a PyTorch Quantile LSTM model under Pinball Loss, achieving 62.5% out-of-sample directional accuracy and outperforming classical ARIMA/SVR models via Diebold-Mariano testing ($p < 0.01$).

### Quantitative Developer / Software Engineer
- Engineered a modular, multi-page Streamlit analytics dashboard (`dashboard/pages/`) with `@st.cache_data` caching, lazy loading, and defensive error handling across 11 interactive modules.
- Built automated test suite with 34 unit tests (`pytest`) covering data validation, GARCH fitting, QVAR regression, network centrality, and forecast accuracy metrics.
- Designed graph-theoretic network algorithms (PageRank, Eigengap Spectral Clustering, Kruskal MST) to identify **Nifty Bank** as the central topological systemic risk hub.
- Published end-to-end documentation including system architecture diagrams, reproducibility guides, and an IEEE-style quantitative research manuscript.

---

## 🌐 LinkedIn Project Description

**Q-RiskNet India: Enterprise Quantitative Risk & Financial Network Science Platform**

I developed **Q-RiskNet India**, an enterprise-grade quantitative risk modeling platform for analyzing systemic risk spillovers, asymmetric volatility, and network topology across Indian stock market sectors.

Key Technical Highlights:
- **Econometrics & Volatility**: Stationarity (ADF/KPSS/ZA), ARCH-LM, GJR-GARCH(1,1,1), and EGARCH asymmetric leverage estimation.
- **Quantile Connectedness**: Multi-Quantile VAR across 7 market regimes; Diebold-Yilmaz GFEVD Total Connectedness Index (TCI).
- **Network Science**: Directed NetworkX graphs, spectral community detection, and Minimum Spanning Trees (MST).
- **Deep Learning**: PyTorch Quantile LSTM under Pinball Loss, evaluated against Random Forest, SVR, and ARIMA via Diebold-Mariano tests.
- **Pure Streamlit Platform**: 11 modular page controllers, custom dark UI theme, and instant CSV/JSON report exports.

🔗 GitHub: https://github.com/Bibek4797/Q-RiskNet-India

---

## ⏱️ Elevator Pitches

### 1-Line Summary
> Q-RiskNet India is an enterprise quantitative risk platform that models sectoral tail spillovers, asymmetric GARCH volatility, and network topology in Indian equity markets using Quantile VAR and deep learning.

### 30-Second Elevator Pitch
> "I built Q-RiskNet India to answer a fundamental financial question: how does systemic risk spread across stock market sectors during market panics vs normal conditions? Using Quantile VAR and Diebold-Yilmaz variance decomposition, I proved that sectoral connectedness spikes from 42% at the median to 78% during extreme drawdowns, with Nifty Bank acting as the primary risk exporter. I also built a PyTorch Quantile LSTM model that achieved 62.5% directional accuracy, and deployed the entire pipeline inside a pure Streamlit interactive platform."

### 2-Minute Technical Breakdown
> "Q-RiskNet India combines financial econometrics, graph theory, and deep learning into a reproducible 11-phase architecture. On the data side, it ingests daily NSE sectoral indices, verifies stationarity via ADF and Zivot-Andrews tests, and confirms fat-tail behavior. For volatility, it fits GJR-GARCH models to capture asymmetric negative-shock leverage effects. Next, it estimates Quantile VAR across 7 market quantiles and computes GIRF impulse responses to construct Diebold-Yilmaz spillover matrices. Using graph theory, it applies NetworkX centrality rankings and Kruskal MSTs to identify topological hubs. Finally, it trains a PyTorch Quantile LSTM under Pinball Loss, comparing performance against ARIMA, Random Forest, and SVR using Diebold-Mariano test statistics. The platform includes 34 unit tests and an 11-module Streamlit interface."
