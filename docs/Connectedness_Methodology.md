# Dynamic Connectedness & Systemic Risk Transmission Methodology

**Document Version**: 1.0.0  
**Project**: Q-RiskNet India  
**Date**: August 2026  

---

## 1. Executive Summary

Understanding how financial shocks transmit across sectoral indices is essential for systemic risk monitoring, macroprudential policy, and portfolio risk management. 

Traditional correlation analysis measures static linear co-movement but fails to distinguish directionality (which sector transmits shock to which) or track time-varying risk propagation during market crises.

**Phase 6** implements the **Diebold & Yilmaz (2009, 2012, 2014)** spillover framework built on Generalized Forecast Error Variance Decomposition (GFEVD). Combined with our **Quantile VAR (QVAR)** framework, it quantifies directional risk transmission across normal ($\tau=0.50$) and extreme bearish tail-risk ($\tau=0.05$) market regimes.

---

## 2. Diebold-Yilmaz GFEVD Framework

Let $y_t$ be a $K$-variable vector process following a VAR or QVAR process with moving average representation:
$$y_t = \sum_{i=0}^{\infty} A_i \epsilon_{t-i}$$

### 2.1 Generalized Forecast Error Variance Decomposition (GFEVD)
The $H$-step-ahead generalized forecast error variance decomposition entry $\theta_{ij}(H)$ measures the proportion of sector $i$'s forecast error variance contributed by shocks in sector $j$:

$$\theta_{ij}(H) = \frac{\sigma_{jj}^{-1} \sum_{h=0}^{H-1} \left( e_i^\top A_h \Sigma e_j \right)^2}{\sum_{h=0}^{H-1} \left( e_i^\top A_h \Sigma A_h^\top e_i \right)}$$

where $\Sigma$ is the covariance matrix of error vector $\epsilon$, $\sigma_{jj}$ is the standard deviation of error for variable $j$, and $e_i$ is a selection vector.

### 2.2 Normalization
Since rows of the GFEVD matrix do not automatically sum to one ($\sum_{j=1}^K \theta_{ij}(H) \neq 1$), each entry is normalized:

$$\tilde{\theta}_{ij}(H) = \frac{\theta_{ij}(H)}{\sum_{j=1}^{K} \theta_{ij}(H)} \times 100$$

such that $\sum_{j=1}^K \tilde{\theta}_{ij}(H) = 100\%$.

---

## 3. Connectedness Metrics Definitions

| Connectedness Metric | Mathematical Formula | Interpretation |
| :--- | :--- | :--- |
| **Gross Directional TO Others** | $\mathcal{S}_{i \cdot}(H) = \sum_{\substack{j=1 \\ j \neq i}}^{K} \tilde{\theta}_{ji}(H)$ | Total volatility shock transmitted from sector $i$ to all other sectors |
| **Gross Directional FROM Others** | $\mathcal{S}_{\cdot i}(H) = \sum_{\substack{j=1 \\ j \neq i}}^{K} \tilde{\theta}_{ij}(H)$ | Total volatility shock absorbed by sector $i$ from all other sectors |
| **NET Directional Connectedness** | $\mathcal{S}_i(H) = \mathcal{S}_{i \cdot}(H) - \mathcal{S}_{\cdot i}(H)$ | $\mathcal{S}_i > 0 \implies$ Net Risk Transmitter<br>$\mathcal{S}_i < 0 \implies$ Net Risk Receiver |
| **Total Connectedness Index (TCI)** | $\text{TCI}(H) = \frac{\sum_{\substack{i,j=1 \\ i \neq j}}^{K} \tilde{\theta}_{ij}(H)}{K} \times 100$ | Overall systemic tightness / risk integration of the entire system (%) |

---

## 4. Systemic Risk Classification

1. **Persistent Net Transmitters ($\mathcal{S}_i > 0$)**: Sectors that export volatility shocks to the financial system (typically Nifty Bank and Nifty Financial Services in India due to high leverage and credit interlinkages).
2. **Persistent Net Receivers ($\mathcal{S}_i < 0$)**: Sectors that absorb market shocks (e.g., Nifty FMCG or Nifty Pharma behaving as defensive sinks).
3. **Dynamic TCI Spikes**: Sharp upward shifts in rolling TCI identify market stress events (e.g. COVID-19 panic in March 2020 where TCI spiked above $80\%$).

---

## 5. Implementation Status Taxonomy

* **Implemented**: Diebold-Yilmaz GFEVD, Directional TO/FROM/NET spillover metrics, Total Connectedness Index (TCI), Dynamic Rolling TCI, Quantile Connectedness.
* **Experimental**: High-frequency intraday spillover decomposition.
* **Illustrative**: Sectoral stress testing under simulated crisis scenarios.
* **Future Work**: High-dimensional LASSO-VAR connectedness for 50+ individual equities.
