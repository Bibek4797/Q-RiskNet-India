# Quantile Vector Autoregression (QVAR) Methodology & Tail Dependence Framework

**Document Version**: 1.0.0  
**Project**: Q-RiskNet India  
**Date**: August 2026  

---

## 1. Executive Summary

Classical Vector Autoregression (VAR) estimated via Ordinary Least Squares (OLS) measures relationships exclusively at the conditional mean ($\mathbb{E}[y_t | y_{t-1}]$). In financial markets, conditional mean models assume that cross-sector return dependencies are constant regardless of market state.

However, empirical financial returns exhibit **asymmetric tail dependence**: cross-sector shock transmission intensifies dramatically during market panics ($\tau = 0.05$) compared to normal trading conditions ($\tau = 0.50$). 

**Quantile VAR (QVAR)** generalizes classical VAR by estimating vector autoregressive parameters at arbitrary conditional quantiles $\tau \in (0, 1)$, capturing non-linear dynamic interactions across bearish ($\tau \le 0.10$), normal ($\tau = 0.50$), and bullish ($\tau \ge 0.90$) market regimes.

---

## 2. Mathematical Formulation of QVAR

Consider a $K$-dimensional vector of sectoral returns $y_t = (y_{1,t}, y_{2,t}, \dots, y_{K,t})^\top$. The $p$-th order Quantile VAR model at quantile $\tau \in (0, 1)$ is formulated as:

$$y_t = \mu(\tau) + \sum_{j=1}^{p} \Phi_j(\tau) y_{t-j} + u_t(\tau)$$

where:
* $\mu(\tau)$ is a $K \times 1$ quantile-dependent intercept vector.
* $\Phi_j(\tau)$ is a $K \times K$ autoregressive coefficient matrix at lag $j$ for quantile $\tau$.
* $u_t(\tau)$ is a $K \times 1$ vector of error terms satisfying the conditional quantile restriction:
  $$\text{Quant}_\tau \left( u_t(\tau) \mid y_{t-1}, \dots, y_{t-p} \right) = 0$$

### 2.1 Optimization via Pinball Loss
Parameters for equation $i$ ($i=1, \dots, K$) are estimated independently by minimizing the asymmetric quantile loss function:

$$\min_{\mu_i(\tau), \Phi_i(\tau)} \sum_{t=p+1}^{T} \rho_\tau \left( y_{i,t} - \mu_i(\tau) - \sum_{j=1}^{p} \mathbf{\Phi}_{i,j}(\tau) y_{t-j} \right)$$

where the pinball loss check function $\rho_\tau(u)$ is defined as:

$$\rho_\tau(u) = u \cdot \left( \tau - \mathbf{I}(u < 0) \right) = \begin{cases} \tau u & \text{if } u \ge 0 \\ (\tau - 1) u & \text{if } u < 0 \end{cases}$$

---

## 3. Quantile Regimes & Financial Interpretation

| Quantile ($\tau$) | Market Regime | Financial & Economic Interpretation |
| :--- | :--- | :--- |
| **$\tau = 0.05$** | **Extreme Bearish (Crash)** | Evaluates systemic contagion & panic spillover during tail-risk events. |
| **$\tau = 0.10$** | **Bearish (Downturn)** | Measures transmission dynamics during market stress and elevated volatility. |
| **$\tau = 0.50$** | **Median (Normal)** | Baseline central tendency relationship (equivalent to OLS VAR baseline). |
| **$\tau = 0.90$** | **Bullish (Rally)** | Captures cross-sector euphoria transmission during strong market expansions. |
| **$\tau = 0.95$** | **Extreme Bullish (Boom)** | Evaluates extreme upside co-movement during market surges. |

---

## 4. Generalized Impulse Response Functions (GIRF)

To trace shock propagation without relying on arbitrary Cholesky ordering, the **Generalized Impulse Response Function (GIRF)** (Pesaran & Shin, 1998) simulates the response of variable $i$ at horizon $h$ to a $+2\sigma$ shock in variable $j$ at quantile $\tau$:

$$\text{GIRF}_{i,j}(h, \tau) = \mathbb{E} \left[ y_{i,t+h} \mid u_{j,t} = \delta_j, \, \mathfrak{F}_{t-1} \right] - \mathbb{E} \left[ y_{i,t+h} \mid \mathfrak{F}_{t-1} \right]$$

where $\delta_j = 2 \cdot \hat{\sigma}_j$ is a two-standard-deviation shock to sector $j$.

---

## 5. Research Contribution & Advantages

1. **Non-Linear Dynamics**: Uncovers structural asymmetries invisible to OLS regression.
2. **Robustness to Outliers**: Quantile estimation is inherently robust to extreme fat-tailed outliers.
3. **Regime-Specific Sensitivity**: Enables portfolio managers to stress-test sector allocations under tail-risk regimes ($\tau = 0.05$).
