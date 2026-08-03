# Volatility Modelling Methodology & Leverage Effect Analysis

**Document Version**: 1.0.0  
**Project**: Q-RiskNet India  
**Date**: August 2026  

---

## 1. Executive Summary

Financial return series exhibit **volatility clustering**, where large price swings are followed by large price swings, and quiet periods are followed by quiet periods. Traditional OLS models assuming constant variance ($\sigma^2 = \text{const}$) fail to capture this phenomenon.

**Phase 4** implements an enterprise-grade Autoregressive Conditional Heteroskedasticity (GARCH) model family to capture, forecast, and compare conditional volatility dynamics across Indian stock market sectors.

---

## 2. GARCH Family Model Formulations

### 2.1 ARCH(1) Model (Engle, 1982)
The conditional variance depends on the previous period's squared return shock:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2$$
* $\omega > 0$: Baseline variance constant.
* $\alpha \ge 0$: Reaction parameter (sensitivity to immediate news shocks).

### 2.2 GARCH(1,1) Model (Bollerslev, 1986)
Extends ARCH by incorporating lagged conditional variance (volatility memory):
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
* $\beta \ge 0$: Persistence parameter (memory of past volatility).
* **Stationarity Condition**: $\alpha + \beta < 1$.
* **Volatility Persistence**: $P = \alpha + \beta$.
* **Half-Life of Volatility Shocks**: $HL = \frac{\ln(0.5)}{\ln(\alpha + \beta)}$ days.
* **Unconditional Long-Run Variance**: $\sigma_{LR}^2 = \frac{\omega}{1 - (\alpha + \beta)}$.

### 2.3 GJR-GARCH(1,1,1) Model (Glosten, Jagannathan, & Runkle, 1993)
Captures asymmetric response to news (the **leverage effect**), where bad news ($\epsilon_{t-1} < 0$) increases volatility more than good news ($\epsilon_{t-1} > 0$):
$$\sigma_t^2 = \omega + \left( \alpha + \gamma I_{t-1} \right) \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
where $I_{t-1} = 1$ if $\epsilon_{t-1} < 0$, and $0$ otherwise.
* $\gamma > 0$: Asymmetry / leverage parameter.
* **Effective Persistence**: $P = \alpha + \beta + \frac{\gamma}{2}$.

### 2.4 EGARCH(1,1,1) Model (Nelson, 1991)
Exponential GARCH models the log of conditional variance, ensuring positivity without parameter constraints:
$$\ln(\sigma_t^2) = \omega + \alpha \left( \left| \frac{\epsilon_{t-1}}{\sigma_{t-1}} \right| - \sqrt{\frac{2}{\pi}} \right) + \gamma \frac{\epsilon_{t-1}}{\sigma_{t-1}} + \beta \ln(\sigma_{t-1}^2)$$
* $\gamma < 0$: Negative shocks increase volatility more than positive shocks of equal magnitude.
* **Persistence**: $P = \beta$.

---

## 3. Model Comparison Metrics

| Metric | Mathematical Definition | Interpretation |
| :--- | :--- | :--- |
| **Log-Likelihood** | $\mathcal{L}(\theta) = -\frac{1}{2} \sum \left( \ln(2\pi) + \ln(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2} \right)$ | Higher value indicates better fit |
| **AIC** | $-2\mathcal{L} + 2k$ | Lower value indicates optimal parsimonious fit |
| **BIC** | $-2\mathcal{L} + k \ln(N)$ | Penalizes model complexity more heavily than AIC |
| **ARCH-LM p-val** | $p$-value from Engle's ARCH-LM test on $\hat{e}_t = \epsilon_t / \hat{\sigma}_t$ | $p > 0.05$ indicates all ARCH effects removed |

---

## 4. Economic Interpretation for Indian Stock Sectors

1. **Leverage Effect in Emerging Equity Markets**: Indian sector indices (Nifty Bank, Nifty IT, Nifty 50) display significant positive $\gamma$ in GJR-GARCH ($\gamma > 0$) and negative $\gamma$ in EGARCH ($\gamma < 0$). This confirms that negative market returns trigger higher volatility spikes due to financial leverage and risk aversion.
2. **Volatility Half-Life**: Financial sectors exhibit persistence $P \approx 0.95-0.98$, resulting in a volatility half-life of $13.5 - 34.3$ trading days.
