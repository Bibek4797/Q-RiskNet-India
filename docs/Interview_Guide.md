# Q-RiskNet India — Quantitative & Software Engineering Interview Guide

**Target Roles**: Quantitative Researcher, Quantitative Developer, Risk Analyst, Machine Learning Engineer  

---

## 📈 1. Econometrics & Volatility Questions

### Q: What is the difference between ARCH, GARCH, EGARCH, and GJR-GARCH?
- **ARCH(p)**: Models conditional variance as a linear function of $p$ past squared residuals $\varepsilon_{t-i}^2$.
- **GARCH(p,q)**: Adds $q$ lagged conditional variance terms $\sigma_{t-j}^2$, capturing persistence with far fewer parameters.
- **EGARCH(p,o,q)**: Models $\ln(\sigma_t^2)$ using standardized residuals, eliminating non-negativity parameter constraints and capturing exponential asymmetric shock impacts.
- **GJR-GARCH(p,o,q)**: Adds an indicator function $I_{t-1} = 1_{\{\varepsilon_{t-1} < 0\}}$ to model asymmetric leverage effects, where negative return shocks increase conditional volatility more than positive shocks of equal magnitude.

### Q: How do you interpret the persistence parameter in a GARCH(1,1) model?
- Persistence is defined as $P = \alpha_1 + \beta_1$.
- If $P < 1$, the variance process is mean-reverting with long-run variance $\sigma_\infty^2 = \omega / (1 - \alpha_1 - \beta_1)$.
- The shock half-life (in trading days) is $\text{HL} = \frac{\ln(0.5)}{\ln(\alpha_1 + \beta_1)}$. If $P \approx 0.98$, the half-life is $\sim 34$ trading days.

---

## 📊 2. Quantile VAR & Connectedness Questions

### Q: Why is Quantile VAR superior to standard VAR for systemic risk analysis?
- Standard VAR estimates conditional mean relationships $\mathbb{E}[\mathbf{r}_t \mid \mathbf{r}_{t-1}]$, ignoring tail behavior.
- Quantile VAR estimates conditional quantiles $\mathbf{Q}_\tau(\mathbf{r}_t \mid \mathbf{r}_{t-1})$ across extreme bearish ($\tau=0.05$) and bullish ($\tau=0.95$) states.
- Systemic risk spillovers intensify significantly during market panics ($\tau=0.05$), which standard VAR completely obscures.

### Q: How is the Diebold-Yilmaz Total Connectedness Index (TCI) calculated?
$$\text{TCI} = \frac{\sum_{i=1}^K \sum_{j=1, j \neq i}^K \tilde{\theta}_{ij}^H(\tau)}{K} \times 100$$
- It sums all off-diagonal non-self spillover elements in the variance decomposition matrix $\mathbf{S}^H(\tau)$ and divides by the total number of sectors $K$.

---

## 🕸️ 3. Network Science Questions

### Q: How is the correlation distance matrix constructed for Minimum Spanning Trees (MST)?
- Pearson correlation $\rho_{ij} \in [-1, 1]$ is transformed into a metric Euclidean distance:
  $$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$
- $d_{ij} \in [0, 2]$, satisfying non-negativity, symmetry, and triangle inequality.
- Kruskal's algorithm extracts the $K-1$ edge tree spanning all nodes with minimal total distance.

### Q: What does High Betweenness Centrality indicate in a financial spillover network?
- Betweenness centrality measures the fraction of shortest directed paths passing through a node.
- A sector with high betweenness acts as a **critical risk bridge** or conduit through which contagion spreads across clusters.

---

## 🤖 4. Machine Learning & Deep Learning Questions

### Q: Explain the Pinball Loss function used in Quantile LSTM training.
$$\mathcal{L}_\tau(y, \hat{y}) = \max \left( (\tau - 1)(y - \hat{y}), \, \tau(y - \hat{y}) \right)$$
- If the model under-predicts ($y > \hat{y}$), the penalty is proportional to $\tau$.
- If the model over-predicts ($y < \hat{y}$), the penalty is proportional to $1 - \tau$.
- Minimizing expected pinball loss yields the true conditional quantile $\mathbf{Q}_\tau(y|X)$.

### Q: How does the Diebold-Mariano test evaluate out-of-sample forecast superiority?
- It tests the null hypothesis $H_0: \mathbb{E}[d_t] = 0$ where loss differential $d_t = e_{1,t}^2 - e_{2,t}^2$.
- The test statistic $DM = \bar{d} / \sqrt{\hat{V}(\bar{d}) / N} \sim \mathcal{N}(0,1)$ determines if model 2 significantly outperforms benchmark model 1.

---

## 🛠️ 5. Software Architecture & Streamlit Questions

### Q: How is state managed in the multi-page Streamlit architecture?
- `st.session_state` stores persistent objects (`spillover_df`, `metrics`, `pipeline_output`).
- Heavy computation functions are decorated with `@st.cache_data` to ensure zero redundant recalculations during UI reruns.
- Page navigation is handled dynamically via `dashboard/app.py` routing to isolated modules in `dashboard/pages/`.
