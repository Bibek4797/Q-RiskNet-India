# Financial Network Science & Systemic Topology Methodology

**Document Version**: 1.0.0  
**Project**: Q-RiskNet India  
**Date**: August 2026  

---

## 1. Executive Summary

Financial markets operate as complex adaptive networks. Sectoral indices do not trade in isolation; shocks transmit through direct and indirect interlinkages driven by shared macroeconomic drivers, institutional cross-holdings, and credit channels.

**Phase 7** maps the Diebold-Yilmaz spillover matrices into **Weighted Directed Graphs** $G = (V, E, W)$, enabling network-theoretic analysis of systemic importance, sector centrality, vulnerability, and community segmentation across the Indian stock market.

---

## 2. Graph Construction & Formulations

A financial spillover network is represented as a directed graph $G = (V, E, W)$:
* $V = \{v_1, v_2, \dots, v_K\}$: Set of $K$ sectoral index nodes.
* $E \subseteq V \times V$: Set of directed edges where edge $(v_j \to v_i)$ represents risk spillover from sector $j$ to sector $i$.
* $W = [\tilde{\theta}_{ij}]$: Weighted adjacency matrix derived from Normalized Forecast Error Variance Decomposition (GFEVD), where edge weight $w_{j \to i} = \tilde{\theta}_{ij}$.

### 2.1 Edge Thresholding
To eliminate spurious noise and minor spillovers, edges are filtered using a threshold $\tau_{\text{edge}} \ge 2.0\%$:
$$E_{\text{filtered}} = \{ (v_j, v_i) \mid \tilde{\theta}_{ij} \ge \tau_{\text{edge}}, \, i \neq j \}$$

---

## 3. Network Centrality Metrics

| Centrality Metric | Mathematical Formulation | Financial Interpretation |
| :--- | :--- | :--- |
| **Out-Degree Centrality** | $C_D^{\text{out}}(i) = \sum_{j \neq i} \tilde{\theta}_{ji}$ | Total gross risk exported to the network (Systemic Risk Hub) |
| **In-Degree Centrality** | $C_D^{\text{in}}(i) = \sum_{j \neq i} \tilde{\theta}_{ij}$ | Total gross risk imported from the network (Vulnerable Receiver) |
| **Betweenness Centrality** | $C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$ | Measures how often sector $i$ acts as a bridge for shock transmission |
| **PageRank Centrality** | $PR(i) = \frac{1-d}{K} + d \sum_{j \in M(i)} \frac{PR(j)}{L(j)}$ | Identifies sectors connected to other highly central risk exporters |
| **Eigenvector Centrality** | $\lambda v_i = \sum_{j} A_{ij} v_j$ | Measures systemic importance based on connection to influential nodes |

---

## 4. Topology & Community Structure Analysis

### 4.1 Minimum Spanning Tree (MST) Risk Backbone
To isolate the ultra-clean backbone of non-redundant dependencies, Pearson correlations $\rho_{ij}$ are converted to metric distances:
$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$
Kruskal's algorithm computes the **Minimum Spanning Tree (MST)** spanning all $K$ nodes with minimal total distance $T = \sum e_{ij}$.

### 4.2 Spectral Community Detection
Partitions $K$ sectors into $N_c$ disjoint communities by computing the top $k$ eigenvectors of the normalized graph Laplacian matrix:
$$L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$$
The optimal number of communities $N_c$ is determined endogenously using the **Eigengap Heuristic** ($\max_k (\lambda_{k+1} - \lambda_k)$).

---

## 5. Implementation Status Taxonomy

* **Implemented**: Weighted Directed Network Construction, Out/In-Degree Centrality, Betweenness Centrality, PageRank, Eigenvector Centrality, Minimum Spanning Tree (MST), Spectral Community Detection, Dynamic Network Density tracking.
* **Experimental**: Intraday dynamic network layout animations.
* **Illustrative**: Sectoral node removal / cascade failure simulation.
* **Future Work**: Multilayer multiplex networks combining equity, debt, and derivative interlinkages.
