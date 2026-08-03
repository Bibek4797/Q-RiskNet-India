# Phase 7 Report: Financial Network Science & Systemic Topology

**Project Name**: Q-RiskNet India  
**Phase Completed**: Phase 7  
**Status**: 100% Verified & Pushed  
**Date**: August 2026  

---

## 1. Executive Summary

Phase 7 constructed an enterprise-grade **Financial Network Science & Systemic Topology Engine** for the Q-RiskNet India platform.

The framework converts Diebold-Yilmaz spillover matrices into **Weighted Directed Graphs** $G = (V, E, W)$. It evaluates sectoral systemic importance using full network centrality metrics (Out-Degree, In-Degree, Betweenness, Closeness, PageRank, Eigenvector Centrality), partitions sector nodes into natural risk clusters via **Spectral Community Detection**, and constructs non-redundant risk backbones via **Minimum Spanning Trees (MST)**. Zero machine learning forecasting or deep learning models were added in this phase, maintaining strict modularity.

---

## 2. Graph Construction & Centrality Formulations

| Network Metric | Equation / Definition | Financial Interpretation |
| :--- | :--- | :--- |
| **Adjacency Weight** | $w_{j \to i} = \tilde{\theta}_{ij} \ge \tau_{\text{edge}}$ | Normalized risk spillover exported from sector $j$ to sector $i$ |
| **Out-Degree Export** | $C_D^{\text{out}}(i) = \sum_{j \neq i} \tilde{\theta}_{ji}$ | Gross risk exported to the network (Systemic Risk Hub) |
| **In-Degree Import** | $C_D^{\text{in}}(i) = \sum_{j \neq i} \tilde{\theta}_{ij}$ | Gross risk imported from the network (Vulnerable Risk Receiver) |
| **Betweenness Centrality** | $C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$ | Frequency of acting as a bridge for shock transmission |
| **PageRank Centrality** | $PR(i) = \frac{1-d}{K} + d \sum_{j \in M(i)} \frac{PR(j)}{L(j)}$ | Systemic centrality based on connection to influential nodes |
| **Network Density** | $D = \frac{\|E\|}{K(K-1)}$ | Overall interconnectedness / tightness of the network ($0 \le D \le 1$) |

---

## 3. Systemic Topology Insights

1. **Systemic Risk Hubs**:
   * **Nifty Bank & Nifty Financial Services**: Rank highest in Out-Degree Export and PageRank Centrality, confirming their role as primary systemic risk drivers in the Indian equity market.
2. **Bridge Sectors**:
   * **Nifty IT & Nifty Energy**: Rank highest in Betweenness Centrality, acting as structural bridges connecting financial shocks to industrial sectors.
3. **Spectral Community Clusters**:
   * Eigengap heuristic partitions Indian sectors into 2-3 distinct clusters (e.g. Financials/Capital Goods Cluster vs Defensive FMCG/Pharma Cluster vs Export IT/Energy Cluster).

---

## 4. Reports Generated in `reports/`

The network runner (`src/network/network_runner.py`) automatically exports 4 structured summary reports to `reports/`:
1. `reports/network_centrality_metrics.csv`
2. `reports/network_community_clusters.csv`
3. `reports/network_topology_summary.json`
4. `reports/mst_edges_summary.csv`

---

## 5. Implementation Status Taxonomy

* **Implemented**: Weighted Directed Graph Construction, Out/In-Degree Centrality, Betweenness Centrality, Closeness Centrality, PageRank, Eigenvector Centrality, Minimum Spanning Tree (MST), Spectral Community Detection, Global Network Density.
* **Experimental**: Intraday dynamic layout animation.
* **Illustrative**: Sectoral node deletion cascade simulation.
* **Future Work**: Deep Graph Neural Networks (GNNs) for predictive contagion modeling.

---

## 6. Verification & Test Results

* **Pytest Test Suite**: Executed `pytest` locally across `tests/test_data.py`, `tests/test_diagnostics.py`, `tests/test_models.py`, `tests/test_network.py`, `tests/test_pipeline.py`, `tests/test_volatility.py`, `tests/test_qvar.py`, `tests/test_connectedness.py`, and `tests/test_network_science.py`.
* **Results**: **24/24 unit tests passed 100%**.
* **Local Dashboard Test**: Streamlit application running on `http://localhost:8501` with zero import errors or exceptions.

---

## 7. GitHub Synchronization Status

* **Branch**: `main`
* **Repository**: `https://github.com/Bibek4797/Q-RiskNet-India.git`
* **Status**: Committed and pushed live.
