import numpy as np
import pandas as pd
import networkx as nx
import src.diagnostics.logger as diag

def compute_correlation_distance(returns_df):
    """
    Computes Mantegna (1999) Euclidean distance metric from Pearson correlations:
        d_ij = sqrt(2 * (1 - rho_ij))
    Properties:
        - Symmetric: d_ij = d_ji
        - 0 <= d_ij <= 2 (0 for perfect correlation rho=1, 2 for perfect anti-correlation rho=-1)
    """
    with diag.DiagnosticTimer("Correlation Distance Matrix calculation"):
        corr = returns_df.corr().fillna(0.0)
        dist = np.sqrt(np.maximum(0, 2.0 * (1.0 - corr)))
        return pd.DataFrame(dist, index=returns_df.columns, columns=returns_df.columns)


def compute_spillover_distance(spillover_df, method="sum"):
    """
    Computes pairwise distance metric from a directed spillover matrix S.

    Since in a directed spillover network S_ij != S_ji (two directional weights):
      - method="sum": Total bilateral spillover W_ij = S_ij + S_ji (aggregate 2-way shock volume)
      - method="max": Peak contagion spillover W_ij = max(S_ij, S_ji) (worst-case one-way vulnerability)

    Distance transformation for Prim's Minimum Spanning Tree:
      Higher spillover implies stronger systemic coupling (closer risk proximity).
      Therefore, distance is defined inversely:
        d_ij = 100.0 / (W_ij + 1e-4)
      so that Prim's Minimum Spanning Tree extracts the strongest systemic risk channels.

    Returns:
        (dist_df, weights_df): Pair of DataFrames containing distance matrix and bilateral weights.
    """
    with diag.DiagnosticTimer(f"Spillover Distance Matrix calculation (method={method})"):
        sectors = list(spillover_df.columns)
        n = len(sectors)
        weights = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    weights[i, j] = 0.0
                else:
                    s_ij = float(spillover_df.iloc[i, j])  # i -> j
                    s_ji = float(spillover_df.iloc[j, i])  # j -> i
                    if method == "max":
                        w = max(s_ij, s_ji)
                    else:  # default "sum"
                        w = s_ij + s_ji
                    weights[i, j] = w

        weights_df = pd.DataFrame(weights, index=sectors, columns=sectors)

        # Invert weights to get distance: higher spillover -> lower distance
        dist = 100.0 / (weights + 1e-4)
        np.fill_diagonal(dist, 0.0)
        dist_df = pd.DataFrame(dist, index=sectors, columns=sectors)

        return dist_df, weights_df


def construct_mst(dist_matrix, algorithm="prim", weights_df=None):
    """
    Constructs Minimum Spanning Tree (MST) using Prim's algorithm.

    Prim's Algorithm Mechanism:
    ---------------------------
    1. Starts at an arbitrary root vertex (sector 0).
    2. Maintains a cut (S, V \\ S) where S is the set of currently spanned sectors.
    3. At each step, uses a priority queue (min-heap) to greedily select the edge (u, v)
       with the minimum distance weight crossing the cut:
           e* = argmin_{u in S, v in V\\S} d(u, v)
    4. Adds v to S and repeats until all N sectors are spanned with exactly N - 1 edges.
    5. Time complexity: O(E log V) in sparse graphs, O(V^2) in dense complete graphs.

    Parameters:
        dist_matrix: pd.DataFrame of pairwise distance metrics (symmetric, zero diagonal).
        algorithm: 'prim' (default) or 'kruskal'.
        weights_df: optional pd.DataFrame of original bilateral weights / spillovers for edge metadata.

    Returns:
        nx.Graph: MST graph with edge attributes 'weight' (distance) and optional 'bilateral_weight'.
    """
    with diag.DiagnosticTimer("Minimum Spanning Tree (MST) extraction via Prim's algorithm"):
        G = nx.Graph()
        sectors = list(dist_matrix.columns)
        n = len(sectors)

        for i in range(n):
            for j in range(i + 1, n):
                u, v = sectors[i], sectors[j]
                d_val = float(dist_matrix.loc[u, v])
                edge_attrs = {"weight": d_val}
                if weights_df is not None and u in weights_df.index and v in weights_df.columns:
                    edge_attrs["bilateral_weight"] = float(weights_df.loc[u, v])
                G.add_edge(u, v, **edge_attrs)

        # Explicitly enforce Prim's algorithm
        mst_graph = nx.minimum_spanning_tree(G, weight="weight", algorithm="prim")
        diag.log_info(
            f"MST generated via Prim's algorithm. Nodes: {mst_graph.number_of_nodes()}, "
            f"Edges: {mst_graph.number_of_edges()}"
        )
        return mst_graph
