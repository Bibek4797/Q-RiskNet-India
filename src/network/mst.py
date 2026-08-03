import numpy as np
import pandas as pd
import networkx as nx
import src.diagnostics.logger as diag

def compute_correlation_distance(returns_df):
    """
    Computes Euclidean distance metric from Pearson correlations: d_ij = sqrt(2 * (1 - rho_ij))
    """
    with diag.DiagnosticTimer("Correlation Distance Matrix calculation"):
        corr = returns_df.corr().fillna(0.0)
        dist = np.sqrt(np.maximum(0, 2.0 * (1.0 - corr)))
        return pd.DataFrame(dist, index=returns_df.columns, columns=returns_df.columns)

def construct_mst(dist_matrix):
    """
    Constructs Minimum Spanning Tree (MST) using Kruskal's algorithm on correlation distance.
    """
    with diag.DiagnosticTimer("Minimum Spanning Tree (MST) extraction"):
        G = nx.Graph()
        sectors = dist_matrix.columns
        for i in range(len(sectors)):
            for j in range(i + 1, len(sectors)):
                u, v = sectors[i], sectors[j]
                weight = dist_matrix.loc[u, v]
                G.add_edge(u, v, weight=weight)
                
        mst_graph = nx.minimum_spanning_tree(G, weight='weight')
        diag.log_info(f"MST generated. Nodes: {mst_graph.number_of_nodes()}, Edges: {mst_graph.number_of_edges()}")
        return mst_graph
