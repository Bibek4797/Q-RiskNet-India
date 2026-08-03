import numpy as np
import pandas as pd
import networkx as nx
import src.diagnostics.logger as diag

def build_networkx_graph(spillover_df, threshold_pct=2.0):
    """
    Constructs a NetworkX Directed Graph (DiGraph) from a Diebold-Yilmaz spillover matrix.
    Edge weight represents risk spillover from source column to target row.
    """
    G = nx.DiGraph()
    sectors = list(spillover_df.columns)
    G.add_nodes_from(sectors)

    for source in sectors:
        for target in sectors:
            if source != target:
                weight = float(spillover_df.loc[target, source]) # Spillover from source -> target
                if weight >= threshold_pct:
                    G.add_edge(source, target, weight=weight)

    return G

def compute_network_centrality_metrics(spillover_df, threshold_pct=2.0):
    """
    Computes comprehensive Network Centrality Metrics for all sector nodes in the graph.
    """
    with diag.DiagnosticTimer(f"Network Centrality Calculation (threshold={threshold_pct}%)"):
        G = build_networkx_graph(spillover_df, threshold_pct=threshold_pct)
        sectors = list(spillover_df.columns)

        out_degree = dict(G.out_degree(weight='weight'))
        in_degree = dict(G.in_degree(weight='weight'))

        try:
            betweenness = nx.betweenness_centrality(G, weight='weight')
        except Exception:
            betweenness = {s: 0.0 for s in sectors}

        try:
            closeness = nx.closeness_centrality(G, distance='weight')
        except Exception:
            closeness = {s: 0.0 for s in sectors}

        try:
            pagerank = nx.pagerank(G, weight='weight')
        except Exception:
            pagerank = {s: 1.0 / len(sectors) for s in sectors}

        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=500, weight='weight')
        except Exception:
            eigenvector = pagerank

        metrics_list = []
        for s in sectors:
            out_d = out_degree.get(s, 0.0)
            in_d = in_degree.get(s, 0.0)
            metrics_list.append({
                "Sector": s,
                "Out_Degree_Export": round(out_d, 2),
                "In_Degree_Import": round(in_d, 2),
                "Net_Degree_Export": round(out_d - in_d, 2),
                "Betweenness_Centrality": round(betweenness.get(s, 0.0), 4),
                "Closeness_Centrality": round(closeness.get(s, 0.0), 4),
                "PageRank_Centrality": round(pagerank.get(s, 0.0), 4),
                "Eigenvector_Centrality": round(eigenvector.get(s, 0.0), 4)
            })

        df = pd.DataFrame(metrics_list).sort_values(by="Out_Degree_Export", ascending=False)
        return df

def compute_global_network_stats(spillover_df, threshold_pct=2.0):
    """
    Computes global topological properties of the financial network.
    """
    G = build_networkx_graph(spillover_df, threshold_pct=threshold_pct)
    density = nx.density(G)
    
    try:
        avg_clustering = nx.average_clustering(G, weight='weight')
    except Exception:
        avg_clustering = 0.0

    is_connected = nx.is_weakly_connected(G)
    num_components = nx.number_weakly_connected_components(G)

    return {
        "Node_Count": G.number_of_nodes(),
        "Edge_Count": G.number_of_edges(),
        "Network_Density": round(density, 4),
        "Avg_Clustering_Coefficient": round(avg_clustering, 4),
        "Is_Weakly_Connected": is_connected,
        "Connected_Components_Count": num_components
    }
