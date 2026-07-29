import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import SpectralClustering

# Import diagnostics
import diagnostics as diag

def compute_correlation_distance(corr_matrix):
    """
    Computes the distance matrix from a correlation matrix: d_ij = sqrt(2 * (1 - rho_ij))
    """
    with diag.DiagnosticTimer("Correlation Distance Matrix calculation"):
        # Clip correlations to [-1, 1] to avoid negative numbers under sqrt due to float precision
        clipped_corr = np.clip(corr_matrix.values, -1.0, 1.0)
        dist_matrix = np.sqrt(2 * (1 - clipped_corr))
        return pd.DataFrame(dist_matrix, index=corr_matrix.index, columns=corr_matrix.columns)

def build_mst_graph(dist_matrix):
    """
    Builds the Minimum Spanning Tree (MST) from a distance matrix.
    
    Returns:
        nx.Graph: NetworkX graph representing the MST.
    """
    with diag.DiagnosticTimer("Minimum Spanning Tree (MST) extraction"):
        G = nx.Graph()
        sectors = dist_matrix.columns
        
        # Add nodes
        for sector in sectors:
            G.add_node(sector)
            
        # Add weighted edges
        for i in range(len(sectors)):
            for j in range(i + 1, len(sectors)):
                G.add_edge(sectors[i], sectors[j], weight=dist_matrix.iloc[i, j])
                
        # Calculate MST using Kruskal's algorithm
        mst = nx.minimum_spanning_tree(G, weight='weight')
        diag.log_info(f"MST generated. Nodes: {len(mst.nodes)}, Edges: {len(mst.edges)}")
        return mst

def find_optimal_eigengap_k(similarity_matrix):
    """
    Determines the optimal number of clusters k using the Eigengap Heuristic.
    k_opt = argmax_k (lambda_{k+1} - lambda_k)
    """
    K = similarity_matrix.shape[0]
    if K <= 2:
        return 1
        
    # Degree matrix D
    deg_vector = np.sum(similarity_matrix, axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(deg_vector, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Normalized Laplacian L_sym = I - D^(-1/2) S D^(-1/2)
    L_sym = np.eye(K) - np.dot(np.dot(D_inv_sqrt, similarity_matrix), D_inv_sqrt)
    
    # Compute sorted eigenvalues
    eigenvalues = np.sort(np.linalg.eigvalsh(L_sym))
    
    # Calculate gaps between consecutive eigenvalues
    max_gap = -1.0
    optimal_k = 2
    
    for k in range(1, K - 1):
        gap = eigenvalues[k] - eigenvalues[k - 1]
        if gap > max_gap:
            max_gap = gap
            optimal_k = k
            
    return max(2, min(optimal_k, K - 1))

def detect_communities(spillover_matrix, n_communities="auto"):
    """
    Uses Spectral Clustering to partition the spillover network into communities.
    Supports manual n_communities or automatic optimal selection via the Eigengap Heuristic.
    """
    sectors = spillover_matrix.columns
    K = len(sectors)
    
    # Symmetrical similarity matrix: S_ij = (spillover(i->j) + spillover(j->i)) / 2
    similarity = 0.5 * (spillover_matrix.values + spillover_matrix.values.T)
    # Ensure diagonal is zero for community detection
    np.fill_diagonal(similarity, 0.0)
    
    # Normalize similarity to [0, 1] scale for robust clustering
    max_val = np.max(similarity)
    if max_val > 0:
        similarity = similarity / max_val
        
    # Determine target communities
    if K < 3:
        diag.log_info(f"K={K} sectors selected. Assigning all sectors to Community 1.")
        return pd.Series(0, index=sectors)

    if n_communities == "auto" or n_communities is None:
        target_communities = find_optimal_eigengap_k(similarity)
        diag.log_info(f"Eigengap Heuristic auto-detected optimal communities k={target_communities}")
    else:
        target_communities = int(n_communities)
        if target_communities >= K:
            target_communities = max(2, K - 1)
            diag.log_warning(f"Target communities ({n_communities}) >= K ({K}). Clamped to: {target_communities}")
        
    with diag.DiagnosticTimer(f"Spectral Clustering Community Detection (target={target_communities})"):
        try:
            sc = SpectralClustering(
                n_clusters=target_communities, 
                affinity='precomputed', 
                assign_labels='kmeans', 
                random_state=42
            )
            labels = sc.fit_predict(similarity)
            diag.log_info(f"Spectral Clustering finished. Output clusters: {set(labels)}")
            return pd.Series(labels, index=sectors)
        except Exception as e:
            # Fallback if spectral clustering fails
            diag.log_error("Spectral clustering failed. Falling back to zero-labels.", e)
            return pd.Series(0, index=sectors)

def draw_plotly_network(spillover_df, metrics, communities=None, layout_type="circular", min_edge_threshold=2.0):
    """
    Generates a beautiful interactive Plotly figure representing the spillover network.
    
    Args:
        spillover_df: K x K matrix of pairwise spillovers.
        metrics: Dictionary containing 'TO', 'FROM', 'NET' series.
        communities: Series containing community labels.
        layout_type: "circular" or "spring".
        min_edge_threshold: Minimum spillover % required to draw an edge.
    """
    sectors = list(spillover_df.columns)
    
    with diag.DiagnosticTimer(f"Generating Plotly Network graph (layout={layout_type}, min_edge={min_edge_threshold}%)"):
        # Create networkx directed graph
        G = nx.DiGraph()
        for s in sectors:
            G.add_node(s)
            
        # Add directed edges where spillover is above threshold
        for i, rec_sec in enumerate(sectors):
            for j, trans_sec in enumerate(sectors):
                if i != j:
                    weight = spillover_df.iloc[i, j]
                    if weight >= min_edge_threshold:
                        G.add_edge(trans_sec, rec_sec, weight=weight)
                        
        # Generate node positions
        if layout_type == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42, k=1.5)
            
        # Prepare node traces
        node_x = []
        node_y = []
        for sector in sectors:
            x, y = pos[sector]
            node_x.append(x)
            node_y.append(y)
            
        # Aligned colors & sizes directly indexed by sector name (prevents index misalignment bugs)
        node_colors = [metrics["NET"][sector] for sector in sectors]
        
        # Color scale limits
        max_net = max(max(abs(x) for x in node_colors), 1.0)
        
        # Prepare hover text
        node_hover_text = []
        for sector in sectors:
            comm_val = communities[sector] + 1 if communities is not None else 1
            c_label = f" | Community {comm_val}" if communities is not None else ""
            hover = (
                f"<b>{sector}</b>{c_label}<br>"
                f"Transmitted (TO): {metrics['TO'][sector]:.2f}%<br>"
                f"Received (FROM): {metrics['FROM'][sector]:.2f}%<br>"
                f"<b>Net Connectedness: {metrics['NET'][sector]:.2f}%</b>"
            )
            node_hover_text.append(hover)
            
        # Size nodes relative to their absolute Net Connectedness
        node_sizes = [20 + 20 * (abs(metrics["NET"][sector]) / max_net) for sector in sectors]
        
        # 1. Edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            u, v, data = edge
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = data['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1.0 + 3.0 * (weight / 50.0), color='rgba(100, 100, 100, 0.4)'),
                hoverinfo='text',
                text=f"{u} → {v}: {weight:.2f}%",
                mode='lines'
            )
            edge_traces.append(edge_trace)
            
        # 2. Node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=sectors,
            textposition="top center",
            hoverinfo='text',
            hovertext=node_hover_text,
            marker=dict(
                showscale=True,
                colorscale='RdBu_r', # Red = Transmitter, Blue = Receiver
                cmin=-max_net,
                cmax=max_net,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    title=dict(
                        text="Net Volatility Spillover (%)",
                        side="right"
                    ),
                    thickness=15,
                    x=1.1
                ),
                line_width=2,
                line_color='white'
            ),
            textfont=dict(
                family="Outfit, Inter, sans-serif",
                size=11,
                color="white"
            )
        )
        
        # Build complete figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=dict(
                    text="Interactive Risk Connectedness Network",
                    font=dict(size=16, color="white"),
                    x=0.05
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=50),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(15, 23, 42, 1)',
                plot_bgcolor='rgba(15, 23, 42, 1)'
            )
        )
        
        diag.log_info(f"Rendering complete. Edges rendered above threshold: {len(edge_traces)}")
        return fig

def draw_mst_network(mst_graph, correlation_df, layout_type="spring"):
    """
    Generates a Plotly figure representing the Minimum Spanning Tree.
    """
    with diag.DiagnosticTimer("Plotly MST network drawing"):
        pos = nx.spring_layout(mst_graph, seed=42) if layout_type == "spring" else nx.circular_layout(mst_graph)
        
        edge_x = []
        edge_y = []
        edge_hover = []
        
        for u, v, d in mst_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            corr_val = correlation_df.loc[u, v]
            edge_hover.append(f"{u} — {v} (Distance: {d['weight']:.4f}, Corr: {corr_val:.4f})")
            
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=2, color='rgba(99, 102, 241, 0.8)'),
            mode='lines',
            hoverinfo='none'
        )
        
        node_x = []
        node_y = []
        node_text = []
        
        for node in mst_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                size=14,
                color='#6366F1',
                line=dict(width=1.5, color='white')
            ),
            textfont=dict(
                family="Outfit, Inter, sans-serif",
                size=11,
                color="white"
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="Minimum Spanning Tree (MST) Risk Backbone",
                    font=dict(size=16, color="white"),
                    x=0.05
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=50),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='rgba(15, 23, 42, 1)',
                plot_bgcolor='rgba(15, 23, 42, 1)'
            )
        )
        
        return fig
