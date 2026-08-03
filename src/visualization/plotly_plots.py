import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

import src.diagnostics.logger as diag

COMMUNITY_COLORS = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", 
    "#8b5cf6", "#ec4899", "#14b8a6", "#6366f1"
]

def render_spillover_network(spillover_df, communities=None, min_threshold_pct=2.0, layout_type="circular"):
    """
    Renders an interactive 2D Directed Plotly Network graph for financial spillover transmission.
    """
    sectors = list(spillover_df.columns)
    K = len(sectors)
    
    with diag.DiagnosticTimer(f"Generating Plotly Network graph (layout={layout_type}, min_edge={min_threshold_pct}%)"):
        G = nx.DiGraph()
        for s in sectors:
            G.add_node(s)
            
        edge_count = 0
        for i, source in enumerate(sectors):
            for j, target in enumerate(sectors):
                if i != j:
                    weight = spillover_df.loc[source, target]
                    if weight >= min_threshold_pct:
                        G.add_edge(source, target, weight=weight)
                        edge_count += 1
                        
        diag.log_info(f"Rendering complete. Edges rendered above threshold: {edge_count}")
        
        if layout_type == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42, k=1.5/np.sqrt(K))
            
        edge_traces = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = d['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=0.8 + (w / 10.0), color='rgba(148, 163, 184, 0.5)'),
                hoverinfo='text',
                text=f"Transmitter: {u}<br>Receiver: {v}<br>Spillover: {w:.2f}%",
                mode='lines'
            )
            edge_traces.append(edge_trace)
            
        node_x, node_y, node_colors, node_text = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            comm_idx = communities[node] if (communities is not None and node in communities) else 0
            color = COMMUNITY_COLORS[comm_idx % len(COMMUNITY_COLORS)]
            node_colors.append(color)
            node_text.append(f"Sector: <b>{node}</b><br>Cluster: Community {comm_idx + 1}")
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=sectors,
            textposition="top center",
            marker=dict(
                color=node_colors,
                size=28,
                line=dict(width=2, color='#ffffff')
            ),
            hovertext=node_text
        )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=dict(text="Network Connectedness Graph", font=dict(size=18)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_dark",
            height=600
        )
        return fig

def render_mst_network(mst_graph, dist_matrix):
    """
    Renders Plotly layout for Minimum Spanning Tree (MST) backbone.
    """
    with diag.DiagnosticTimer("Plotly MST network drawing"):
        pos = nx.spring_layout(mst_graph, seed=42)
        edge_x, edge_y, edge_text = [], [], []
        
        for u, v, d in mst_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            dist_val = d['weight']
            edge_text.append(f"Distance ({u} - {v}): {dist_val:.4f}")
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#38bdf8'),
            mode='lines',
            hoverinfo='none'
        )
        
        node_x, node_y, node_names = [], [], []
        for node in mst_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_names.append(node)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_names,
            textposition="top center",
            marker=dict(
                size=22,
                color='#f43f5e',
                line=dict(width=2, color='#ffffff')
            ),
            hoverinfo='text'
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=dict(text="Minimum Spanning Tree (MST) Risk Backbone", font=dict(size=18)),
            showlegend=False,
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_dark",
            height=550
        )
        return fig

def render_correlation_heatmap(corr_df):
    """
    Renders Pearson correlation matrix heatmap.
    """
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Sectoral Pearson Correlation Matrix"
    )
    fig.update_layout(template="plotly_dark", height=500)
    return fig
