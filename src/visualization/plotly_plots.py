"""
Q-RiskNet India — Plotly Network & Heatmap Visualization Utilities
Copyright (c) 2026 Bibek Rout
"""
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
        arrow_annotations = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = d['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=0.8 + (w / 10.0), color='rgba(148, 163, 184, 0.35)'),
                hoverinfo='text',
                text=f"Transmitter: {u}<br>Receiver: {v}<br>Spillover: {w:.2f}%",
                mode='lines'
            )
            edge_traces.append(edge_trace)

            dx = x1 - x0
            dy = y1 - y0
            dist = np.hypot(dx, dy)
            if dist > 0:
                xt = x0 + 0.82 * dx
                yt = y0 + 0.82 * dy
                arrow_annotations.append(dict(
                    ax=x0, ay=y0,
                    x=xt, y=yt,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=max(1.0, min(3.2, 0.8 + (w / 10.0))),
                    arrowcolor='rgba(199, 210, 254, 0.65)',
                    opacity=0.75
                ))
            
        node_x, node_y, node_colors, node_text = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            if communities is not None and node in communities:
                comm_idx = communities[node]
                color = COMMUNITY_COLORS[comm_idx % len(COMMUNITY_COLORS)]
                node_colors.append(color)
                node_text.append(f"Sector: <b>{node}</b>")
            else:
                node_colors.append("#6366f1")
                node_text.append(f"Sector: <b>{node}</b>")
            
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
            title=dict(text="Network Connectedness Graph (Arrows point from Transmitter → Receiver)", font=dict(size=14, color='#94a3b8'), x=0.0, xanchor='left'),
            showlegend=False,
            hovermode='closest',
            annotations=arrow_annotations,
            margin=dict(b=20, l=20, r=20, t=44),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True, fixedrange=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True, fixedrange=False),
            template="plotly_dark",
            height=600,
            uirevision=f"net_graph_{layout_type}_{min_threshold_pct}"
        )
        return fig


def render_mst_network(mst_graph, dist_matrix, title="Minimum Spanning Tree (MST) Risk Backbone (Prim's Algorithm)"):
    """
    Renders Plotly layout for Minimum Spanning Tree (MST) backbone generated via Prim's algorithm.
    """
    with diag.DiagnosticTimer("Plotly MST network drawing"):
        pos = nx.spring_layout(mst_graph, seed=42)
        edge_x, edge_y = [], []
        edge_mid_x, edge_mid_y, edge_hover_text = [], [], []
        
        for u, v, d in mst_graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            dist_val = d.get('weight', 0.0)
            bw = d.get('bilateral_weight', None)
            
            edge_mid_x.append((x0 + x1) / 2.0)
            edge_mid_y.append((y0 + y1) / 2.0)
            if bw is not None:
                edge_hover_text.append(f"Risk Channel: <b>{u} — {v}</b><br>Bilateral Spillover: <b>{bw:.2f}%</b><br>Prim Distance: {dist_val:.3f}")
            else:
                edge_hover_text.append(f"Correlation Channel: <b>{u} — {v}</b><br>Mantegna Distance: <b>{dist_val:.4f}</b>")
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2.5, color='#38bdf8'),
            mode='lines',
            hoverinfo='none'
        )

        edge_hover_trace = go.Scatter(
            x=edge_mid_x, y=edge_mid_y,
            mode='markers',
            marker=dict(size=14, color='#38bdf8', opacity=0.01),
            hoverinfo='text',
            hovertext=edge_hover_text
        )
        
        node_x, node_y, node_names, node_hovers = [], [], [], []
        degrees = dict(mst_graph.degree())
        for node in mst_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_names.append(node)
            deg = degrees.get(node, 0)
            node_hovers.append(f"Sector: <b>{node}</b><br>MST Degree (Connections): <b>{deg}</b>")
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_names,
            textposition="top center",
            marker=dict(
                size=[min(38, 20 + 4 * degrees.get(n, 1)) for n in mst_graph.nodes()],
                color='#f43f5e',
                line=dict(width=2, color='#ffffff')
            ),
            hoverinfo='text',
            hovertext=node_hovers
        )
        
        fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace])
        fig.update_layout(
            title=dict(text=title, font=dict(size=14, color='#94a3b8'), x=0.0, xanchor='left'),
            showlegend=False,
            margin=dict(b=20, l=20, r=20, t=44),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True, fixedrange=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True, fixedrange=False),
            template="plotly_dark",
            height=550,
            uirevision="mst_graph"
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
    fig.update_layout(
        template="plotly_dark", height=500,
        title=dict(font=dict(size=14, color='#94a3b8'), x=0.0, xanchor='left'),
        xaxis=dict(autorange=True, fixedrange=False),
        yaxis=dict(autorange=True, fixedrange=False),
        uirevision="corr_heatmap"
    )
    return fig
