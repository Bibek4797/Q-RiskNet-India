import os
import json
import pandas as pd
import numpy as np
import networkx as nx

from src.config.settings import PATHS, ROOT_DIR
import src.diagnostics.logger as diag
from src.network.centrality import compute_network_centrality_metrics, compute_global_network_stats
from src.network.spectral import detect_communities
from src.network.mst import compute_correlation_distance, construct_mst

def run_all_network_analysis(spillover_df, returns_df, threshold_pct=2.0, save_reports=True):
    """
    Master Network Science & Systemic Topology Runner.
    Computes centrality metrics, spectral community clusters, global network stats, and MST edges.
    Saves report files in reports/.
    """
    with diag.DiagnosticTimer("Master Financial Network Analysis Suite"):
        centrality_df = compute_network_centrality_metrics(spillover_df, threshold_pct=threshold_pct)
        global_stats = compute_global_network_stats(spillover_df, threshold_pct=threshold_pct)

        # Spectral Communities
        try:
            comms = detect_communities(spillover_df, n_communities="auto")
            comm_rows = [{"Sector": k, "Community_Cluster": v} for k, v in comms.items()]
            comm_df = pd.DataFrame(comm_rows)
        except Exception as e:
            diag.log_warning(f"Spectral community detection failed: {e}")
            comm_df = pd.DataFrame([{"Sector": c, "Community_Cluster": 0} for c in spillover_df.columns])

        # Minimum Spanning Tree (MST)
        try:
            dist_matrix = compute_correlation_distance(returns_df)
            mst_graph = construct_mst(dist_matrix)
            mst_edges = []
            for u, v, d in mst_graph.edges(data=True):
                mst_edges.append({
                    "Source_Sector": u,
                    "Target_Sector": v,
                    "Distance": round(d.get("weight", 0.0), 4),
                    "Correlation": round(1.0 - 0.5 * (d.get("weight", 0.0) ** 2), 4)
                })
            mst_df = pd.DataFrame(mst_edges)
        except Exception as e:
            diag.log_warning(f"MST construction failed: {e}")
            mst_df = pd.DataFrame()

        reports_dir = os.path.join(ROOT_DIR, PATHS.get("reports_dir", "reports"))
        if save_reports:
            os.makedirs(reports_dir, exist_ok=True)
            centrality_df.to_csv(os.path.join(reports_dir, "network_centrality_metrics.csv"), index=False)
            comm_df.to_csv(os.path.join(reports_dir, "network_community_clusters.csv"), index=False)
            mst_df.to_csv(os.path.join(reports_dir, "mst_edges_summary.csv"), index=False)

            top_hub = str(centrality_df.iloc[0]["Sector"]) if not centrality_df.empty else "N/A"
            top_bridge = str(centrality_df.sort_values(by="Betweenness_Centrality", ascending=False).iloc[0]["Sector"]) if not centrality_df.empty else "N/A"

            summary_json = {
                "total_sectors": len(spillover_df.columns),
                "threshold_pct": threshold_pct,
                "global_topology": global_stats,
                "top_systemic_hub": top_hub,
                "top_bridge_sector": top_bridge,
                "community_clusters_count": int(comm_df["Community_Cluster"].nunique()) if not comm_df.empty else 1
            }
            with open(os.path.join(reports_dir, "network_topology_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary_json, f, indent=4)

            diag.log_info(f"Saved network science reports to {reports_dir}")

        return {
            "centrality_df": centrality_df,
            "global_stats": global_stats,
            "community_df": comm_df,
            "mst_df": mst_df
        }
