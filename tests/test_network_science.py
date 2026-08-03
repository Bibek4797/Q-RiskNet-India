import pytest
import pandas as pd
import numpy as np

from src.network.centrality import build_networkx_graph, compute_network_centrality_metrics, compute_global_network_stats
from src.network.spectral import detect_communities
from src.network.mst import compute_correlation_distance, construct_mst
from src.network.network_runner import run_all_network_analysis

def test_graph_construction_and_centrality():
    np.random.seed(42)
    sectors = ["Bank", "IT", "Energy", "Auto"]
    spill_data = np.array([
        [70.0, 10.0, 15.0, 5.0],
        [12.0, 65.0, 18.0, 5.0],
        [8.0, 12.0, 75.0, 5.0],
        [10.0, 13.0, 12.0, 65.0]
    ])
    spill_df = pd.DataFrame(spill_data, index=sectors, columns=sectors)

    G = build_networkx_graph(spill_df, threshold_pct=5.0)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() > 0

    cent_df = compute_network_centrality_metrics(spill_df, threshold_pct=5.0)
    assert cent_df.shape[0] == 4
    assert "Out_Degree_Export" in cent_df.columns
    assert "Betweenness_Centrality" in cent_df.columns

    stats = compute_global_network_stats(spill_df, threshold_pct=5.0)
    assert stats["Node_Count"] == 4
    assert 0.0 <= stats["Network_Density"] <= 1.0

def test_spectral_communities():
    sectors = ["Bank", "IT", "Energy"]
    spill_df = pd.DataFrame([
        [80.0, 10.0, 10.0],
        [15.0, 75.0, 10.0],
        [5.0, 15.0, 80.0]
    ], index=sectors, columns=sectors)

    comms = detect_communities(spill_df, n_communities=2)
    assert len(comms) == 3

def test_network_runner():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=50)
    returns = pd.DataFrame({
        "Bank": np.random.normal(0, 1, 50),
        "IT": np.random.normal(0, 1, 50),
        "Energy": np.random.normal(0, 1, 50)
    }, index=dates)
    spill_df = returns.corr().abs() * 100

    res = run_all_network_analysis(spill_df, returns, threshold_pct=10.0, save_reports=True)
    assert "centrality_df" in res
    assert "global_stats" in res
    assert "community_df" in res
    assert "mst_df" in res
