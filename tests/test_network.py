import pytest
import pandas as pd
import numpy as np
import networkx as nx
from src.network.mst import compute_correlation_distance, compute_spillover_distance, construct_mst
from src.network.spectral import detect_communities

def test_mst_and_spectral_communities():
    spill = pd.DataFrame([
        [80.0, 20.0],
        [30.0, 70.0]
    ], index=["Bank", "IT"], columns=["Bank", "IT"])
    
    comms = detect_communities(spill, n_communities="auto")
    assert len(comms) == 2
    assert "Bank" in comms.index
    
    data = pd.DataFrame({
        "Bank": [1.0, 2.0, 3.0, 4.0],
        "IT": [2.0, 4.0, 6.0, 8.0]
    })
    dist = compute_correlation_distance(data)
    mst_graph = construct_mst(dist, algorithm="prim")
    assert mst_graph.number_of_nodes() == 2
    assert mst_graph.number_of_edges() == 1
    assert nx.is_tree(mst_graph)


def test_spillover_distance_and_prims_algorithm():
    # 3 sectors with asymmetric directed spillover: S_ij != S_ji
    spill = pd.DataFrame([
        [70.0, 15.0, 5.0],   # Bank -> IT=15, Auto=5
        [10.0, 65.0, 20.0],  # IT -> Bank=10, Auto=20
        [2.0, 8.0, 85.0]     # Auto -> Bank=2, IT=8
    ], index=["Bank", "IT", "Auto"], columns=["Bank", "IT", "Auto"])

    # Test 'sum' method: W_ij = S_ij + S_ji
    dist_sum, weights_sum = compute_spillover_distance(spill, method="sum")
    # Bank-IT bilateral sum: 15 + 10 = 25
    assert weights_sum.loc["Bank", "IT"] == 25.0
    assert weights_sum.loc["IT", "Bank"] == 25.0
    # IT-Auto bilateral sum: 20 + 8 = 28
    assert weights_sum.loc["IT", "Auto"] == 28.0
    # Bank-Auto bilateral sum: 5 + 2 = 7
    assert weights_sum.loc["Bank", "Auto"] == 7.0

    # Test 'max' method: W_ij = max(S_ij, S_ji)
    dist_max, weights_max = compute_spillover_distance(spill, method="max")
    # Bank-IT bilateral max: max(15, 10) = 15
    assert weights_max.loc["Bank", "IT"] == 15.0
    # IT-Auto bilateral max: max(20, 8) = 20
    assert weights_max.loc["IT", "Auto"] == 20.0

    # Test Prim's MST construction
    mst_g = construct_mst(dist_sum, algorithm="prim", weights_df=weights_sum)
    assert mst_g.number_of_nodes() == 3
    assert mst_g.number_of_edges() == 2
    assert nx.is_tree(mst_g)
    # The strongest bilateral channels are IT-Auto (28) and Bank-IT (25), so IT is the hub connecting Bank and Auto!
    assert mst_g.has_edge("Bank", "IT")
    assert mst_g.has_edge("IT", "Auto")
    assert not mst_g.has_edge("Bank", "Auto")

