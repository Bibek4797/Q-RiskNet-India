import pytest
import pandas as pd
import numpy as np
from src.network.mst import compute_correlation_distance, construct_mst
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
    mst_graph = construct_mst(dist)
    assert mst_graph.number_of_nodes() == 2
    assert mst_graph.number_of_edges() == 1
