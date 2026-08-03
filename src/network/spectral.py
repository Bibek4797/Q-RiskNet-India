import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
import src.diagnostics.logger as diag

def find_optimal_eigengap_k(similarity_matrix):
    """
    Computes optimal community count k using the Eigengap Heuristic on normalized graph Laplacian.
    """
    K = similarity_matrix.shape[0]
    if K <= 2:
        return 1
        
    deg_vector = np.sum(similarity_matrix, axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(deg_vector, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    L_sym = np.eye(K) - np.dot(np.dot(D_inv_sqrt, similarity_matrix), D_inv_sqrt)
    eigenvalues = np.sort(np.linalg.eigvalsh(L_sym))
    
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
    Uses Spectral Clustering to partition spillover network into natural risk clusters.
    """
    sectors = spillover_matrix.columns
    K = len(sectors)
    
    similarity = 0.5 * (spillover_matrix.values + spillover_matrix.values.T)
    np.fill_diagonal(similarity, 0.0)
    
    max_val = np.max(similarity)
    if max_val > 0:
        similarity = similarity / max_val
        
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
            diag.log_error("Spectral clustering failed. Falling back to zero-labels.", e)
            return pd.Series(0, index=sectors)
