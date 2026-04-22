import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_density_score(z_query: np.ndarray, z_ref: np.ndarray, k: int = 30) -> np.ndarray:
    k_eff = max(1, min(k, len(z_ref)))
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(z_ref)
    distances, _ = nn.kneighbors(z_query)
    # Higher score means better support.
    return -distances.mean(axis=1)


def knn_label_agreement_score(
    z_query: np.ndarray,
    z_ref: np.ndarray,
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    k: int = 30,
) -> np.ndarray:
    k_eff = max(1, min(k, len(z_ref)))
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(z_ref)
    _, indices = nn.kneighbors(z_query)
    neighbor_labels = y_ref[indices]
    agreement = (neighbor_labels == y_pred[:, None]).mean(axis=1)
    return agreement
