import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def ref_distance_shift(z_query: np.ndarray, z_ref: np.ndarray, k: int = 30) -> np.ndarray:
    k_eff = max(1, min(k, len(z_ref)))
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(z_ref)
    distances, _ = nn.kneighbors(z_query)
    # Higher shift means farther from reference.
    return distances.mean(axis=1)


def train_domain_discriminator(z_ref: np.ndarray, z_query_calib: np.ndarray):
    X = np.vstack([z_ref, z_query_calib])
    y = np.hstack([
        np.zeros(len(z_ref), dtype=int),
        np.ones(len(z_query_calib), dtype=int),
    ])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def domain_shift_score(discriminator, z_query: np.ndarray) -> np.ndarray:
    return discriminator.predict_proba(z_query)[:, 1]
