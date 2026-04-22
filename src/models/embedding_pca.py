import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


def fit_pca_embedding(adata, n_components: int = 30):
    X = adata.layers.get("counts", adata.X)
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    # Library-size normalize + log1p as a light baseline.
    lib = X.sum(axis=1, keepdims=True)
    lib = np.clip(lib, 1e-12, None)
    Xn = X / lib * 1e4
    Xn = np.log1p(Xn)

    pca = PCA(n_components=n_components, random_state=42)
    z = pca.fit_transform(Xn)
    return pca, z
