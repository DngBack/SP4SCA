import numpy as np


def inverse_frequency_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups).astype(str)
    uniq, cnt = np.unique(groups, return_counts=True)
    freq = {g: c for g, c in zip(uniq, cnt)}
    w = np.asarray([1.0 / float(freq[g]) for g in groups], dtype=float)
    w *= len(w) / w.sum()
    return w
