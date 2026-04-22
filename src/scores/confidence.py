import numpy as np


def max_prob_score(probs: np.ndarray) -> np.ndarray:
    return probs.max(axis=1)


def margin_score(probs: np.ndarray) -> np.ndarray:
    part = np.partition(probs, -2, axis=1)
    top2 = part[:, -2:]
    return top2[:, 1] - top2[:, 0]


def neg_entropy_score(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    ent = -(probs * np.log(probs + eps)).sum(axis=1)
    return -ent
