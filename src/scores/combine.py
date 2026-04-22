import numpy as np


def combined_score(
    conf: np.ndarray,
    support: np.ndarray,
    shift: np.ndarray,
    l1: float = 1.0,
    l2: float = 1.0,
    l3: float = 1.0,
) -> np.ndarray:
    return l1 * conf + l2 * support - l3 * shift
