from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ood_auroc_auprc(ood_flag: np.ndarray, ood_score: np.ndarray) -> dict:
    y = np.asarray(ood_flag).astype(int)
    s = np.asarray(ood_score, dtype=float)

    if y.min() == y.max():
        return {"auroc": float("nan"), "auprc": float("nan")}

    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }
