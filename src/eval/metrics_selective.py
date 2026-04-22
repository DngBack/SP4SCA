from __future__ import annotations

import numpy as np
import pandas as pd


def selective_risk(y_true: np.ndarray, y_pred: np.ndarray, accept: np.ndarray) -> float:
    n = int(accept.sum())
    if n == 0:
        return float("nan")
    return float((y_true[accept] != y_pred[accept]).mean())


def coverage(accept: np.ndarray) -> float:
    return float(accept.mean())


def risk_coverage_curve(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, taus: np.ndarray) -> pd.DataFrame:
    rows = []
    for tau in np.sort(np.unique(taus)):
        accept = scores >= tau
        rows.append(
            {
                "tau": float(tau),
                "coverage": coverage(accept),
                "risk": selective_risk(y_true, y_pred, accept),
                "n_accept": int(accept.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)


def coverage_at_target_risk(curve: pd.DataFrame, alpha: float) -> float:
    good = curve[curve["risk"] <= alpha]
    if good.empty:
        return 0.0
    return float(good["coverage"].max())


def risk_at_target_coverage(curve: pd.DataFrame, target_cov: float) -> float:
    idx = (curve["coverage"] - target_cov).abs().idxmin()
    return float(curve.loc[idx, "risk"])
