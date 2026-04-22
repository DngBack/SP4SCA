from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w_sum = weights.sum()
    if w_sum <= 0:
        return float("nan")
    return float((values * weights).sum() / w_sum)


def selective_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    tau: float,
    sample_weight: np.ndarray | None = None,
) -> dict:
    accept = scores >= tau
    n_accept = int(accept.sum())
    coverage = float(accept.mean())

    if n_accept == 0:
        return {"tau": float(tau), "coverage": coverage, "risk": np.nan, "n_accept": 0}

    err = (y_true != y_pred).astype(float)
    if sample_weight is None:
        risk = float(err[accept].mean())
    else:
        risk = _safe_weighted_mean(err[accept], sample_weight[accept])

    return {"tau": float(tau), "coverage": coverage, "risk": risk, "n_accept": n_accept}


def threshold_scan(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    alpha: float,
    min_accept: int = 20,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = np.asarray(scores, dtype=float)
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)

    taus = np.unique(scores)
    rows = []
    best = None

    for tau in np.sort(taus):
        st = selective_stats(y_true, y_pred, scores, tau, sample_weight=sample_weight)
        rows.append(st)
        if st["n_accept"] < min_accept or np.isnan(st["risk"]):
            continue
        if st["risk"] <= alpha:
            if best is None or st["coverage"] > best["coverage"]:
                best = st

    curve = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    if best is None:
        # Fallback to lowest-risk operating point if alpha is infeasible.
        feasible = curve[curve["n_accept"] >= min_accept].copy()
        if feasible.empty:
            best = {"tau": float(np.max(scores) + 1e-8), "coverage": 0.0, "risk": np.nan, "n_accept": 0}
        else:
            idx = feasible["risk"].idxmin()
            best = feasible.loc[idx].to_dict()
            best["note"] = "alpha_infeasible_fallback_to_min_risk"

    best["alpha"] = float(alpha)
    return best, curve
