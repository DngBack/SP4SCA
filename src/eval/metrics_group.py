from __future__ import annotations

import numpy as np
import pandas as pd



def group_selective_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    accept: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    groups = np.asarray(groups).astype(str)
    rows = []
    for g in np.unique(groups):
        mask = groups == g
        n = int(mask.sum())
        n_accept = int((accept & mask).sum())
        cov = float(n_accept / max(1, n))
        if n_accept == 0:
            risk = float("nan")
        else:
            risk = float((y_true[accept & mask] != y_pred[accept & mask]).mean())
        rows.append(
            {
                "group_name": g,
                "n": n,
                "accepted": n_accept,
                "coverage": cov,
                "selective_risk": risk,
            }
        )
    return pd.DataFrame(rows).sort_values("group_name").reset_index(drop=True)


def worst_group_risk(group_df: pd.DataFrame, min_accept: int = 1) -> float:
    valid = group_df[(group_df["accepted"] >= min_accept) & (~group_df["selective_risk"].isna())]
    if valid.empty:
        return float("nan")
    return float(valid["selective_risk"].max())


def coverage_disparity(group_df: pd.DataFrame) -> float:
    cov = group_df["coverage"].values
    if len(cov) == 0:
        return float("nan")
    return float(np.max(cov) - np.min(cov))
