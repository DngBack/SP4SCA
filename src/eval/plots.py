from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_risk_coverage(curves: dict[str, pd.DataFrame], out_png: str | Path) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5.5, 4.0))
    for name, cdf in curves.items():
        cdf = cdf.sort_values("coverage")
        plt.plot(cdf["coverage"], cdf["risk"], label=name, linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Selective Risk")
    plt.title("Risk-Coverage")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_subgroup_risk_bar(group_df: pd.DataFrame, out_png: str | Path, title: str) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    gdf = group_df.sort_values("selective_risk", ascending=False).copy()

    plt.figure(figsize=(max(6.0, 0.4 * len(gdf)), 4.0))
    plt.bar(gdf["group_name"], gdf["selective_risk"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Selective Risk")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_ood_curves(ood_flag: np.ndarray, ood_score: np.ndarray, out_prefix: str | Path) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    y = np.asarray(ood_flag).astype(int)
    s = np.asarray(ood_score, dtype=float)

    if y.min() == y.max():
        return

    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    plt.figure(figsize=(5.0, 4.0))
    plt.plot(fpr, tpr, linewidth=2)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("OOD ROC")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_roc.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(5.0, 4.0))
    plt.plot(rec, prec, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("OOD PR")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_pr.png"), dpi=220)
    plt.close()
