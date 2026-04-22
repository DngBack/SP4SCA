#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.metrics import confusion_matrix


def predict_query(adata, vae, clf, label_key: str = "cell_type_coarse") -> pd.DataFrame:
    if label_key not in adata.obs.columns:
        raise KeyError(f"Missing label_key in adata.obs: {label_key}")

    z = vae.get_latent_representation()
    is_qry = adata.obs["ref_query"].values == "qry"

    X_qry = z[is_qry]
    y_true = adata.obs.loc[is_qry, label_key].astype(str).values

    proba = clf.predict_proba(X_qry)
    classes = clf.classes_
    pred = classes[proba.argmax(axis=1)]

    score_conf = proba.max(axis=1)
    entropy = -(proba * np.log(proba + 1e-12)).sum(axis=1)
    score_neg_entropy = -entropy

    out = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": pred,
            "score_conf": score_conf,
            "score_neg_entropy": score_neg_entropy,
        },
        index=adata.obs_names[is_qry],
    )
    out.index.name = "cell_id"
    return out


def selective_curve(df: pd.DataFrame, score_col: str, n_thresh: int = 50) -> pd.DataFrame:
    scores = df[score_col].values
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    thresholds = np.quantile(scores, np.linspace(0.0, 1.0, n_thresh))
    rows = []

    for t in thresholds:
        keep = scores >= t
        if keep.sum() == 0:
            continue

        coverage = keep.mean()
        risk = (y_pred[keep] != y_true[keep]).mean()
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(coverage),
                "risk": float(risk),
                "n_accept": int(keep.sum()),
            }
        )

    curve = pd.DataFrame(rows).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    return curve.reset_index(drop=True)


def summarize_at_coverage(curve_df: pd.DataFrame, target_coverage: float) -> dict[str, float]:
    idx = (curve_df["coverage"] - target_coverage).abs().idxmin()
    row = curve_df.loc[idx]
    return {
        "coverage_target": float(target_coverage),
        "coverage_actual": float(row["coverage"]),
        "risk_actual": float(row["risk"]),
        "n_accept": int(row["n_accept"]),
        "threshold": float(row["threshold"]),
    }


def plot_curves(curves: dict[str, pd.DataFrame], out_png: Path) -> None:
    plt.figure(figsize=(5, 4))
    for name, cdf in curves.items():
        plt.plot(cdf["coverage"], cdf["risk"], label=name)
    plt.xlabel("Coverage")
    plt.ylabel("Selective risk")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


def trend_is_mostly_nonincreasing(values: np.ndarray, tolerance: float = 0.05) -> bool:
    deltas = np.diff(values)
    frac_up = float((deltas > 1e-8).mean()) if len(deltas) else 0.0
    return frac_up <= tolerance


def run_sanity_checks(pred_df: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> list[str]:
    warnings: list[str] = []

    full_acc = float((pred_df["y_true"] == pred_df["y_pred"]).mean())
    if full_acc < 0.2:
        warnings.append(f"Full query accuracy is very low: {full_acc:.3f}")

    for score_name, cdf in curves.items():
        cov = cdf["coverage"].values
        risk = cdf["risk"].values
        if not trend_is_mostly_nonincreasing(cov):
            warnings.append(f"Coverage trend issue for {score_name}: not mostly decreasing with threshold.")
        if not trend_is_mostly_nonincreasing(risk):
            warnings.append(f"Risk trend issue for {score_name}: not mostly decreasing with threshold.")

    shared_truth = set(pred_df["y_true"].unique())
    predicted_classes = set(pred_df["y_pred"].unique())
    collapsed = sorted(shared_truth - predicted_classes)
    if collapsed:
        warnings.append("Some query classes are never predicted: " + ", ".join(collapsed))

    corr = float(pred_df["score_conf"].corr(pred_df["score_neg_entropy"]))
    if corr < 0.5:
        warnings.append(f"Low correlation between confidence and negative entropy: {corr:.3f}")

    return warnings


def save_confusion_on_accepted(
    pred_df: pd.DataFrame,
    score_col: str,
    threshold: float,
    out_csv: Path,
) -> None:
    keep = pred_df[score_col] >= threshold
    accepted = pred_df.loc[keep].copy()

    labels = sorted(set(accepted["y_true"]).union(set(accepted["y_pred"])))
    cm = confusion_matrix(accepted["y_true"], accepted["y_pred"], labels=labels)

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "y_true"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(out_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selective prediction on query cells.")
    parser.add_argument("--adata", required=True, help="Preprocessed .h5ad")
    parser.add_argument("--model-dir", required=True, help="Directory from train_scvi_probe.py")
    parser.add_argument("--out-dir", required=True, help="Output directory for metrics/figures")
    parser.add_argument("--label-key", default="cell_type_coarse")
    parser.add_argument("--n-thresh", type=int, default=50)
    parser.add_argument("--cm-score", default="score_conf")
    parser.add_argument("--cm-target-coverage", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.adata)
    model_dir = Path(args.model_dir)

    vae = scvi.model.SCVI.load(model_dir / "scvi_model", adata=adata)
    clf = joblib.load(model_dir / "linear_probe.joblib")

    pred_df = predict_query(adata, vae, clf, label_key=args.label_key)
    pred_df.to_csv(out_dir / "predictions_query.csv")

    curve_conf = selective_curve(pred_df, "score_conf", n_thresh=args.n_thresh)
    curve_ent = selective_curve(pred_df, "score_neg_entropy", n_thresh=args.n_thresh)

    curve_conf.to_csv(out_dir / "curve_confidence.csv", index=False)
    curve_ent.to_csv(out_dir / "curve_neg_entropy.csv", index=False)

    curves = {"confidence": curve_conf, "neg_entropy": curve_ent}
    plot_curves(curves, out_dir / "risk_coverage.png")

    full_acc = float((pred_df["y_true"] == pred_df["y_pred"]).mean())
    rows = [{"metric": "full_query_accuracy", "score": "all", "value": full_acc}]

    for score_name, cdf in curves.items():
        for target_cov in (0.9, 0.7, 0.5, 0.3):
            summary = summarize_at_coverage(cdf, target_cov)
            rows.append(
                {
                    "metric": f"risk_at_cov_{target_cov}",
                    "score": score_name,
                    "value": summary["risk_actual"],
                }
            )
            rows.append(
                {
                    "metric": f"actual_cov_for_target_{target_cov}",
                    "score": score_name,
                    "value": summary["coverage_actual"],
                }
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    curve_for_cm = curve_conf if args.cm_score == "score_conf" else curve_ent
    cm_summary = summarize_at_coverage(curve_for_cm, args.cm_target_coverage)
    save_confusion_on_accepted(
        pred_df=pred_df,
        score_col=args.cm_score,
        threshold=cm_summary["threshold"],
        out_csv=out_dir / "confusion_matrix_accepted.csv",
    )

    warnings = run_sanity_checks(pred_df, curves)
    if warnings:
        pd.DataFrame({"warning": warnings}).to_csv(out_dir / "sanity_warnings.csv", index=False)
        print("Sanity warnings detected; see sanity_warnings.csv")
    else:
        print("Sanity checks passed.")

    print(f"Saved eval outputs to {out_dir}")


if __name__ == "__main__":
    main()
