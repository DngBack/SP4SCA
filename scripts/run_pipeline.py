#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
import scvi
import yaml

from eval_selective import (
    plot_curves,
    predict_query,
    run_sanity_checks,
    save_confusion_on_accepted,
    selective_curve,
    summarize_at_coverage,
)
from preprocess import preprocess_ref_query
from train_scvi_probe import choose_batch_key, train_scvi_probe


def _resolve_path(path_str: str, config_path: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _validate_required_input_paths(config_path: Path, cfg: dict) -> tuple[Path, Path]:
    if "data" not in cfg:
        raise KeyError("Missing 'data' section in config.")
    data_cfg = cfg["data"]
    if "ref_h5ad" not in data_cfg or "qry_h5ad" not in data_cfg:
        raise KeyError("Config 'data' must contain both 'ref_h5ad' and 'qry_h5ad'.")

    ref_path = _resolve_path(data_cfg["ref_h5ad"], config_path)
    qry_path = _resolve_path(data_cfg["qry_h5ad"], config_path)

    missing = [p for p in (ref_path, qry_path) if not p.exists()]
    if missing:
        missing_lines = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "Required input file(s) not found:\n"
            f"{missing_lines}\n"
            "Update `data.ref_h5ad` and `data.qry_h5ad` in your config to valid .h5ad paths."
        )
    return ref_path, qry_path


def run_pipeline(config_path: Path) -> None:
    config_path = config_path.resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ref_path, qry_path = _validate_required_input_paths(config_path, cfg)

    keys_cfg = cfg["keys"]
    pre_cfg = cfg["preprocess"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    out_cfg = cfg["output"]

    ref = sc.read_h5ad(ref_path)
    qry = sc.read_h5ad(qry_path)

    adata, _ = preprocess_ref_query(
        ref=ref,
        qry=qry,
        batch_key_candidates=pre_cfg.get("batch_key_candidates", ["study_id", "batch_id"]),
        min_genes=int(pre_cfg.get("min_genes", 200)),
        min_cells=int(pre_cfg.get("min_cells", 10)),
        max_pct_mt=float(pre_cfg.get("max_pct_mt", 20.0)),
        n_top_genes=int(pre_cfg.get("n_top_genes", 3000)),
    )

    preprocessed_path = Path(out_cfg["preprocessed_h5ad"])
    preprocessed_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(preprocessed_path)

    scvi.settings.seed = int(train_cfg.get("seed", 0))
    batch_key = choose_batch_key(adata, pre_cfg.get("batch_key_candidates", ["study_id", "batch_id"]))

    vae, clf, train_acc = train_scvi_probe(
        adata=adata,
        label_key=keys_cfg["label_key"],
        batch_key=batch_key,
        n_latent=int(train_cfg.get("n_latent", 30)),
        max_epochs=int(train_cfg.get("max_epochs", 100)),
    )

    model_dir = Path(out_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    vae.save(model_dir / "scvi_model", overwrite=True, save_anndata=False)

    import joblib

    joblib.dump(clf, model_dir / "linear_probe.joblib")
    pd.DataFrame(
        [
            {
                "train_ref_accuracy": train_acc,
                "batch_key_used": batch_key,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
            }
        ]
    ).to_csv(model_dir / "train_summary.csv", index=False)

    pred_df = predict_query(adata, vae, clf, label_key=keys_cfg["label_key"])
    eval_dir = Path(out_cfg["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(eval_dir / "predictions_query.csv")

    n_thresh = int(eval_cfg.get("n_thresh", 50))
    curve_conf = selective_curve(pred_df, "score_conf", n_thresh=n_thresh)
    curve_ent = selective_curve(pred_df, "score_neg_entropy", n_thresh=n_thresh)
    curve_conf.to_csv(eval_dir / "curve_confidence.csv", index=False)
    curve_ent.to_csv(eval_dir / "curve_neg_entropy.csv", index=False)

    curves = {"confidence": curve_conf, "neg_entropy": curve_ent}
    plot_curves(curves, Path(out_cfg["risk_coverage_png"]))

    rows = [
        {
            "metric": "full_query_accuracy",
            "score": "all",
            "value": float((pred_df["y_true"] == pred_df["y_pred"]).mean()),
        }
    ]

    coverage_targets = eval_cfg.get("coverage_targets", [0.9, 0.7, 0.5, 0.3])
    for score_name, cdf in curves.items():
        for target_cov in coverage_targets:
            summary = summarize_at_coverage(cdf, float(target_cov))
            rows.append({"metric": f"risk_at_cov_{target_cov}", "score": score_name, "value": summary["risk_actual"]})
            rows.append(
                {
                    "metric": f"actual_cov_for_target_{target_cov}",
                    "score": score_name,
                    "value": summary["coverage_actual"],
                }
            )

    metrics = pd.DataFrame(rows)
    metrics_path = Path(out_cfg["metrics_csv"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    cm_score = eval_cfg.get("cm_score", "score_conf")
    cm_target_coverage = float(eval_cfg.get("cm_target_coverage", 0.7))
    cm_curve = curve_conf if cm_score == "score_conf" else curve_ent
    cm_summary = summarize_at_coverage(cm_curve, cm_target_coverage)
    save_confusion_on_accepted(
        pred_df=pred_df,
        score_col=cm_score,
        threshold=cm_summary["threshold"],
        out_csv=Path(out_cfg["confusion_matrix_csv"]),
    )

    warnings = run_sanity_checks(pred_df, curves)
    if warnings:
        pd.DataFrame({"warning": warnings}).to_csv(out_cfg["sanity_warnings_csv"], index=False)

    print(f"Preprocessed data: {preprocessed_path}")
    print(f"Model dir: {model_dir}")
    print(f"Eval dir: {eval_dir}")
    print(f"Metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full selective prediction pipeline from YAML config.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(Path(args.config))


if __name__ == "__main__":
    main()
