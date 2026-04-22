#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.threshold_scan import threshold_scan
from src.calibration.weighted_calibration import inverse_frequency_weights
from src.data.io import read_h5ad, write_h5ad
from src.data.preprocess import preprocess_ref_query
from src.data.splitters import assign_v1_splits, summarize_splits
from src.eval.metrics_group import coverage_disparity, group_selective_metrics, worst_group_risk
from src.eval.metrics_ood import ood_auroc_auprc
from src.eval.metrics_selective import coverage_at_target_risk, risk_at_target_coverage, risk_coverage_curve
from src.eval.plots import plot_ood_curves, plot_risk_coverage, plot_subgroup_risk_bar
from src.models.embedding_pca import fit_pca_embedding
from src.models.embedding_scvi import choose_batch_key, train_scvi_and_embed
from src.models.predict import predict_all_cells
from src.models.trainer_classifier import train_classifier_on_split
from src.scores.combine import combined_score
from src.scores.confidence import margin_score, max_prob_score, neg_entropy_score
from src.scores.normalize import normalize_scores
from src.scores.shift import ref_distance_shift
from src.scores.support import knn_density_score, knn_label_agreement_score
from src.utils.seed import seed_everything


def _resolve_path(path_str: str, config_path: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _safe_col(obs: pd.DataFrame, col: str, fallback: str = "unknown") -> np.ndarray:
    if col not in obs.columns:
        return np.asarray([fallback] * len(obs), dtype=object)
    return obs[col].astype(str).values


def run_pipeline(config_path: Path) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    data_cfg = cfg["data"]
    keys_cfg = cfg["keys"]
    pre_cfg = cfg.get("preprocess", {})
    split_cfg = cfg.get("split", {})
    model_cfg = cfg.get("model", {})
    scoring_cfg = cfg.get("scoring", {})
    cal_cfg = cfg.get("calibration", {})
    eval_cfg = cfg.get("evaluation", {})
    out_cfg = cfg.get("output", {})

    label_key = keys_cfg.get("label_key", "cell_type_coarse")
    group_keys = keys_cfg.get("group_keys", ["study_id", "batch_id"])
    split_key = keys_cfg.get("split_key", "split")
    ood_key = keys_cfg.get("ood_key", "is_ood")

    ref = read_h5ad(_resolve_path(data_cfg["ref_h5ad"], config_path))
    qry = read_h5ad(_resolve_path(data_cfg["qry_h5ad"], config_path))

    adata, used_hvg_batch = preprocess_ref_query(
        ref=ref,
        qry=qry,
        batch_key_candidates=pre_cfg.get("batch_key_candidates", ["study_id", "batch_id"]),
        min_genes=int(pre_cfg.get("min_genes", 200)),
        min_cells=int(pre_cfg.get("min_cells", 10)),
        max_pct_mt=float(pre_cfg.get("max_pct_mt", 20.0)),
        n_top_genes=int(pre_cfg.get("n_top_genes", 3000)),
    )

    assign_v1_splits(
        adata=adata,
        label_key=label_key,
        ref_query_key="ref_query",
        split_key=split_key,
        ood_key=ood_key,
        calib_frac=float(split_cfg.get("calib_frac", 0.5)),
        seed=int(split_cfg.get("seed", seed)),
        holdout_cell_types=split_cfg.get("holdout_cell_types", []),
        calib_exclude_ood=bool(split_cfg.get("calib_exclude_ood", True)),
    )

    output_root = _resolve_path(out_cfg.get("root", "outputs/v1"), config_path)
    fig_root = _resolve_path(out_cfg.get("figures_dir", "figures/v1"), config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    write_h5ad(adata, output_root / "preprocessed_with_splits.h5ad")
    split_summary = summarize_splits(adata, split_key=split_key, ood_key=ood_key, label_key=label_key)
    split_summary.to_csv(output_root / "split_summary.csv", index=False)

    embedding_mode = str(model_cfg.get("embedding", "scvi")).lower()
    latent_dim = int(model_cfg.get("latent_dim", 30))

    if embedding_mode == "scvi":
        import scvi

        scvi.settings.seed = seed
        batch_key = choose_batch_key(adata, model_cfg.get("batch_key_candidates", ["study_id", "batch_id"]))
        vae, latent = train_scvi_and_embed(
            adata=adata,
            batch_key=batch_key,
            n_latent=latent_dim,
            max_epochs=int(model_cfg.get("max_epochs", 100)),
        )
        (output_root / "models").mkdir(parents=True, exist_ok=True)
        vae.save(output_root / "models" / "scvi_model", overwrite=True, save_anndata=False)
        model_note = {"embedding": "scvi", "batch_key": batch_key, "hvg_batch_key": used_hvg_batch}
    elif embedding_mode == "pca":
        pca, latent = fit_pca_embedding(adata, n_components=latent_dim)
        (output_root / "models").mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, output_root / "models" / "pca_model.joblib")
        model_note = {"embedding": "pca", "hvg_batch_key": used_hvg_batch}
    else:
        raise ValueError(f"Unsupported embedding mode: {embedding_mode}")

    np.save(output_root / "models" / "latent.npy", latent)

    clf, train_meta = train_classifier_on_split(
        adata=adata,
        latent=latent,
        label_key=label_key,
        split_key=split_key,
        train_split_name="train_ref",
        hidden_dim=int(model_cfg.get("classifier_hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        max_iter=int(model_cfg.get("classifier_max_iter", 200)),
        seed=seed,
    )
    joblib.dump(clf, output_root / "models" / "classifier_mlp.joblib")
    train_meta.to_csv(output_root / "models" / "train_summary.csv", index=False)

    pred_df, probs, logits, classes = predict_all_cells(adata, latent, clf, label_key=label_key, split_key=split_key)

    pred_dir = output_root / "predictions"
    score_dir = output_root / "scores"
    cal_dir = output_root / "calibration"
    metric_dir = output_root / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(pred_dir / "all_predictions.csv", index=False)
    np.save(pred_dir / "all_probs.npy", probs)
    np.save(pred_dir / "all_logits.npy", logits)
    pd.DataFrame({"class_name": classes}).to_csv(pred_dir / "classes.csv", index=False)

    ref_mask = adata.obs[split_key].astype(str).values == "train_ref"
    calib_mask = adata.obs[split_key].astype(str).values == "calib_query"
    test_mask = adata.obs[split_key].astype(str).values == "test_query"
    qry_mask = calib_mask | test_mask

    z_ref = latent[ref_mask]
    z_qry = latent[qry_mask]

    conf_max = max_prob_score(probs)
    conf_margin = margin_score(probs)
    conf_neg_ent = neg_entropy_score(probs)

    support_density = np.full(len(adata), np.nan)
    support_label_agree = np.full(len(adata), np.nan)
    support = np.full(len(adata), np.nan)
    shift = np.full(len(adata), np.nan)

    if len(z_qry) > 0 and len(z_ref) > 0:
        y_ref = adata.obs.loc[ref_mask, label_key].astype(str).values
        y_pred_q = pred_df.loc[qry_mask, "y_pred"].astype(str).values
        q_support_density = knn_density_score(z_qry, z_ref, k=int(scoring_cfg.get("knn_k", 30)))
        q_support_agree = knn_label_agreement_score(
            z_query=z_qry,
            z_ref=z_ref,
            y_ref=y_ref,
            y_pred=y_pred_q,
            k=int(scoring_cfg.get("knn_k", 30)),
        )
        q_shift = ref_distance_shift(z_qry, z_ref, k=int(scoring_cfg.get("knn_k", 30)))
        support_density[qry_mask] = q_support_density
        support_label_agree[qry_mask] = q_support_agree
        shift[qry_mask] = q_shift

    support_method = str(scoring_cfg.get("support_method", "knn_label_agreement"))
    if support_method == "knn_density":
        support = support_density
    elif support_method == "knn_label_agreement":
        support = support_label_agree
    else:
        raise ValueError(f"Unsupported support_method: {support_method}")

    score_df = pred_df.copy()
    score_df["is_ood"] = adata.obs[ood_key].astype(int).values

    for gk in group_keys:
        score_df[gk] = _safe_col(adata.obs, gk)

    score_df["conf_max_prob"] = conf_max
    score_df["conf_margin"] = conf_margin
    score_df["conf_neg_entropy"] = conf_neg_ent
    score_df["support_density"] = support_density
    score_df["support_label_agreement"] = support_label_agree
    score_df["support"] = support
    score_df["shift"] = shift

    norm_method = str(scoring_cfg.get("normalize", "zscore"))
    l_cfg = scoring_cfg.get("lambdas", {"l1": 1.0, "l2": 1.0, "l3": 1.0})
    alpha = float(cal_cfg.get("alpha", 0.05))
    min_accept = int(cal_cfg.get("min_accept", 20))

    calib_conf = score_df.loc[calib_mask, "conf_max_prob"].values
    calib_support = score_df.loc[calib_mask, "support"].values
    calib_shift = score_df.loc[calib_mask, "shift"].values

    conf_n_all, conf_stats = normalize_scores(score_df["conf_max_prob"].values, method=norm_method, stats=None)
    support_n_all, support_stats = normalize_scores(score_df["support"].fillna(np.nanmean(calib_support)).values, method=norm_method, stats=None)
    shift_n_all, shift_stats = normalize_scores(score_df["shift"].fillna(np.nanmean(calib_shift)).values, method=norm_method, stats=None)

    score_df["conf_max_prob_norm"] = conf_n_all
    score_df["support_norm"] = support_n_all
    score_df["shift_norm"] = shift_n_all
    # Optional V1 lambda grid-search on calibration split.
    selected_l1 = float(l_cfg.get("l1", 1.0))
    selected_l2 = float(l_cfg.get("l2", 1.0))
    selected_l3 = float(l_cfg.get("l3", 1.0))
    lambda_grid = scoring_cfg.get("lambda_grid")
    if lambda_grid:
        l1_list = [float(v) for v in lambda_grid.get("l1", [selected_l1])]
        l2_list = [float(v) for v in lambda_grid.get("l2", [selected_l2])]
        l3_list = [float(v) for v in lambda_grid.get("l3", [selected_l3])]

        calib_work = score_df.loc[calib_mask].copy()
        y_true_cal_tmp = calib_work["y_true"].values
        y_pred_cal_tmp = calib_work["y_pred"].values

        best_combo = None
        for l1 in l1_list:
            for l2 in l2_list:
                for l3 in l3_list:
                    s_tmp = combined_score(
                        conf=calib_work["conf_max_prob_norm"].values,
                        support=calib_work["support_norm"].values,
                        shift=calib_work["shift_norm"].values,
                        l1=l1,
                        l2=l2,
                        l3=l3,
                    )
                    best_tmp, _ = threshold_scan(
                        y_true=y_true_cal_tmp,
                        y_pred=y_pred_cal_tmp,
                        scores=s_tmp,
                        alpha=alpha,
                        min_accept=min_accept,
                    )

                    candidate = {
                        "l1": l1,
                        "l2": l2,
                        "l3": l3,
                        "cov": float(best_tmp["coverage"]),
                        "risk": float(best_tmp["risk"]) if not pd.isna(best_tmp["risk"]) else np.inf,
                    }

                    if best_combo is None:
                        best_combo = candidate
                    else:
                        if candidate["risk"] <= alpha and best_combo["risk"] <= alpha and candidate["cov"] > best_combo["cov"]:
                            best_combo = candidate
                        elif candidate["risk"] <= alpha and best_combo["risk"] > alpha:
                            best_combo = candidate
                        elif candidate["risk"] > alpha and best_combo["risk"] > alpha and candidate["risk"] < best_combo["risk"]:
                            best_combo = candidate

        if best_combo is not None:
            selected_l1 = best_combo["l1"]
            selected_l2 = best_combo["l2"]
            selected_l3 = best_combo["l3"]

    score_df["score_combined"] = combined_score(
        conf=score_df["conf_max_prob_norm"].values,
        support=score_df["support_norm"].values,
        shift=score_df["shift_norm"].values,
        l1=selected_l1,
        l2=selected_l2,
        l3=selected_l3,
    )

    # For thresholding, every method is transformed so higher is better.
    score_df["score_confidence"] = score_df["conf_max_prob"]
    score_df["score_support"] = score_df["support"]
    score_df["score_shift"] = -score_df["shift"]

    score_df.to_csv(score_dir / "all_scores.csv", index=False)
    score_df.loc[calib_mask].to_csv(score_dir / "calib_scores.csv", index=False)
    score_df.loc[test_mask].to_csv(score_dir / "test_scores.csv", index=False)

    with open(score_dir / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": norm_method,
                "conf": conf_stats,
                "support": support_stats,
                "shift": shift_stats,
                "lambdas_selected": {"l1": selected_l1, "l2": selected_l2, "l3": selected_l3},
                "lambdas_default": {
                    "l1": float(l_cfg.get("l1", 1.0)),
                    "l2": float(l_cfg.get("l2", 1.0)),
                    "l3": float(l_cfg.get("l3", 1.0)),
                },
                "lambda_grid": lambda_grid,
                "model": model_note,
            },
            f,
            indent=2,
        )

    calibration_mode = str(cal_cfg.get("mode", "standard"))
    weight_group_key = str(cal_cfg.get("weight_group_key", group_keys[0] if group_keys else "study_id"))

    methods = [
        "score_confidence",
        "score_support",
        "score_shift",
        "score_combined",
    ]

    calib_df = score_df.loc[calib_mask].copy()
    test_df = score_df.loc[test_mask].copy()

    calib_summary_rows = []
    curve_map_test = {}

    y_true_cal = calib_df["y_true"].values
    y_pred_cal = calib_df["y_pred"].values

    if calibration_mode == "weighted":
        weights = inverse_frequency_weights(calib_df[weight_group_key].astype(str).values)
    else:
        weights = None

    for method in methods:
        c_scores = calib_df[method].values.astype(float)
        best, c_curve = threshold_scan(
            y_true=y_true_cal,
            y_pred=y_pred_cal,
            scores=c_scores,
            alpha=alpha,
            min_accept=min_accept,
            sample_weight=weights,
        )

        c_curve.insert(0, "method", method)
        c_curve.to_csv(cal_dir / f"curve_calib_{method}.csv", index=False)

        calib_summary_rows.append(
            {
                "method": method,
                "alpha": alpha,
                "tau": float(best["tau"]),
                "calib_risk": float(best["risk"]) if not pd.isna(best["risk"]) else np.nan,
                "calib_cov": float(best["coverage"]),
                "calib_n_accept": int(best["n_accept"]),
                "calibration_mode": calibration_mode,
            }
        )

        y_true_t = test_df["y_true"].values
        y_pred_t = test_df["y_pred"].values
        s_test = test_df[method].values.astype(float)
        taus = np.quantile(s_test, np.linspace(0.0, 1.0, int(eval_cfg.get("n_curve_points", 80))))
        t_curve = risk_coverage_curve(y_true_t, y_pred_t, s_test, taus)
        t_curve.insert(0, "method", method)
        t_curve.to_csv(metric_dir / f"curve_test_{method}.csv", index=False)
        curve_map_test[method] = t_curve

    calib_summary_df = pd.DataFrame(calib_summary_rows)
    calib_summary_df.to_csv(cal_dir / "calibration_summary.csv", index=False)

    operating_rows = []
    target_rows = []
    subgroup_summary_rows = []
    target_covs = eval_cfg.get("target_coverages", [0.5, 0.7, 0.9])

    ood_rows = []
    group_rows = []

    for _, row in calib_summary_df.iterrows():
        method = row["method"]
        tau = float(row["tau"])

        s_test = test_df[method].values.astype(float)
        y_true_t = test_df["y_true"].values
        y_pred_t = test_df["y_pred"].values
        accept = s_test >= tau

        risk = float((y_true_t[accept] != y_pred_t[accept]).mean()) if accept.sum() > 0 else np.nan
        cov = float(accept.mean())

        operating_rows.append(
            {
                "method": method,
                "tau": tau,
                "test_risk": risk,
                "test_coverage": cov,
                "test_n_accept": int(accept.sum()),
            }
        )

        curve = curve_map_test[method]
        target_rows.append(
            {
                "method": method,
                "metric": f"coverage_at_risk_{alpha}",
                "value": coverage_at_target_risk(curve, alpha),
            }
        )
        for tc in target_covs:
            target_rows.append(
                {
                    "method": method,
                    "metric": f"risk_at_coverage_{tc}",
                    "value": risk_at_target_coverage(curve, float(tc)),
                }
            )

        if method == "score_combined":
            for gk in group_keys:
                gdf = group_selective_metrics(
                    y_true=y_true_t,
                    y_pred=y_pred_t,
                    accept=accept,
                    groups=test_df[gk].astype(str).values,
                )
                gdf.insert(0, "group_key", gk)
                gdf.insert(0, "method", method)
                group_rows.append(gdf)

                subgroup_summary_rows.append(
                    {
                        "method": method,
                        "group_key": gk,
                        "worst_group_risk": worst_group_risk(gdf),
                        "coverage_disparity": coverage_disparity(gdf),
                    }
                )

            if ood_key in test_df.columns:
                ood_flag = test_df[ood_key].astype(int).values
                ood_score_combined = -test_df["score_combined"].values
                ood_score_shift = test_df["shift"].values
                ood_comb = ood_auroc_auprc(ood_flag, ood_score_combined)
                ood_shift = ood_auroc_auprc(ood_flag, ood_score_shift)
                ood_rows.append({"method": "combined", **ood_comb})
                ood_rows.append({"method": "shift", **ood_shift})

                plot_ood_curves(ood_flag, ood_score_combined, fig_root / "ood_combined")

    operating_df = pd.DataFrame(operating_rows)
    operating_df.to_csv(metric_dir / "operating_points.csv", index=False)
    pd.DataFrame(target_rows).to_csv(metric_dir / "target_metrics.csv", index=False)
    pd.DataFrame(subgroup_summary_rows).to_csv(metric_dir / "subgroup_summary.csv", index=False)

    operating_df.to_csv(metric_dir / "evaluation_summary.csv", index=False)

    if group_rows:
        group_df = pd.concat(group_rows, ignore_index=True)
        group_df.to_csv(metric_dir / "group_metrics_combined.csv", index=False)

        first_key = group_keys[0]
        first = group_df[group_df["group_key"] == first_key].copy()
        plot_subgroup_risk_bar(
            first,
            fig_root / f"subgroup_risk_{first_key}.png",
            title=f"Worst-group diagnostic ({first_key})",
        )

    if ood_rows:
        pd.DataFrame(ood_rows).to_csv(metric_dir / "ood_metrics.csv", index=False)

    plot_risk_coverage(
        {
            "confidence": curve_map_test["score_confidence"],
            "support": curve_map_test["score_support"],
            "shift": curve_map_test["score_shift"],
            "combined": curve_map_test["score_combined"],
        },
        fig_root / "risk_coverage_main.png",
    )

    with open(output_root / "RUN_COMPLETE.txt", "w", encoding="utf-8") as f:
        f.write("V1 pipeline complete. See figures/ and outputs/ folders.\n")

    print(f"[OK] Output root: {output_root}")
    print(f"[OK] Figure root: {fig_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V1 selective prediction pipeline.")
    parser.add_argument("--config", required=True, help="Path to V1 config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(Path(args.config).resolve())


if __name__ == "__main__":
    main()
