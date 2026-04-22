#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def choose_batch_key(adata, batch_key_candidates: list[str]) -> str:
    for key in batch_key_candidates:
        if key in adata.obs.columns:
            return key
    raise KeyError(f"None of batch key candidates found in adata.obs: {batch_key_candidates}")


def train_scvi_probe(
    adata,
    label_key: str = "cell_type_coarse",
    batch_key: str = "batch_id",
    n_latent: int = 30,
    max_epochs: int = 100,
):
    if label_key not in adata.obs.columns:
        raise KeyError(f"Missing label_key in adata.obs: {label_key}")

    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)

    vae = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        gene_likelihood="nb",
    )
    vae.train(max_epochs=max_epochs, early_stopping=True)

    z = vae.get_latent_representation()

    is_ref = adata.obs["ref_query"].values == "ref"
    X_ref = z[is_ref]
    y_ref = adata.obs.loc[is_ref, label_key].astype(str).values

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_ref, y_ref)

    train_pred = clf.predict(X_ref)
    train_acc = float(accuracy_score(y_ref, train_pred))

    return vae, clf, train_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scVI backbone + linear probe.")
    parser.add_argument("--adata", required=True, help="Preprocessed .h5ad")
    parser.add_argument("--out-dir", required=True, help="Model output directory")
    parser.add_argument("--label-key", default="cell_type_coarse")
    parser.add_argument("--batch-key-candidates", default="study_id,batch_id")
    parser.add_argument("--n-latent", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scvi.settings.seed = args.seed
    adata = sc.read_h5ad(args.adata)

    batch_candidates = [x.strip() for x in args.batch_key_candidates.split(",") if x.strip()]
    batch_key = choose_batch_key(adata, batch_candidates)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, clf, train_acc = train_scvi_probe(
        adata=adata,
        label_key=args.label_key,
        batch_key=batch_key,
        n_latent=args.n_latent,
        max_epochs=args.max_epochs,
    )

    model_dir = out_dir / "scvi_model"
    vae.save(model_dir, overwrite=True, save_anndata=False)
    joblib.dump(clf, out_dir / "linear_probe.joblib")

    summary = pd.DataFrame(
        [
            {
                "train_ref_accuracy": train_acc,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "batch_key_used": batch_key,
            }
        ]
    )
    summary.to_csv(out_dir / "train_summary.csv", index=False)

    print(f"Saved model artifacts to {out_dir}")
    print(f"Used batch_key for SCVI: {batch_key}")
    print(f"Reference train accuracy (sanity): {train_acc:.4f}")


if __name__ == "__main__":
    main()
