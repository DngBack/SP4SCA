#!/usr/bin/env python3
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def standardize_obs(adata: ad.AnnData, mapping: dict[str, str]) -> ad.AnnData:
    for src, dst in mapping.items():
        if src not in adata.obs.columns:
            raise KeyError(f"Missing source obs column: {src}")
        adata.obs[dst] = adata.obs[src].astype(str)
    return adata


def _ensure_counts_layer(adata: ad.AnnData) -> None:
    adata.layers["counts"] = adata.X.copy()


def _choose_batch_key(adata: ad.AnnData, candidates: list[str]) -> str | None:
    for key in candidates:
        if key in adata.obs.columns:
            return key
    return None


def preprocess_ref_query(
    ref: ad.AnnData,
    qry: ad.AnnData,
    batch_key_candidates: list[str] | None = None,
    min_genes: int = 200,
    min_cells: int = 10,
    max_pct_mt: float = 20.0,
    n_top_genes: int = 3000,
) -> tuple[ad.AnnData, str | None]:
    if batch_key_candidates is None:
        batch_key_candidates = ["study_id", "batch_id"]

    common = ref.var_names.intersection(qry.var_names)
    if len(common) == 0:
        raise ValueError("No common genes between reference and query datasets.")

    ref = ref[:, common].copy()
    qry = qry[:, common].copy()
    ref.obs_names = pd.Index([f"ref_{x}" for x in ref.obs_names.astype(str)])
    qry.obs_names = pd.Index([f"qry_{x}" for x in qry.obs_names.astype(str)])

    _ensure_counts_layer(ref)
    _ensure_counts_layer(qry)

    ref.obs["ref_query"] = "ref"
    qry.obs["ref_query"] = "qry"

    adata = ad.concat([ref, qry], join="inner", merge="same")
    adata.obs_names_make_unique()
    adata.X = adata.layers["counts"].copy()

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mt].copy()

    hvg_batch_key = _choose_batch_key(adata, batch_key_candidates)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        batch_key=hvg_batch_key,
        layer="counts",
        subset=True,
    )

    adata.X = adata.layers["counts"].copy()
    if not sparse.issparse(adata.layers["counts"]):
        adata.layers["counts"] = np.asarray(adata.layers["counts"])

    return adata, hvg_batch_key


def parse_obs_mapping(raw: str | None) -> dict[str, str]:
    if raw is None or raw.strip() == "":
        return {}
    out: dict[str, str] = {}
    for item in raw.split(","):
        src, dst = item.split(":", 1)
        out[src.strip()] = dst.strip()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ref+query for selective prediction.")
    parser.add_argument("--ref", required=True, help="Path to reference .h5ad")
    parser.add_argument("--qry", required=True, help="Path to query .h5ad")
    parser.add_argument("--out", required=True, help="Output preprocessed .h5ad")
    parser.add_argument("--batch-key-candidates", default="study_id,batch_id")
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--max-pct-mt", type=float, default=20.0)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--ref-obs-map", default="")
    parser.add_argument("--qry-obs-map", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ref = sc.read_h5ad(args.ref)
    qry = sc.read_h5ad(args.qry)

    ref_map = parse_obs_mapping(args.ref_obs_map)
    qry_map = parse_obs_mapping(args.qry_obs_map)
    if ref_map:
        ref = standardize_obs(ref, ref_map)
    if qry_map:
        qry = standardize_obs(qry, qry_map)

    candidates = [x.strip() for x in args.batch_key_candidates.split(",") if x.strip()]
    adata, used_batch = preprocess_ref_query(
        ref=ref,
        qry=qry,
        batch_key_candidates=candidates,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        max_pct_mt=args.max_pct_mt,
        n_top_genes=args.n_top_genes,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)

    n_ref = int((adata.obs["ref_query"] == "ref").sum())
    n_qry = int((adata.obs["ref_query"] == "qry").sum())
    print(f"Saved preprocessed AnnData to {out_path}")
    print(f"Used batch_key for HVG: {used_batch}")
    print(f"Cells: ref={n_ref}, qry={n_qry}; genes(HVG)={adata.n_vars}")


if __name__ == "__main__":
    main()
